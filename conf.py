# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Jafar Arash Mehr'
copyright = '2022, Jafar Arash Mehr'
author = 'Jafar Arash Mehr'

# The full version, including alpha/beta/rc tags
release = '2022'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_logo = "profile.jpg"
html_static_path = ['_static']
html_theme_options = {'logo_only': True,'display_version': False,}

html_css_files = ['custom.css',]

extensions = ['sphinx.ext.intersphinx','sphinx.ext.mathjax']

intersphinx_mapping = {'sphinx': ('https://www.sphinx-doc.org/en/master/', None),}
#html_theme_options = {'collapse_navigation':True} 
math_numfig = True
numfig = True
master_doc = 'index'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
