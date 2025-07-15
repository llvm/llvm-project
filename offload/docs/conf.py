# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Offload"
copyright = "2025, LLVM project"
author = "LLVM project"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

# -- C domain configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#c-config

c_maximum_signature_line_length = 60

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "llvm-theme"
html_theme_path = ["_themes"]
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
