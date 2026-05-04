ORC-RT Documentation
====================

The ORC-RT documentation is written using the Sphinx documentation generator. It is
currently tested with Sphinx 5.3.0.

To build the documents into html configure ORC-RT with the following cmake options:

  * -DLLVM_ENABLE_SPHINX=ON
  * -DORC_RT_INCLUDE_DOCS=ON

After configuring ORC-RT with these options the make rule `docs-orc-rt-html`
should be available.
