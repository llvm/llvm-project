OpenMP LLVM Documentation
==================

OpenMP LLVM's documentation is written in MyST Markdown, a lightweight
plaintext markup language (file extension `.md`). While the
Markdown documentation should be quite readable in source form, it
is mostly meant to be processed by the Sphinx documentation generation
system to create HTML pages which are hosted on <https://llvm.org/docs/> and
updated after every commit. Manpage output is also supported, see below.

If you instead would like to generate and view the HTML locally, install
Sphinx <http://sphinx-doc.org/> and then do:

    cd <build-dir>
    cmake -DLLVM_ENABLE_SPHINX=true -DSPHINX_OUTPUT_HTML=true -DCMAKE_MODULE_PATH=/path/to/llvm/cmake/modules <src-dir>
    make docs-openmp-html
    $BROWSER <build-dir>/docs/html/index.html

The mapping between reStructuredText files and generated documentation is
`docs/Foo.md` <-> `<build-dir>/projects/openmp/docs//html/Foo.html` <->
`https://openmp.llvm.org/docs/Foo.html`.

If you are interested in writing new documentation, you will want to read
`llvm/docs/SphinxQuickstartTemplate.md` which will get you writing
documentation very fast and includes examples of the most important MyST
Markdown syntax.

Manpage Output
===============

Building the manpages is similar to building the HTML documentation. The
primary difference is to use the `man` makefile target, instead of the
default (which is `html`). Sphinx then produces the man pages in the
directory `<build-dir>/docs/man/`.

    cd <build-dir>
    cmake -DLLVM_ENABLE_SPHINX=true -DSPHINX_OUTPUT_MAN=true <src-dir>
    make
    man -l >build-dir>/docs/man/FileCheck.1

The correspondence between .md files and man pages is
`docs/CommandGuide/Foo.md` <-> `<build-dir>/projects/openmp/docs//man/Foo.1`.
These .md files are also included during HTML generation so they are also
viewable online (as noted above) at e.g.
`https://openmp.llvm.org/docs/CommandGuide/Foo.html`.
