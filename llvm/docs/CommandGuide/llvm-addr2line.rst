llvm-addr2line - a drop-in replacement for addr2line
====================================================

.. program:: llvm-addr2line

SYNOPSIS
--------

:program:`llvm-addr2line` [*options*]

DESCRIPTION
-----------

:program:`llvm-addr2line` is an alias for the :manpage:`llvm-symbolizer(1)`
tool with different defaults. The goal is to make it a drop-in replacement for
GNU's :program:`addr2line`.

Here are some of those differences:

-  Defaults not to print function names. Use `-f`_ to enable that.

-  Defaults not to demangle function names. Use `-C`_ to switch the
   demangling on.

-  Defaults not to print inlined frames. Use `-i`_ to show inlined
   frames for a source code location in an inlined function.

-  Uses `--output-style=GNU`_ by default.

-  Parses options from the environment variable ``LLVM_ADDR2LINE_OPTS``.

SEE ALSO
--------

:manpage:`llvm-symbolizer(1)`

.. _-f: llvm-symbolizer.html#llvm-symbolizer-opt-f
.. _-C: llvm-symbolizer.html#llvm-symbolizer-opt-c
.. _-i: llvm-symbolizer.html#llvm-symbolizer-opt-i
.. _--output-style=GNU: llvm-symbolizer.html#llvm-symbolizer-opt-output-style
