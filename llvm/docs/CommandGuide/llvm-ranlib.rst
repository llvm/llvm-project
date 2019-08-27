llvm-ranlib - generates an archive index
========================================

.. program:: llvm-ranlib

SYNOPSIS
--------

:program:`llvm-ranlib` [*options*]

DESCRIPTION
-----------

:program:`llvm-ranlib` is an alias for the :doc:`llvm-ar <llvm-ar>` tool that
generates an index for an archive. It can be used as a replacement for GNU's
:program:`ranlib` tool.

Running :program:`llvm-ranlib` is equivalent to running ``llvm-ar s``.

SEE ALSO
--------

:manpage:`llvm-ar(1)`
