===========================================
Release Notes |release| |ReleaseNotesTitle|
===========================================

In Polly |version| the following important changes have been incorporated.

.. only:: PreRelease

  .. warning::
    These release notes are for the next release of Polly and describe
    the new features that have recently been committed to our development
    branch.

- The command line option -polly-opt-fusion has been removed. What the
  flag does was frequently misunderstood and is rarely useful. However,
  the functionality is still accessible using

  .. code-block:: console

    -polly-isl-arg=--no-schedule-serialize-sccs

- The command line option -polly-loopfusion-greedy has been added.
  This will aggressively try to fuse any loop regardless of
  profitability. The is what users might have expected what
  -polly-opt-fusion=max would do.

- Support for gfortran-generated code has been removed. This includes
  Fortran Array Descriptors (-polly-detect-fortran-arrays) and the
  -polly-rewrite-byref-params pass.
