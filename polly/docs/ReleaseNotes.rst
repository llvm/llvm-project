=============================
Release Notes 14.0 (upcoming)
=============================

In Polly 14 the following important changes have been incorporated.

.. warning::

  These release notes are for the next release of Polly and describe
  the new features that have recently been committed to our development
  branch.

- Change ...

 * The command line option -polly-opt-fusion has been removed. What the
   flag does was frequently misunderstood and is rarely useful. However,
   the functionality is still accessible using
```
    -polly-isl-arg=--no-schedule-serialize-sccs
```

 * The command line option -polly-loopfusion-greedy has been added.
   This will agressively try to fuse any loop regardless of
   profitability. The is what users might have expected what
   -polly-opt-fusion=max would do.
