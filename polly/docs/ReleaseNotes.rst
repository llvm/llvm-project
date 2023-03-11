===========================================
Release Notes |release| |ReleaseNotesTitle|
===========================================

In Polly |version| the following important changes have been incorporated.

.. only:: PreRelease

  .. warning::
    These release notes are for the next release of Polly and describe
    the new features that have recently been committed to our development
    branch.


- Support for -polly-vectorizer=polly has been removed. Polly's internal
  vectorizer is not well maintained and is known to not work in some cases
  such as region ScopStmts. Unlike LLVM's LoopVectorize pass it also does
  not have a target-dependent cost heuristics, and we recommend using
  LoopVectorize instead of -polly-vectorizer=polly.

  In the future we hope that Polly can collaborate better with LoopVectorize,
  like Polly marking a loop is safe to vectorize with a specific simd width,
  instead of replicating its functionality.

- Polly-ACC has been removed.
