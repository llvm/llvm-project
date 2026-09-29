===========================================
Release Notes |release| |ReleaseNotesTitle|
===========================================

In Polly |version| the following important changes have been incorporated.

.. only:: PreRelease

  .. warning::
    These release notes are for the next release of Polly and describe
    the new features that have recently been committed to our development
    branch.

 * ScopInliner has been updated for the New Pass Manager.

 * Polly now is a monolithic pass split into phases.

 * Polly's support for the legacy pass manager has been removed.

 * The infrastructure around ScopPasses has been removed.

 * Polly now distributes the innermost point loop of a tile over its statements
   if this makes the loop of every statement parallel, so that the loop
   vectorizer can vectorize each of them. This mainly affects time-iterated
   stencils such as jacobi-1d, jacobi-2d and heat-3d. It is enabled by default
   and can be disabled with ``-polly-distribute-point-loops=false``.

