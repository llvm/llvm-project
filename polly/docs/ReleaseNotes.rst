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

 * Polly can now separate the complete tiles of a tiled loop nest from the
   partial ones. The point loops of the complete tiles then have constant
   bounds, which lets the loop vectorizer vectorize them without a scalar
   epilogue. This is enabled with ``-polly-isolate-complete-tiles``, and is off
   by default because it increases code size. The options
   ``-polly-isolate-complete-tiles-2nd-level`` and
   ``-polly-isolate-complete-register-tiles`` do the same for the second level
   of tiling and for register tiling.

