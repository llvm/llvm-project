===========================================
Release Notes |release| |ReleaseNotesTitle|
===========================================

In Polly |version| the following important changes have been incorporated.

.. only:: PreRelease

  .. warning::
    These release notes are for the next release of Polly and describe
    the new features that have recently been committed to our development
    branch.

 * Polly's options are passed to ``opt`` with ``-plugin-arg=Polly,<option>``,
   whether Polly is linked into ``opt`` or loaded as a plugin, e.g.
   ``opt -load-pass-plugin=LLVMPolly.so -plugin-arg=Polly,-polly-process-unprofitable``.
   When Polly is linked into ``opt``, the plain ``-polly-*`` options remain
   accepted for now.

 * The matrix multiplication optimization allocates packed arrays larger than
   ``-polly-pattern-matching-max-stack-array-size`` (1 MiB by default) on the
   heap instead of the stack. ``-1`` keeps all of them on the stack.

