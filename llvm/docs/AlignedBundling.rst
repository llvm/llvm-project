============================
Aligned Instruction Bundling
============================

Overview
========

*Aligned instruction bundling* partitions the instructions in a section into
fixed-size, naturally aligned groups called bundles, and guarantees that no
instruction ever crosses a bundle boundary. Consecutive instructions can also
be grouped so that the assembler guarantees the whole group resides within a
single bundle.

Bundling is a building block for software-based fault isolation (SFI) and
sandboxing schemes. Forcing every instruction to begin at one of a statically
known set of offsets gives the instruction stream a single canonical decoding:
control flow cannot jump into the middle of an instruction to manufacture a
different, unchecked instruction sequence. When combined with masking of
indirect branch targets to bundle-aligned addresses, this constrains all
control flow to a statically verifiable set of locations and instructions.
Bundling is used by the x86-64 implementation of :doc:`Lightweight Fault
Isolation (LFI) <LFI>`.

.. note::

   The current LLVM implementation supports bundling only for x86 ELF targets.

``.bundle_align_mode``
======================

::

   .bundle_align_mode abs-expr

Enables aligned bundle mode and sets the bundle size to ``2^abs-expr`` bytes,
where ``abs-expr`` is a power-of-two exponent between 0 and 30 (as for the
``.p2align`` directive). For example, ``.bundle_align_mode 5`` selects 32-byte
bundles.

While bundling is enabled, the assembler ensures that no single instruction
spans a boundary between two bundles. When an instruction would not fit in the
space remaining in the current bundle, that space is filled with no-op
instructions so the instruction starts at the beginning of the next bundle.

Enabling bundle mode also raises the alignment of every text section that
receives instructions to at least the bundle size.

Once enabled, bundle mode stays in effect for the rest of the file and its
bundle size is fixed.

``.bundle_lock`` and ``.bundle_unlock``
=======================================

::

   .bundle_lock [align_to_end]
   ...instructions...
   .bundle_unlock

A ``.bundle_lock`` / ``.bundle_unlock`` pair encloses a sequence of
instructions that must all be placed in a single bundle. The assembler inserts
padding before the sequence, if necessary, so that the entire group lands
within one bundle rather than straddling a boundary.

The enclosed sequence must fit within a single bundle -- it is an error if the
total size of the locked instructions exceeds the bundle size.

Both directives are only valid after bundle mode has been enabled with
``.bundle_align_mode``. A ``.bundle_unlock`` must be matched by a preceding
``.bundle_lock`` in the same section, and a section may not be switched while a
``.bundle_lock`` is open.

``align_to_end``
----------------

By default a locked group is padded at the front so that it starts far enough
into the bundle to fit. With the ``align_to_end`` option the group is instead
padded so that its last instruction ends exactly on a bundle boundary.

Padding
=======

By default, bundle padding is emitted as no-op instructions, and neither an
instruction nor a padding no-op is ever allowed to cross a bundle boundary.

Prefix padding (x86)
--------------------

On x86 the assembler can instead absorb some of the required padding into
neighboring instructions by prepending otherwise-ignored instruction prefixes
to them, avoiding standalone no-ops. This is controlled by:

.. option:: --x86-pad-max-prefix-size=<N>

   Maximum number of prefixes the assembler may add to an instruction for
   padding. ``0`` (the default) disables prefix padding, so only no-op
   instructions are used.
