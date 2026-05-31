================================
User Guide for the AIR Target
================================

.. warning::
   The AIR backend is experimental and under active development. It does
   not ship with any release builds of LLVM tools today.

.. contents::
   :local:

Introduction
============

The AIR target turns LLVM IR into Apple `AIR <Apple Intermediate
Representation>`_ bitcode wrapped in a ``.metallib`` container. The container
is what Apple's AIR runtime loads with ``MTLLibrary``; AIR itself is a
typed-pointer LLVM bitcode dialect that the AIR GPU JIT lowers to native
M-series GPU code at load time.

There is no public Apple specification for either AIR or the ``.metallib``
container format. The layout documented in :doc:`AIRLibFormat` was
reconstructed empirically from the output of Apple's AIR toolchain and is
byte-identical to it for the kernels exercised by the test suite.

Target Triples
==============

The AIR target is registered under ``Triple::air`` with the architecture
prefix ``air``. A canonical triple is::

    air64-apple-macosx<version>

Object format is ``AIRLib``. To build the target, add ``AIR`` to
``LLVM_EXPERIMENTAL_TARGETS_TO_BUILD`` when configuring CMake.

File Type Matrix
================

``llc -mtriple=air64-apple-macosx ...`` accepts two file types:

* ``-filetype=asm`` (default) emits the post-pipeline LLVM IR as text. AIR
  has no textual assembly form of its own; the text dump exists for
  diagnostic purposes only.
* ``-filetype=obj`` emits a complete ``.metallib`` container. The MC
  pipeline routes the bytes through ``MCAIRLibStreamer`` and
  ``AIRLibObjectWriter``; the actual container is built by
  ``AIREmbedderPass`` inside the AIR writer.

Pass Pipeline
=============

The AIR-specific module pipeline (registered in
``llvm/lib/Target/AIRPassRegistry.def``) runs the following passes in
order:

============================================ ====================================
Pass name                                    Responsibility
============================================ ====================================
``air-inline-non-kernel``                  Force-inline non-kernel functions
``air-lower-fneg``                         ``fneg`` → ``0.0 - x``
``air-nan-min-max``                        Match AIR's NaN semantics
``air-lower-int-min-max``                  Replace integer ``min/max``
``air-bitcast-zero-init``                  Constant-fold zero bitcasts
``air-llvm-to-air-intrinsics``             LLVM intrinsics → AIR intrinsics
``air-barrier-rename``                     ``llvm.amdgcn.barrier`` → AIR
``air-lower-atomic-rmw``                   AS-aware atomic lowering
``air-split-i64-shuffle``                  i64 shuffle decomposition
``air-tg-global-dead-elim``                Drop unused TG globals
``air-scalar-store-guard``                 Guard scalar device stores
``air-tg-global-coalesce``                 Merge TG-global accesses
``air-tg-barrier-insert``                  Insert TG barriers
``air-widen-device-loads``                 Widen device loads/stores to float
``air-device-loads-volatile``              Mark device loads volatile in loops
``air-async-event-to-alloca``              Convert TG event globals to alloca
``air-normalize-allocas``                  Normalize allocas to entry block
``air-bfloat16-cast-decompose``            Decompose bfloat16 casts
``air-scalar-buffer-packing``              Pack scalar params into one buffer
``air-air-system-values``                  Materialise AIR system values
``air-prepare``                            Final AIR-conformance fixups
============================================ ====================================

Address Spaces
==============

AIR uses the following address-space numbering, centralised in
``llvm/lib/Target/AIRAddressSpaces.h``:

============== ===== ====================
Name           Value Description
============== ===== ====================
``Default``    0     Generic / private
``Device``     1     Device buffer
``Constant``   2     Constant buffer
``Threadgroup`` 3     Threadgroup (TG)
============== ===== ====================

Known Limitations
=================

* AIR has no textual assembly form; ``-filetype=asm`` dumps post-pipeline IR.
* The MC layer is a stub: ``AIRMCAsmInfo``, ``AIRMCCodeEmitter`` and
  ``AIRAsmBackend`` exist only to satisfy the ``llc`` plumbing, and the
  encoding entry points are ``llvm_unreachable`` because no MC instructions
  are ever produced.
* **Baseline policy: AIR 4 / macOS 26+.** The backend targets the AIR
  bitcode version and metallib container format used by AIR 4 on
  macOS 26 only, and hardcodes ``air64_v28-apple-macosx26.0.0`` together
  with the corresponding container header bytes (see
  ``llvm/lib/Target/AIRWriter/AIRLibWriter.h``). Older OS / AIR
  versions are out of scope for the initial submission. A future
  ``Triple`` sub-architecture set (``metal3.0``/``3.1``/``3.2``) could
  expose per-version emission, but Apple has not published the AIR
  bitcode version / metallib container header mapping for prior AIR
  versions, so each subarch would have to be reverse-engineered from the
  corresponding Xcode toolchain.
* The metallib container is monolithic; ``AIRLibObjectWriter`` asserts
  that exactly one non-empty section is present.
* ``f64`` in device memory is not supported (AIR Shading Language does not
  expose double precision on Apple GPUs).
* Threadgroup memory is capped at 32 KiB per the AIR Shading Language
  Specification.

Typed-Pointer Reconstruction
============================

LLVM 17+ uses opaque pointers in memory, but the AIR GPU JIT requires
typed ``POINTER`` records (code 8) in bitcode. The AIR writer recovers the
pointee type from use-sites and emits the typed records by way of
``PointeeTypeMap``. See :doc:`AIRLibFormat` for the bitcode-level
specifics.

Intrinsic Mapping
=================

The mapping from generic LLVM intrinsics to AIR-specific names is defined
declaratively in ``llvm/lib/Target/AIRIntrinsicMappings.td`` and
consumed by ``LLVMToAIRIntrinsics``. New intrinsics are added by editing
the TableGen file; no C++ pass changes are required for routine additions.
