================================
User Guide for the Metal Target
================================

.. warning::
   The Metal backend is experimental and under active development. It does
   not ship with any release builds of LLVM tools today.

.. contents::
   :local:

Introduction
============

The Metal target turns LLVM IR into Apple `AIR <Apple Intermediate
Representation>`_ bitcode wrapped in a ``.metallib`` container. The container
is what Apple's Metal runtime loads with ``MTLLibrary``; AIR itself is a
typed-pointer LLVM bitcode dialect that the Metal GPU JIT lowers to native
M-series GPU code at load time.

There is no public Apple specification for either AIR or the ``.metallib``
container format. The layout documented in :doc:`MetallibFormat` was
reconstructed empirically from the output of Apple's Metal toolchain and is
byte-identical to it for the kernels exercised by the test suite.

Target Triples
==============

The Metal target is registered under ``Triple::air`` with the architecture
prefix ``air``. A canonical triple is::

    air64-apple-macosx<version>

Object format is ``MetalLib``. To build the target, add ``Metal`` to
``LLVM_EXPERIMENTAL_TARGETS_TO_BUILD`` when configuring CMake.

File Type Matrix
================

``llc -mtriple=air64-apple-macosx ...`` accepts two file types:

* ``-filetype=asm`` (default) emits the post-pipeline LLVM IR as text. AIR
  has no textual assembly form of its own; the text dump exists for
  diagnostic purposes only.
* ``-filetype=obj`` emits a complete ``.metallib`` container. The MC
  pipeline routes the bytes through ``MCMetalLibStreamer`` and
  ``MetalLibObjectWriter``; the actual container is built by
  ``MetalEmbedderPass`` inside the AIR writer.

Pass Pipeline
=============

The Metal-specific module pipeline (registered in
``llvm/lib/Target/Metal/MetalPassRegistry.def``) runs the following passes in
order:

============================================ ====================================
Pass name                                    Responsibility
============================================ ====================================
``metal-inline-non-kernel``                  Force-inline non-kernel functions
``metal-lower-fneg``                         ``fneg`` → ``0.0 - x``
``metal-nan-min-max``                        Match Metal's NaN semantics
``metal-lower-int-min-max``                  Replace integer ``min/max``
``metal-bitcast-zero-init``                  Constant-fold zero bitcasts
``metal-llvm-to-air-intrinsics``             LLVM intrinsics → AIR intrinsics
``metal-barrier-rename``                     ``llvm.amdgcn.barrier`` → AIR
``metal-lower-atomic-rmw``                   AS-aware atomic lowering
``metal-split-i64-shuffle``                  i64 shuffle decomposition
``metal-tg-global-dead-elim``                Drop unused TG globals
``metal-scalar-store-guard``                 Guard scalar device stores
``metal-tg-global-coalesce``                 Merge TG-global accesses
``metal-tg-barrier-insert``                  Insert TG barriers
``metal-widen-device-loads``                 Widen device loads/stores to float
``metal-device-loads-volatile``              Mark device loads volatile in loops
``metal-async-event-to-alloca``              Convert TG event globals to alloca
``metal-normalize-allocas``                  Normalize allocas to entry block
``metal-bfloat16-cast-decompose``            Decompose bfloat16 casts
``metal-scalar-buffer-packing``              Pack scalar params into one buffer
``metal-air-system-values``                  Materialise AIR system values
``metal-prepare``                            Final AIR-conformance fixups
============================================ ====================================

Address Spaces
==============

AIR uses the following address-space numbering, centralised in
``llvm/lib/Target/Metal/MetalAddressSpaces.h``:

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
* The MC layer is a stub: ``MetalMCAsmInfo``, ``MetalMCCodeEmitter`` and
  ``MetalAsmBackend`` exist only to satisfy the ``llc`` plumbing, and the
  encoding entry points are ``llvm_unreachable`` because no MC instructions
  are ever produced.
* **Baseline policy: Metal 4 / macOS 26+.** The backend targets the AIR
  bitcode version and metallib container format used by Metal 4 on
  macOS 26 only, and hardcodes ``air64_v28-apple-macosx26.0.0`` together
  with the corresponding container header bytes (see
  ``llvm/lib/Target/Metal/AIRWriter/MetallibWriter.h``). Older OS / Metal
  versions are out of scope for the initial submission. A future
  ``Triple`` sub-architecture set (``metal3.0``/``3.1``/``3.2``) could
  expose per-version emission, but Apple has not published the AIR
  bitcode version / metallib container header mapping for prior Metal
  versions, so each subarch would have to be reverse-engineered from the
  corresponding Xcode toolchain.
* The metallib container is monolithic; ``MetalLibObjectWriter`` asserts
  that exactly one non-empty section is present.
* ``f64`` in device memory is not supported (Metal Shading Language does not
  expose double precision on Apple GPUs).
* Threadgroup memory is capped at 32 KiB per the Metal Shading Language
  Specification.

Typed-Pointer Reconstruction
============================

LLVM 17+ uses opaque pointers in memory, but the Metal GPU JIT requires
typed ``POINTER`` records (code 8) in bitcode. The AIR writer recovers the
pointee type from use-sites and emits the typed records by way of
``PointeeTypeMap``. See :doc:`MetallibFormat` for the bitcode-level
specifics.

Intrinsic Mapping
=================

The mapping from generic LLVM intrinsics to AIR-specific names is defined
declaratively in ``llvm/lib/Target/Metal/MetalAIRIntrinsicMappings.td`` and
consumed by ``MetalLLVMToAIRIntrinsics``. New intrinsics are added by editing
the TableGen file; no C++ pass changes are required for routine additions.
