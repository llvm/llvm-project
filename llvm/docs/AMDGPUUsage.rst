=============================
User Guide for AMDGPU Backend
=============================

.. contents::
   :local:

Introduction
============

The AMDGPU backend provides ISA code generation for AMD GPUs, starting with the
R600 family up until the current GCN families. It lives in the
``llvm/lib/Target/AMDGPU`` directory.

LLVM
====

.. _amdgpu-target-triples:

Target Triples
--------------

Use the ``clang -target <Architecture>-<Vendor>-<OS>-<Environment>`` option to
specify the target triple:

  .. table:: AMDGPU Architectures
     :name: amdgpu-architecture-table

     ============ ==============================================================
     Architecture Description
     ============ ==============================================================
     ``r600``     AMD GPUs HD2XXX-HD6XXX for graphics and compute shaders.
     ``amdgcn``   AMD GPUs GCN GFX6 onwards for graphics and compute shaders.
     ============ ==============================================================

  .. table:: AMDGPU Vendors
     :name: amdgpu-vendor-table

     ============ ==============================================================
     Vendor       Description
     ============ ==============================================================
     ``amd``      Can be used for all AMD GPU usage.
     ``mesa3d``   Can be used if the OS is ``mesa3d``.
     ============ ==============================================================

  .. table:: AMDGPU Operating Systems
     :name: amdgpu-os-table

     ============== ============================================================
     OS             Description
     ============== ============================================================
     *<empty>*      Defaults to the *unknown* OS.
     ``amdhsa``     Compute kernels executed on HSA [HSA]_ compatible runtimes
                    such as AMD's ROCm [AMD-ROCm]_.
     ``amdpal``     Graphic shaders and compute kernels executed on AMD PAL
                    runtime.
     ``mesa3d``     Graphic shaders and compute kernels executed on Mesa 3D
                    runtime.
     ============== ============================================================

  .. table:: AMDGPU Environments
     :name: amdgpu-environment-table

     ============ ==============================================================
     Environment  Description
     ============ ==============================================================
     *<empty>*    Default.
     ============ ==============================================================

.. _amdgpu-processors:

Processors
----------

Use the ``clang -mcpu <Processor>`` option to specify the AMDGPU processor. The
names from both the *Processor* and *Alternative Processor* can be used.

  .. table:: AMDGPU Processors
     :name: amdgpu-processor-table

     =========== =============== ============ ===== ================= ======= ======================
     Processor   Alternative     Target       dGPU/ Target            ROCm    Example
                 Processor       Triple       APU   Features          Support Products
                                 Architecture       Supported
                                                    [Default]
     =========== =============== ============ ===== ================= ======= ======================
     **Radeon HD 2000/3000 Series (R600)** [AMD-RADEON-HD-2000-3000]_
     -----------------------------------------------------------------------------------------------
     ``r600``                    ``r600``     dGPU
     ``r630``                    ``r600``     dGPU
     ``rs880``                   ``r600``     dGPU
     ``rv670``                   ``r600``     dGPU
     **Radeon HD 4000 Series (R700)** [AMD-RADEON-HD-4000]_
     -----------------------------------------------------------------------------------------------
     ``rv710``                   ``r600``     dGPU
     ``rv730``                   ``r600``     dGPU
     ``rv770``                   ``r600``     dGPU
     **Radeon HD 5000 Series (Evergreen)** [AMD-RADEON-HD-5000]_
     -----------------------------------------------------------------------------------------------
     ``cedar``                   ``r600``     dGPU
     ``cypress``                 ``r600``     dGPU
     ``juniper``                 ``r600``     dGPU
     ``redwood``                 ``r600``     dGPU
     ``sumo``                    ``r600``     dGPU
     **Radeon HD 6000 Series (Northern Islands)** [AMD-RADEON-HD-6000]_
     -----------------------------------------------------------------------------------------------
     ``barts``                   ``r600``     dGPU
     ``caicos``                  ``r600``     dGPU
     ``cayman``                  ``r600``     dGPU
     ``turks``                   ``r600``     dGPU
     **GCN GFX6 (Southern Islands (SI))** [AMD-GCN-GFX6]_
     -----------------------------------------------------------------------------------------------
     ``gfx600``  - ``tahiti``    ``amdgcn``   dGPU
     ``gfx601``  - ``hainan``    ``amdgcn``   dGPU
                 - ``oland``
                 - ``pitcairn``
                 - ``verde``
     **GCN GFX7 (Sea Islands (CI))** [AMD-GCN-GFX7]_
     -----------------------------------------------------------------------------------------------
     ``gfx700``  - ``kaveri``    ``amdgcn``   APU                             - A6-7000
                                                                              - A6 Pro-7050B
                                                                              - A8-7100
                                                                              - A8 Pro-7150B
                                                                              - A10-7300
                                                                              - A10 Pro-7350B
                                                                              - FX-7500
                                                                              - A8-7200P
                                                                              - A10-7400P
                                                                              - FX-7600P
     ``gfx701``  - ``hawaii``    ``amdgcn``   dGPU                    ROCm    - FirePro W8100
                                                                              - FirePro W9100
                                                                              - FirePro S9150
                                                                              - FirePro S9170
     ``gfx702``                  ``amdgcn``   dGPU                    ROCm    - Radeon R9 290
                                                                              - Radeon R9 290x
                                                                              - Radeon R390
                                                                              - Radeon R390x
     ``gfx703``  - ``kabini``    ``amdgcn``   APU                             - E1-2100
                 - ``mullins``                                                - E1-2200
                                                                              - E1-2500
                                                                              - E2-3000
                                                                              - E2-3800
                                                                              - A4-5000
                                                                              - A4-5100
                                                                              - A6-5200
                                                                              - A4 Pro-3340B
     ``gfx704``  - ``bonaire``   ``amdgcn``   dGPU                            - Radeon HD 7790
                                                                              - Radeon HD 8770
                                                                              - R7 260
                                                                              - R7 260X
     **GCN GFX8 (Volcanic Islands (VI))** [AMD-GCN-GFX8]_
     -----------------------------------------------------------------------------------------------
     ``gfx801``  - ``carrizo``   ``amdgcn``   APU   - xnack                   - A6-8500P
                                                      [on]                    - Pro A6-8500B
                                                                              - A8-8600P
                                                                              - Pro A8-8600B
                                                                              - FX-8800P
                                                                              - Pro A12-8800B
     \                           ``amdgcn``   APU   - xnack           ROCm    - A10-8700P
                                                      [on]                    - Pro A10-8700B
                                                                              - A10-8780P
     \                           ``amdgcn``   APU   - xnack                   - A10-9600P
                                                      [on]                    - A10-9630P
                                                                              - A12-9700P
                                                                              - A12-9730P
                                                                              - FX-9800P
                                                                              - FX-9830P
     \                           ``amdgcn``   APU   - xnack                   - E2-9010
                                                      [on]                    - A6-9210
                                                                              - A9-9410
     ``gfx802``  - ``iceland``   ``amdgcn``   dGPU  - xnack           ROCm    - FirePro S7150
                 - ``tonga``                          [off]                   - FirePro S7100
                                                                              - FirePro W7100
                                                                              - Radeon R285
                                                                              - Radeon R9 380
                                                                              - Radeon R9 385
                                                                              - Mobile FirePro
                                                                                M7170
     ``gfx803``  - ``fiji``      ``amdgcn``   dGPU  - xnack           ROCm    - Radeon R9 Nano
                                                      [off]                   - Radeon R9 Fury
                                                                              - Radeon R9 FuryX
                                                                              - Radeon Pro Duo
                                                                              - FirePro S9300x2
                                                                              - Radeon Instinct MI8
     \           - ``polaris10`` ``amdgcn``   dGPU  - xnack           ROCm    - Radeon RX 470
                                                      [off]                   - Radeon RX 480
                                                                              - Radeon Instinct MI6
     \           - ``polaris11`` ``amdgcn``   dGPU  - xnack           ROCm    - Radeon RX 460
                                                      [off]
     ``gfx810``  - ``stoney``    ``amdgcn``   APU   - xnack
                                                      [on]
     **GCN GFX9** [AMD-GCN-GFX9]_
     -----------------------------------------------------------------------------------------------
     ``gfx900``                  ``amdgcn``   dGPU  - xnack           ROCm    - Radeon Vega
                                                      [off]                     Frontier Edition
                                                                              - Radeon RX Vega 56
                                                                              - Radeon RX Vega 64
                                                                              - Radeon RX Vega 64
                                                                                Liquid
                                                                              - Radeon Instinct MI25
     ``gfx902``                  ``amdgcn``   APU   - xnack                   - Ryzen 3 2200G
                                                      [on]                    - Ryzen 5 2400G
     ``gfx904``                  ``amdgcn``   dGPU  - xnack                   *TBA*
                                                      [off]
                                                                              .. TODO::
                                                                                 Add product
                                                                                 names.
     ``gfx906``                  ``amdgcn``   dGPU  - xnack                   - Radeon Instinct MI50
                                                      [off]                   - Radeon Instinct MI60
     ``gfx908``                  ``amdgcn``   dGPU  - xnack                   *TBA*
                                                      [off]
                                                      sram-ecc
                                                      [on]
     ``gfx909``                  ``amdgcn``   APU   - xnack                   *TBA* (Raven Ridge 2)
                                                      [on]
                                                                              .. TODO::
                                                                                 Add product
                                                                                 names.
     **GCN GFX10** [AMD-GCN-GFX10]_
     -----------------------------------------------------------------------------------------------
     ``gfx1010``                 ``amdgcn``   dGPU  - xnack                   *TBA*
                                                      [off]
                                                    - wavefrontsize64
                                                      [off]
                                                    - cumode
                                                      [off]
                                                                              .. TODO::
                                                                                 Add product
                                                                                 names.
     ``gfx1011``                 ``amdgcn``   dGPU  - xnack                   *TBA*
                                                      [off]
                                                    - wavefrontsize64
                                                      [off]
                                                    - cumode
                                                      [off]
                                                                              .. TODO::
                                                                                 Add product
                                                                                 names.
     ``gfx1012``                 ``amdgcn``   dGPU  - xnack                   *TBA*
                                                      [off]
                                                    - wavefrontsize64
                                                      [off]
                                                    - cumode
                                                      [off]
                                                                              .. TODO::
                                                                                 Add product
                                                                                 names.
     =========== =============== ============ ===== ================= ======= ======================

.. _amdgpu-target-features:

Target Features
---------------

Target features control how code is generated to support certain
processor specific features. Not all target features are supported by
all processors. The runtime must ensure that the features supported by
the device used to execute the code match the features enabled when
generating the code. A mismatch of features may result in incorrect
execution, or a reduction in performance.

The target features supported by each processor, and the default value
used if not specified explicitly, is listed in
:ref:`amdgpu-processor-table`.

Use the ``clang -m[no-]<TargetFeature>`` option to specify the AMDGPU
target features.

For example:

``-mxnack``
  Enable the ``xnack`` feature.
``-mno-xnack``
  Disable the ``xnack`` feature.

  .. table:: AMDGPU Target Features
     :name: amdgpu-target-feature-table

     ====================== ==================================================
     Target Feature         Description
     ====================== ==================================================
     -m[no-]xnack           Enable/disable generating code that has
                            memory clauses that are compatible with
                            having XNACK replay enabled.

                            This is used for demand paging and page
                            migration. If XNACK replay is enabled in
                            the device, then if a page fault occurs
                            the code may execute incorrectly if the
                            ``xnack`` feature is not enabled. Executing
                            code that has the feature enabled on a
                            device that does not have XNACK replay
                            enabled will execute correctly but may
                            be less performant than code with the
                            feature disabled.

     -m[no-]sram-ecc        Enable/disable generating code that assumes SRAM
                            ECC is enabled/disabled.

     -m[no-]wavefrontsize64 Control the default wavefront size used when
                            generating code for kernels. When disabled
                            native wavefront size 32 is used, when enabled
                            wavefront size 64 is used.

     -m[no-]cumode          Control the default wavefront execution mode used
                            when generating code for kernels. When disabled
                            native WGP wavefront execution mode is used,
                            when enabled CU wavefront execution mode is used
                            (see :ref:`amdgpu-amdhsa-memory-model`).
     ====================== ==================================================

.. _amdgpu-address-spaces:

Address Spaces
--------------

The AMDGPU architecture supports a number of memory address spaces. The address
space names use the OpenCL standard names, with some additions.

The AMDGPU address spaces correspond to target architecture specific LLVM
address space numbers used in LLVM IR.

The AMDGPU address spaces are described in
:ref:`amdgpu-address-spaces-table`. Only 64-bit process address spaces are
supported for the ``amdgcn`` target.

  .. table:: AMDGPU Address Spaces
     :name: amdgpu-address-spaces-table

     ================================= =============== =========== ================ ======= ============================
     ..                                                                                     64-Bit Process Address Space
     --------------------------------- --------------- ----------- ---------------- ------------------------------------
     Address Space Name                LLVM IR Address HSA Segment Hardware         Address NULL Value
                                       Space Number    Name        Name             Size
     ================================= =============== =========== ================ ======= ============================
     Generic                           0               flat        flat             64      0x0000000000000000
     Global                            1               global      global           64      0x0000000000000000
     Region                            2               N/A         GDS              32      *not implemented for AMDHSA*
     Local                             3               group       LDS              32      0xFFFFFFFF
     Constant                          4               constant    *same as global* 64      0x0000000000000000
     Private                           5               private     scratch          32      0x00000000
     Constant 32-bit                   6               *TODO*
     Buffer Fat Pointer (experimental) 7               *TODO*
     ================================= =============== =========== ================ ======= ============================

**Generic**
  The generic address space uses the hardware flat address support available in
  GFX7-GFX10. This uses two fixed ranges of virtual addresses (the private and
  local apertures), that are outside the range of addressable global memory, to
  map from a flat address to a private or local address.

  FLAT instructions can take a flat address and access global, private
  (scratch), and group (LDS) memory depending on if the address is within one
  of the aperture ranges. Flat access to scratch requires hardware aperture
  setup and setup in the kernel prologue (see
  :ref:`amdgpu-amdhsa-kernel-prolog-flat-scratch`). Flat access to LDS requires
  hardware aperture setup and M0 (GFX7-GFX8) register setup (see
  :ref:`amdgpu-amdhsa-kernel-prolog-m0`).

  To convert between a private or group address space address (termed a segment
  address) and a flat address the base address of the corresponding aperture
  can be used. For GFX7-GFX8 these are available in the
  :ref:`amdgpu-amdhsa-hsa-aql-queue` the address of which can be obtained with
  Queue Ptr SGPR (see :ref:`amdgpu-amdhsa-initial-kernel-execution-state`). For
  GFX9-GFX10 the aperture base addresses are directly available as inline
  constant registers ``SRC_SHARED_BASE/LIMIT`` and ``SRC_PRIVATE_BASE/LIMIT``.
  In 64-bit address mode the aperture sizes are 2^32 bytes and the base is
  aligned to 2^32 which makes it easier to convert from flat to segment or
  segment to flat.

  A global address space address has the same value when used as a flat address
  so no conversion is needed.

**Global and Constant**
  The global and constant address spaces both use global virtual addresses,
  which are the same virtual address space used by the CPU. However, some
  virtual addresses may only be accessible to the CPU, some only accessible
  by the GPU, and some by both.

  Using the constant address space indicates that the data will not change
  during the execution of the kernel. This allows scalar read instructions to
  be used. The vector and scalar L1 caches are invalidated of volatile data
  before each kernel dispatch execution to allow constant memory to change
  values between kernel dispatches.

**Region**
  The region address space uses the hardware Global Data Store (GDS). All
  wavefronts executing on the same device will access the same memory for any
  given region address. However, the same region address accessed by wavefronts
  executing on different devices will access different memory. It is higher
  performance than global memory. It is allocated by the runtime. The data
  store (DS) instructions can be used to access it.

**Local**
  The local address space uses the hardware Local Data Store (LDS) which is
  automatically allocated when the hardware creates the wavefronts of a
  work-group, and freed when all the wavefronts of a work-group have
  terminated. All wavefronts belonging to the same work-group will access the
  same memory for any given local address. However, the same local address
  accessed by wavefronts belonging to different work-groups will access
  different memory. It is higher performance than global memory. The data store
  (DS) instructions can be used to access it.

**Private**
  The private address space uses the hardware scratch memory support which
  automatically allocates memory when it creates a wavefront and frees it when
  a wavefronts terminates. The memory accessed by a lane of a wavefront for any
  given private address will be different to the memory accessed by another lane
  of the same or different wavefront for the same private address.

  If a kernel dispatch uses scratch, then the hardware allocates memory from a
  pool of backing memory allocated by the runtime for each wavefront. The lanes
  of the wavefront access this using dword (4 byte) interleaving. The mapping
  used from private address to backing memory address is:

    ``wavefront-scratch-base +
    ((private-address / 4) * wavefront-size * 4) +
    (wavefront-lane-id * 4) + (private-address % 4)``

  If each lane of a wavefront accesses the same private address, the
  interleaving results in adjacent dwords being accessed and hence requires
  fewer cache lines to be fetched.

  There are different ways that the wavefront scratch base address is
  determined by a wavefront (see
  :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

  Scratch memory can be accessed in an interleaved manner using buffer
  instructions with the scratch buffer descriptor and per wavefront scratch
  offset, by the scratch instructions, or by flat instructions. Multi-dword
  access is not supported except by flat and scratch instructions in
  GFX9-GFX10.

**Constant 32-bit**
  *TODO*

**Buffer Fat Pointer**
  The buffer fat pointer is an experimental address space that is currently
  unsupported in the backend. It exposes a non-integral pointer that is in
  the future intended to support the modelling of 128-bit buffer descriptors
  plus a 32-bit offset into the buffer (in total encapsulating a 160-bit
  *pointer*), allowing normal LLVM load/store/atomic operations to be used to
  model the buffer descriptors used heavily in graphics workloads targeting
  the backend.

.. _amdgpu-memory-scopes:

Memory Scopes
-------------

This section provides LLVM memory synchronization scopes supported by the AMDGPU
backend memory model when the target triple OS is ``amdhsa`` (see
:ref:`amdgpu-amdhsa-memory-model` and :ref:`amdgpu-target-triples`).

The memory model supported is based on the HSA memory model [HSA]_ which is
based in turn on HRF-indirect with scope inclusion [HRF]_. The happens-before
relation is transitive over the synchronizes-with relation independent of scope
and synchronizes-with allows the memory scope instances to be inclusive (see
table :ref:`amdgpu-amdhsa-llvm-sync-scopes-table`).

This is different to the OpenCL [OpenCL]_ memory model which does not have scope
inclusion and requires the memory scopes to exactly match. However, this
is conservatively correct for OpenCL.

  .. table:: AMDHSA LLVM Sync Scopes
     :name: amdgpu-amdhsa-llvm-sync-scopes-table

     ======================= ===================================================
     LLVM Sync Scope         Description
     ======================= ===================================================
     *none*                  The default: ``system``.

                             Synchronizes with, and participates in modification
                             and seq_cst total orderings with, other operations
                             (except image operations) for all address spaces
                             (except private, or generic that accesses private)
                             provided the other operation's sync scope is:

                             - ``system``.
                             - ``agent`` and executed by a thread on the same
                               agent.
                             - ``workgroup`` and executed by a thread in the
                               same work-group.
                             - ``wavefront`` and executed by a thread in the
                               same wavefront.

     ``agent``               Synchronizes with, and participates in modification
                             and seq_cst total orderings with, other operations
                             (except image operations) for all address spaces
                             (except private, or generic that accesses private)
                             provided the other operation's sync scope is:

                             - ``system`` or ``agent`` and executed by a thread
                               on the same agent.
                             - ``workgroup`` and executed by a thread in the
                               same work-group.
                             - ``wavefront`` and executed by a thread in the
                               same wavefront.

     ``workgroup``           Synchronizes with, and participates in modification
                             and seq_cst total orderings with, other operations
                             (except image operations) for all address spaces
                             (except private, or generic that accesses private)
                             provided the other operation's sync scope is:

                             - ``system``, ``agent`` or ``workgroup`` and
                               executed by a thread in the same work-group.
                             - ``wavefront`` and executed by a thread in the
                               same wavefront.

     ``wavefront``           Synchronizes with, and participates in modification
                             and seq_cst total orderings with, other operations
                             (except image operations) for all address spaces
                             (except private, or generic that accesses private)
                             provided the other operation's sync scope is:

                             - ``system``, ``agent``, ``workgroup`` or
                               ``wavefront`` and executed by a thread in the
                               same wavefront.

     ``singlethread``        Only synchronizes with and participates in
                             modification and seq_cst total orderings with,
                             other operations (except image operations) running
                             in the same thread for all address spaces (for
                             example, in signal handlers).

     ``one-as``              Same as ``system`` but only synchronizes with other
                             operations within the same address space.

     ``agent-one-as``        Same as ``agent`` but only synchronizes with other
                             operations within the same address space.

     ``workgroup-one-as``    Same as ``workgroup`` but only synchronizes with
                             other operations within the same address space.

     ``wavefront-one-as``    Same as ``wavefront`` but only synchronizes with
                             other operations within the same address space.

     ``singlethread-one-as`` Same as ``singlethread`` but only synchronizes with
                             other operations within the same address space.
     ======================= ===================================================

LLVM IR Intrinsics
------------------

The AMDGPU backend implements the following LLVM IR intrinsics.

*This section is WIP.*

.. TODO::

   List AMDGPU intrinsics.

LLVM IR Attributes
------------------

The AMDGPU backend supports the following LLVM IR attributes.

  .. table:: AMDGPU LLVM IR Attributes
     :name: amdgpu-llvm-ir-attributes-table

     ======================================= ==========================================================
     LLVM Attribute                          Description
     ======================================= ==========================================================
     "amdgpu-flat-work-group-size"="min,max" Specify the minimum and maximum flat work group sizes that
                                             will be specified when the kernel is dispatched. Generated
                                             by the ``amdgpu_flat_work_group_size`` CLANG attribute [CLANG-ATTR]_.
     "amdgpu-implicitarg-num-bytes"="n"      Number of kernel argument bytes to add to the kernel
                                             argument block size for the implicit arguments. This
                                             varies by OS and language (for OpenCL see
                                             :ref:`opencl-kernel-implicit-arguments-appended-for-amdhsa-os-table`).
     "amdgpu-num-sgpr"="n"                   Specifies the number of SGPRs to use. Generated by
                                             the ``amdgpu_num_sgpr`` CLANG attribute [CLANG-ATTR]_.
     "amdgpu-num-vgpr"="n"                   Specifies the number of VGPRs to use. Generated by the
                                             ``amdgpu_num_vgpr`` CLANG attribute [CLANG-ATTR]_.
     "amdgpu-waves-per-eu"="m,n"             Specify the minimum and maximum number of waves per
                                             execution unit. Generated by the ``amdgpu_waves_per_eu``
                                             CLANG attribute [CLANG-ATTR]_.
     "amdgpu-ieee" true/false.               Specify whether the function expects the IEEE field of the
                                             mode register to be set on entry. Overrides the default for
                                             the calling convention.
     "amdgpu-dx10-clamp" true/false.         Specify whether the function expects the DX10_CLAMP field of
                                             the mode register to be set on entry. Overrides the default
                                             for the calling convention.
     ======================================= ==========================================================

.. _amdgpu-elf-code-object:

ELF Code Object
===============

The AMDGPU backend generates a standard ELF [ELF]_ relocatable code object that
can be linked by ``lld`` to produce a standard ELF shared code object which can
be loaded and executed on an AMDGPU target.

.. _amdgpu-elf-header:

Header
------

The AMDGPU backend uses the following ELF header:

  .. table:: AMDGPU ELF Header
     :name: amdgpu-elf-header-table

     ========================== ===============================
     Field                      Value
     ========================== ===============================
     ``e_ident[EI_CLASS]``      ``ELFCLASS64``
     ``e_ident[EI_DATA]``       ``ELFDATA2LSB``
     ``e_ident[EI_OSABI]``      - ``ELFOSABI_NONE``
                                - ``ELFOSABI_AMDGPU_HSA``
                                - ``ELFOSABI_AMDGPU_PAL``
                                - ``ELFOSABI_AMDGPU_MESA3D``
     ``e_ident[EI_ABIVERSION]`` - ``ELFABIVERSION_AMDGPU_HSA``
                                - ``ELFABIVERSION_AMDGPU_PAL``
                                - ``ELFABIVERSION_AMDGPU_MESA3D``
     ``e_type``                 - ``ET_REL``
                                - ``ET_DYN``
     ``e_machine``              ``EM_AMDGPU``
     ``e_entry``                0
     ``e_flags``                See :ref:`amdgpu-elf-header-e_flags-table`
     ========================== ===============================

..

  .. table:: AMDGPU ELF Header Enumeration Values
     :name: amdgpu-elf-header-enumeration-values-table

     =============================== =====
     Name                            Value
     =============================== =====
     ``EM_AMDGPU``                   224
     ``ELFOSABI_NONE``               0
     ``ELFOSABI_AMDGPU_HSA``         64
     ``ELFOSABI_AMDGPU_PAL``         65
     ``ELFOSABI_AMDGPU_MESA3D``      66
     ``ELFABIVERSION_AMDGPU_HSA``    1
     ``ELFABIVERSION_AMDGPU_PAL``    0
     ``ELFABIVERSION_AMDGPU_MESA3D`` 0
     =============================== =====

``e_ident[EI_CLASS]``
  The ELF class is:

  * ``ELFCLASS32`` for ``r600`` architecture.

  * ``ELFCLASS64`` for ``amdgcn`` architecture which only supports 64-bit
    process address space applications.

``e_ident[EI_DATA]``
  All AMDGPU targets use ``ELFDATA2LSB`` for little-endian byte ordering.

``e_ident[EI_OSABI]``
  One of the following AMDGPU target architecture specific OS ABIs
  (see :ref:`amdgpu-os-table`):

  * ``ELFOSABI_NONE`` for *unknown* OS.

  * ``ELFOSABI_AMDGPU_HSA`` for ``amdhsa`` OS.

  * ``ELFOSABI_AMDGPU_PAL`` for ``amdpal`` OS.

  * ``ELFOSABI_AMDGPU_MESA3D`` for ``mesa3D`` OS.

``e_ident[EI_ABIVERSION]``
  The ABI version of the AMDGPU target architecture specific OS ABI to which the code
  object conforms:

  * ``ELFABIVERSION_AMDGPU_HSA`` is used to specify the version of AMD HSA
    runtime ABI.

  * ``ELFABIVERSION_AMDGPU_PAL`` is used to specify the version of AMD PAL
    runtime ABI.

  * ``ELFABIVERSION_AMDGPU_MESA3D`` is used to specify the version of AMD MESA
    3D runtime ABI.

``e_type``
  Can be one of the following values:


  ``ET_REL``
    The type produced by the AMDGPU backend compiler as it is relocatable code
    object.

  ``ET_DYN``
    The type produced by the linker as it is a shared code object.

  The AMD HSA runtime loader requires a ``ET_DYN`` code object.

``e_machine``
  The value ``EM_AMDGPU`` is used for the machine for all processors supported
  by the ``r600`` and ``amdgcn`` architectures (see
  :ref:`amdgpu-processor-table`). The specific processor is specified in the
  ``EF_AMDGPU_MACH`` bit field of the ``e_flags`` (see
  :ref:`amdgpu-elf-header-e_flags-table`).

``e_entry``
  The entry point is 0 as the entry points for individual kernels must be
  selected in order to invoke them through AQL packets.

``e_flags``
  The AMDGPU backend uses the following ELF header flags:

  .. table:: AMDGPU ELF Header ``e_flags``
     :name: amdgpu-elf-header-e_flags-table

     ================================= ========== =============================
     Name                              Value      Description
     ================================= ========== =============================
     **AMDGPU Processor Flag**                    See :ref:`amdgpu-processor-table`.
     -------------------------------------------- -----------------------------
     ``EF_AMDGPU_MACH``                0x000000ff AMDGPU processor selection
                                                  mask for
                                                  ``EF_AMDGPU_MACH_xxx`` values
                                                  defined in
                                                  :ref:`amdgpu-ef-amdgpu-mach-table`.
     ``EF_AMDGPU_XNACK``               0x00000100 Indicates if the ``xnack``
                                                  target feature is
                                                  enabled for all code
                                                  contained in the code object.
                                                  If the processor
                                                  does not support the
                                                  ``xnack`` target
                                                  feature then must
                                                  be 0.
                                                  See
                                                  :ref:`amdgpu-target-features`.
     ``EF_AMDGPU_SRAM_ECC``            0x00000200 Indicates if the ``sram-ecc``
                                                  target feature is
                                                  enabled for all code
                                                  contained in the code object.
                                                  If the processor
                                                  does not support the
                                                  ``sram-ecc`` target
                                                  feature then must
                                                  be 0.
                                                  See
                                                  :ref:`amdgpu-target-features`.
     ================================= ========== =============================

  .. table:: AMDGPU ``EF_AMDGPU_MACH`` Values
     :name: amdgpu-ef-amdgpu-mach-table

     ================================= ========== =============================
     Name                              Value      Description (see
                                                  :ref:`amdgpu-processor-table`)
     ================================= ========== =============================
     ``EF_AMDGPU_MACH_NONE``           0x000      *not specified*
     ``EF_AMDGPU_MACH_R600_R600``      0x001      ``r600``
     ``EF_AMDGPU_MACH_R600_R630``      0x002      ``r630``
     ``EF_AMDGPU_MACH_R600_RS880``     0x003      ``rs880``
     ``EF_AMDGPU_MACH_R600_RV670``     0x004      ``rv670``
     ``EF_AMDGPU_MACH_R600_RV710``     0x005      ``rv710``
     ``EF_AMDGPU_MACH_R600_RV730``     0x006      ``rv730``
     ``EF_AMDGPU_MACH_R600_RV770``     0x007      ``rv770``
     ``EF_AMDGPU_MACH_R600_CEDAR``     0x008      ``cedar``
     ``EF_AMDGPU_MACH_R600_CYPRESS``   0x009      ``cypress``
     ``EF_AMDGPU_MACH_R600_JUNIPER``   0x00a      ``juniper``
     ``EF_AMDGPU_MACH_R600_REDWOOD``   0x00b      ``redwood``
     ``EF_AMDGPU_MACH_R600_SUMO``      0x00c      ``sumo``
     ``EF_AMDGPU_MACH_R600_BARTS``     0x00d      ``barts``
     ``EF_AMDGPU_MACH_R600_CAICOS``    0x00e      ``caicos``
     ``EF_AMDGPU_MACH_R600_CAYMAN``    0x00f      ``cayman``
     ``EF_AMDGPU_MACH_R600_TURKS``     0x010      ``turks``
     *reserved*                        0x011 -    Reserved for ``r600``
                                       0x01f      architecture processors.
     ``EF_AMDGPU_MACH_AMDGCN_GFX600``  0x020      ``gfx600``
     ``EF_AMDGPU_MACH_AMDGCN_GFX601``  0x021      ``gfx601``
     ``EF_AMDGPU_MACH_AMDGCN_GFX700``  0x022      ``gfx700``
     ``EF_AMDGPU_MACH_AMDGCN_GFX701``  0x023      ``gfx701``
     ``EF_AMDGPU_MACH_AMDGCN_GFX702``  0x024      ``gfx702``
     ``EF_AMDGPU_MACH_AMDGCN_GFX703``  0x025      ``gfx703``
     ``EF_AMDGPU_MACH_AMDGCN_GFX704``  0x026      ``gfx704``
     *reserved*                        0x027      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX801``  0x028      ``gfx801``
     ``EF_AMDGPU_MACH_AMDGCN_GFX802``  0x029      ``gfx802``
     ``EF_AMDGPU_MACH_AMDGCN_GFX803``  0x02a      ``gfx803``
     ``EF_AMDGPU_MACH_AMDGCN_GFX810``  0x02b      ``gfx810``
     ``EF_AMDGPU_MACH_AMDGCN_GFX900``  0x02c      ``gfx900``
     ``EF_AMDGPU_MACH_AMDGCN_GFX902``  0x02d      ``gfx902``
     ``EF_AMDGPU_MACH_AMDGCN_GFX904``  0x02e      ``gfx904``
     ``EF_AMDGPU_MACH_AMDGCN_GFX906``  0x02f      ``gfx906``
     ``EF_AMDGPU_MACH_AMDGCN_GFX908``  0x030      ``gfx908``
     ``EF_AMDGPU_MACH_AMDGCN_GFX909``  0x031      ``gfx909``
     *reserved*                        0x032      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX1010`` 0x033      ``gfx1010``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1011`` 0x034      ``gfx1011``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1012`` 0x035      ``gfx1012``
     ================================= ========== =============================

Sections
--------

An AMDGPU target ELF code object has the standard ELF sections which include:

  .. table:: AMDGPU ELF Sections
     :name: amdgpu-elf-sections-table

     ================== ================ =================================
     Name               Type             Attributes
     ================== ================ =================================
     ``.bss``           ``SHT_NOBITS``   ``SHF_ALLOC`` + ``SHF_WRITE``
     ``.data``          ``SHT_PROGBITS`` ``SHF_ALLOC`` + ``SHF_WRITE``
     ``.debug_``\ *\**  ``SHT_PROGBITS`` *none*
     ``.dynamic``       ``SHT_DYNAMIC``  ``SHF_ALLOC``
     ``.dynstr``        ``SHT_PROGBITS`` ``SHF_ALLOC``
     ``.dynsym``        ``SHT_PROGBITS`` ``SHF_ALLOC``
     ``.got``           ``SHT_PROGBITS`` ``SHF_ALLOC`` + ``SHF_WRITE``
     ``.hash``          ``SHT_HASH``     ``SHF_ALLOC``
     ``.note``          ``SHT_NOTE``     *none*
     ``.rela``\ *name*  ``SHT_RELA``     *none*
     ``.rela.dyn``      ``SHT_RELA``     *none*
     ``.rodata``        ``SHT_PROGBITS`` ``SHF_ALLOC``
     ``.shstrtab``      ``SHT_STRTAB``   *none*
     ``.strtab``        ``SHT_STRTAB``   *none*
     ``.symtab``        ``SHT_SYMTAB``   *none*
     ``.text``          ``SHT_PROGBITS`` ``SHF_ALLOC`` + ``SHF_EXECINSTR``
     ================== ================ =================================

These sections have their standard meanings (see [ELF]_) and are only generated
if needed.

``.debug``\ *\**
  The standard DWARF sections. See :ref:`amdgpu-dwarf-debug-information` for
  information on the DWARF produced by the AMDGPU backend.

``.dynamic``, ``.dynstr``, ``.dynsym``, ``.hash``
  The standard sections used by a dynamic loader.

``.note``
  See :ref:`amdgpu-note-records` for the note records supported by the AMDGPU
  backend.

``.rela``\ *name*, ``.rela.dyn``
  For relocatable code objects, *name* is the name of the section that the
  relocation records apply. For example, ``.rela.text`` is the section name for
  relocation records associated with the ``.text`` section.

  For linked shared code objects, ``.rela.dyn`` contains all the relocation
  records from each of the relocatable code object's ``.rela``\ *name* sections.

  See :ref:`amdgpu-relocation-records` for the relocation records supported by
  the AMDGPU backend.

``.text``
  The executable machine code for the kernels and functions they call. Generated
  as position independent code. See :ref:`amdgpu-code-conventions` for
  information on conventions used in the isa generation.

.. _amdgpu-note-records:

Note Records
------------

The AMDGPU backend code object contains ELF note records in the ``.note``
section. The set of generated notes and their semantics depend on the code
object version; see :ref:`amdgpu-note-records-v2` and
:ref:`amdgpu-note-records-v3`.

As required by ``ELFCLASS32`` and ``ELFCLASS64``, minimal zero-byte padding
must be generated after the ``name`` field to ensure the ``desc`` field is 4
byte aligned. In addition, minimal zero-byte padding must be generated to
ensure the ``desc`` field size is a multiple of 4 bytes. The ``sh_addralign``
field of the ``.note`` section must be at least 4 to indicate at least 8 byte
alignment.

.. _amdgpu-note-records-v2:

Code Object V2 Note Records (-mattr=-code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Code Object V2 is not the default code object version emitted by
  this version of LLVM. For a description of the notes generated with the
  default configuration (Code Object V3) see :ref:`amdgpu-note-records-v3`.

The AMDGPU backend code object uses the following ELF note record in the
``.note`` section when compiling for Code Object V2 (-mattr=-code-object-v3).

Additional note records may be present, but any which are not documented here
are deprecated and should not be used.

  .. table:: AMDGPU Code Object V2 ELF Note Records
     :name: amdgpu-elf-note-records-table-v2

     ===== ============================== ======================================
     Name  Type                           Description
     ===== ============================== ======================================
     "AMD" ``NT_AMD_AMDGPU_HSA_METADATA`` <metadata null terminated string>
     ===== ============================== ======================================

..

  .. table:: AMDGPU Code Object V2 ELF Note Record Enumeration Values
     :name: amdgpu-elf-note-record-enumeration-values-table-v2

     ============================== =====
     Name                           Value
     ============================== =====
     *reserved*                       0-9
     ``NT_AMD_AMDGPU_HSA_METADATA``    10
     *reserved*                        11
     ============================== =====

``NT_AMD_AMDGPU_HSA_METADATA``
  Specifies extensible metadata associated with the code objects executed on HSA
  [HSA]_ compatible runtimes such as AMD's ROCm [AMD-ROCm]_. It is required when
  the target triple OS is ``amdhsa`` (see :ref:`amdgpu-target-triples`). See
  :ref:`amdgpu-amdhsa-code-object-metadata-v2` for the syntax of the code
  object metadata string.

.. _amdgpu-note-records-v3:

Code Object V3 Note Records (-mattr=+code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMDGPU backend code object uses the following ELF note record in the
``.note`` section when compiling for Code Object V3 (-mattr=+code-object-v3).

Additional note records may be present, but any which are not documented here
are deprecated and should not be used.

  .. table:: AMDGPU Code Object V3 ELF Note Records
     :name: amdgpu-elf-note-records-table-v3

     ======== ============================== ======================================
     Name     Type                           Description
     ======== ============================== ======================================
     "AMDGPU" ``NT_AMDGPU_METADATA``         Metadata in Message Pack [MsgPack]_
                                             binary format.
     ======== ============================== ======================================

..

  .. table:: AMDGPU Code Object V3 ELF Note Record Enumeration Values
     :name: amdgpu-elf-note-record-enumeration-values-table-v3

     ============================== =====
     Name                           Value
     ============================== =====
     *reserved*                     0-31
     ``NT_AMDGPU_METADATA``         32
     ============================== =====

``NT_AMDGPU_METADATA``
  Specifies extensible metadata associated with an AMDGPU code
  object. It is encoded as a map in the Message Pack [MsgPack]_ binary
  data format. See :ref:`amdgpu-amdhsa-code-object-metadata-v3` for the
  map keys defined for the ``amdhsa`` OS.

.. _amdgpu-symbols:

Symbols
-------

Symbols include the following:

  .. table:: AMDGPU ELF Symbols
     :name: amdgpu-elf-symbols-table

     ===================== ================== ================ ==================
     Name                  Type               Section          Description
     ===================== ================== ================ ==================
     *link-name*           ``STT_OBJECT``     - ``.data``      Global variable
                                              - ``.rodata``
                                              - ``.bss``
     *link-name*\ ``.kd``  ``STT_OBJECT``     - ``.rodata``    Kernel descriptor
     *link-name*           ``STT_FUNC``       - ``.text``      Kernel entry point
     *link-name*           ``STT_OBJECT``     - SHN_AMDGPU_LDS Global variable in LDS
     ===================== ================== ================ ==================

Global variable
  Global variables both used and defined by the compilation unit.

  If the symbol is defined in the compilation unit then it is allocated in the
  appropriate section according to if it has initialized data or is readonly.

  If the symbol is external then its section is ``STN_UNDEF`` and the loader
  will resolve relocations using the definition provided by another code object
  or explicitly defined by the runtime.

  If the symbol resides in local/group memory (LDS) then its section is the
  special processor specific section name ``SHN_AMDGPU_LDS``, and the
  ``st_value`` field describes alignment requirements as it does for common
  symbols.

  .. TODO::

     Add description of linked shared object symbols. Seems undefined symbols
     are marked as STT_NOTYPE.

Kernel descriptor
  Every HSA kernel has an associated kernel descriptor. It is the address of the
  kernel descriptor that is used in the AQL dispatch packet used to invoke the
  kernel, not the kernel entry point. The layout of the HSA kernel descriptor is
  defined in :ref:`amdgpu-amdhsa-kernel-descriptor`.

Kernel entry point
  Every HSA kernel also has a symbol for its machine code entry point.

.. _amdgpu-relocation-records:

Relocation Records
------------------

AMDGPU backend generates ``Elf64_Rela`` relocation records. Supported
relocatable fields are:

``word32``
  This specifies a 32-bit field occupying 4 bytes with arbitrary byte
  alignment. These values use the same byte order as other word values in the
  AMDGPU architecture.

``word64``
  This specifies a 64-bit field occupying 8 bytes with arbitrary byte
  alignment. These values use the same byte order as other word values in the
  AMDGPU architecture.

Following notations are used for specifying relocation calculations:

**A**
  Represents the addend used to compute the value of the relocatable field.

**G**
  Represents the offset into the global offset table at which the relocation
  entry's symbol will reside during execution.

**GOT**
  Represents the address of the global offset table.

**P**
  Represents the place (section offset for ``et_rel`` or address for ``et_dyn``)
  of the storage unit being relocated (computed using ``r_offset``).

**S**
  Represents the value of the symbol whose index resides in the relocation
  entry. Relocations not using this must specify a symbol index of
  ``STN_UNDEF``.

**B**
  Represents the base address of a loaded executable or shared object which is
  the difference between the ELF address and the actual load address.
  Relocations using this are only valid in executable or shared objects.

The following relocation types are supported:

  .. table:: AMDGPU ELF Relocation Records
     :name: amdgpu-elf-relocation-records-table

     ========================== ======= =====  ==========  ==============================
     Relocation Type            Kind    Value  Field       Calculation
     ========================== ======= =====  ==========  ==============================
     ``R_AMDGPU_NONE``                  0      *none*      *none*
     ``R_AMDGPU_ABS32_LO``      Static, 1      ``word32``  (S + A) & 0xFFFFFFFF
                                Dynamic
     ``R_AMDGPU_ABS32_HI``      Static, 2      ``word32``  (S + A) >> 32
                                Dynamic
     ``R_AMDGPU_ABS64``         Static, 3      ``word64``  S + A
                                Dynamic
     ``R_AMDGPU_REL32``         Static  4      ``word32``  S + A - P
     ``R_AMDGPU_REL64``         Static  5      ``word64``  S + A - P
     ``R_AMDGPU_ABS32``         Static, 6      ``word32``  S + A
                                Dynamic
     ``R_AMDGPU_GOTPCREL``      Static  7      ``word32``  G + GOT + A - P
     ``R_AMDGPU_GOTPCREL32_LO`` Static  8      ``word32``  (G + GOT + A - P) & 0xFFFFFFFF
     ``R_AMDGPU_GOTPCREL32_HI`` Static  9      ``word32``  (G + GOT + A - P) >> 32
     ``R_AMDGPU_REL32_LO``      Static  10     ``word32``  (S + A - P) & 0xFFFFFFFF
     ``R_AMDGPU_REL32_HI``      Static  11     ``word32``  (S + A - P) >> 32
     *reserved*                         12
     ``R_AMDGPU_RELATIVE64``    Dynamic 13     ``word64``  B + A
     ========================== ======= =====  ==========  ==============================

``R_AMDGPU_ABS32_LO`` and ``R_AMDGPU_ABS32_HI`` are only supported by
the ``mesa3d`` OS, which does not support ``R_AMDGPU_ABS64``.

There is no current OS loader support for 32-bit programs and so
``R_AMDGPU_ABS32`` is not used.

.. _amdgpu-dwarf-6-proposal-for-heterogeneous-debugging:

DWARF Version 6 Proposal For Heterogeneous Debugging
====================================================

.. warning::

   This section describes a **provisional proposal** for DWARF Version 6
   [DWARF]_ to support heterogeneous debugging. It is not currently fully
   implemented and is subject to change.

.. note::

  This section proposes a set of backwards compatible extensions to DWARF
  Version 5 [DWARF]_ for consideration of inclusion into a future DWARF Version
  6 standard to support heterogeneous debugging.

  The remainder of this note provides motivation for each proposed feature in
  terms of heterogeneous debugging on commercially available AMD GPU hardware
  (AMDGPU). However, the proposal is intended to be vendor and architecture
  neutral. It is believed to apply to other heterogeous hardware devices
  including GPUs, DSPs, FPGAs, and other specialized hardware. These
  collectively include similar characteristics and requirements as AMDGPUs.
  Parts of the proposal can also apply to traditional CPU hardware that supports
  large vector registers. Compilers can map source languages and extensions that
  describe large scale parallel execution onto the lanes of the vector
  registers. This is common in programming languages used in ML and HPC. The
  proposal also includes improved support for optimized code on any
  architecture. Some of the generalizations may also benefit other issues that
  have been raised.

  The proposal has evolved though collaboration with many individuals and active
  prototyping within the gdb debugger and LLVM compiler. Input has also been
  very much appreciated from the developers working on the Totalview debugger
  and gcc compiler.

  The AMDGPU has several features that require additional DWARF functionality in
  order to support optimized code.

  AMDGPU optimized code may spill vector registers to non-global address space
  memory, and this spilling may be done only for lanes that are active on entry
  to the subprogram. To support this, a location description that can be created
  as a masked select is required. See ``DW_OP_LLVM_select_bit_piece``.

  Since the active lane mask may be held in a register, a way to get the value
  of a register on entry to a subprogram is required. To support this an
  operation that returns the caller value of a register as specified by the Call
  Frame Information (CFI) is required. See ``DW_OP_LLVM_call_frame_entry_reg``
  and :ref:`amdgpu-dwarf-call-frame-information`.

  Current DWARF uses an empty expression to indicate an undefined location
  description. Since the masked select composite location description operation
  takes more than one location description, it is necessary to have an explicit
  way to specify an undefined location description. Otherwise it is not possible
  to specify that a particular one of the input location descriptions is
  undefined. See ``DW_OP_LLVM_undefined``.

  CFI describes restoring callee saved registers that are spilled. Currently CFI
  only allows a location description that is a register, memory address, or
  implicit location description. AMDGPU optimized code may spill scalar
  registers into portions of vector registers. This requires extending CFI to
  allow any location description. See
  :ref:`amdgpu-dwarf-call-frame-information`.

  The vector registers of the AMDGPU are represented as their full wavefront
  size, meaning the wavefront size times the dword size. This reflects the
  actual hardware and allows the compiler to generate DWARF for languages that
  map a thread to the complete wavefront. It also allows more efficient DWARF to
  be generated to describe the CFI as only a single expression is required for
  the whole vector register, rather than a separate expression for each lane's
  dword of the vector register. It also allows the compiler to produce DWARF
  that indexes the vector register if it spills scalar registers into portions
  of a vector registers.

  Since DWARF stack value entries have a base type and AMDGPU registers are a
  vector of dwords, the ability to specify that a base type is a vector is
  required. See ``DW_AT_LLVM_vector_size``.

  If the source language is mapped onto the AMDGPU wavefronts in a SIMT manner,
  then the variable DWARF location expressions must compute the location for a
  single lane of the wavefront. Therefore, a DWARF operation is required to
  denote the current lane, much like ``DW_OP_push_object_address`` denotes the
  current object. The ``DW_OP_*piece`` operations only allow literal indices.
  Therefore, a way to use a computed offset of an arbitrary location description
  (such as a vector register) is required. See ``DW_OP_LLVM_push_lane``,
  ``DW_OP_LLVM_offset``, ``DW_OP_LLVM_offset_constu``, and
  ``DW_OP_LLVM_bit_offset``.

  If the source language is mapped onto the AMDGPU wavefronts in a SIMT manner
  the compiler can use the AMDGPU execution mask register to control which lanes
  are active. To describe the conceptual location of non-active lanes a DWARF
  expression is needed that can compute a per lane PC. For efficiency, this is
  done for the wavefront as a whole. This expression benefits by having a masked
  select composite location description operation. This requires an attribute
  for source location of each lane. The AMDGPU may update the execution mask for
  whole wavefront operations and so needs an attribute that computes the current
  active lane mask. See ``DW_OP_LLVM_select_bit_piece``, ``DW_OP_LLVM_extend``,
  ``DW_AT_LLVM_lane_pc``, and ``DW_AT_LLVM_active_lane``.

  AMDGPU needs to be able to describe addresses that are in different kinds of
  memory. Optimized code may need to describe a variable that resides in pieces
  that are in different kinds of storage which may include parts of registers,
  memory that is in a mixture of memory kinds, implicit values, or be undefined.
  DWARF has the concept of segment addresses. However, the segment cannot be
  specified within a DWARF expression, which is only able to specify the offset
  portion of a segment address. The segment index is only provided by the entity
  that specifies the DWARF expression. Therefore, the segment index is a
  property that can only be put on complete objects, such as a variable. That
  makes it only suitable for describing an entity (such as variable or
  subprogram code) that is in a single kind of memory. Therefore, AMDGPU uses
  the DWARF concept of address spaces. For example, a variable may be allocated
  in a register that is partially spilled to the call stack which is in the
  private address space, and partially spilled to the local address space.

  DWARF uses the concept of an address in many expression operations but does not
  define how it relates to address spaces. For example,
  ``DW_OP_push_object_address`` pushes the address of an object. Other contexts
  implicitly push an address on the stack before evaluating an expression. For
  example, the ``DW_AT_use_location`` attribute of the
  ``DW_TAG_ptr_to_member_type``. The expression that uses the address needs to
  do so in a general way and not need to be dependent on the address space of
  the address. For example, a pointer to member value may want to be applied to
  an object that may reside in any address space.

  The number of registers and the cost of memory operations is much higher for
  AMDGPU than a typical CPU. The compiler attempts to optimize whole variables
  and arrays into registers. Currently DWARF only allows
  ``DW_OP_push_object_address`` and related operations to work with a global
  memory location. To support AMDGPU optimized code it is required to generalize
  DWARF to allow any location description to be used. This allows registers, or
  composite location descriptions that may be a mixture of memory, registers, or
  even implicit values.

  DWARF Version 5 does not allow location descriptions to be entries on the
  DWARF stack. They can only be the final result of the evaluation of a DWARF
  expression. However, by allowing a location description to be a first-class
  entry on the DWARF stack it becomes possible to compose expressions containing
  both values and location descriptions naturally. It allows objects to be
  located in any kind of memory address space, in registers, be implicit values,
  be undefined, or a composite of any of these. By extending DWARF carefully,
  all existing DWARF expressions can retain their current semantic meaning.
  DWARF has implicit conversions that convert from a value that represents an
  address in the default address space to a memory location description. This
  can be extended to allow a default address space memory location description
  to be implicitly converted back to its address value. This allows all DWARF
  Version 5 expressions to retain their same meaning, while adding the ability
  to explicitly create memory location descriptions in non-default address
  spaces and generalizing the power of composite location descriptions to any
  kind of location description. See :ref:`amdgpu-dwarf-operation-expressions`.

  To allow composition of composite location descriptions, an explicit operation
  that indicates the end of the definition of a composite location description
  is required. This can be implied if the end of a DWARF expression is reached,
  allowing current DWARF expressions to remain legal. See
  ``DW_OP_LLVM_piece_end``.

  The ``DW_OP_plus`` and ``DW_OP_minus`` can be defined to operate on a memory
  location description in the default target architecture specific address space
  and a generic type value to produce an updated memory location description.
  This allows them to continue to be used to offset an address. To generalize
  offsetting to any location description, including location descriptions that
  describe when bytes are in registers, are implicit, or a composite of these,
  the ``DW_OP_LLVM_offset``, ``DW_OP_LLVM_offset_constu`` and
  ``DW_OP_LLVM_bit_offset`` operations are added. These do not perform wrapping
  which would be hard to define for location descriptions of non-memory kinds.
  This allows ``DW_OP_push_object_address`` to push a location description that
  may be in a register, or be an implicit value, and the DWARF expression of
  ``DW_TAG_ptr_to_member_type`` can contain ``DW_OP_LLVM_offset`` to offset
  within it. ``DW_OP_LLVM_bit_offset`` generalizes DWARF to work with bit fields
  which is not possible in DWARF Version 5.

  The DWARF ``DW_OP_xderef*`` operations allow a value to be converted into an
  address of a specified address space which is then read. But it provides no
  way to create a memory location description for an address in the non-default
  address space. For example, AMDGPU variables can be allocated in the local
  address space at a fixed address. It is required to have an operation to
  create an address in a specific address space that can be used to define the
  location description of the variable. Defining this operation to produce a
  location description allows the size of addresses in an address space to be
  larger than the generic type. See ``DW_OP_LLVM_form_aspace_address``.

  If the ``DW_OP_LLVM_form_aspace_address`` operation had to produce a value
  that can be implicitly converted to a memory location description, then it
  would be limited to the size of the generic type which matches the size of the
  default address space. Its value would be unspecified and likely not match any
  value in the actual program. By making the result a location description, it
  allows a consumer great freedom in how it implements it. The implicit
  conversion back to a value can be limited only to the default address space to
  maintain compatibility with DWARF Version 5. For other address spaces the
  producer can use the new operations that explicitly specify the address space.

  ``DW_OP_breg*`` treats the register as containing an address in the default
  address space. It is required to be able to specify the address space of the
  register value. See ``DW_OP_LLVM_aspace_bregx``.

  Similarly, ``DW_OP_implicit_pointer`` treats its implicit pointer value as
  being in the default address space. It is required to be able to specify the
  address space of the pointer value. See
  ``DW_OP_LLVM_aspace_implicit_pointer``.

  Almost all uses of addresses in DWARF are limited to defining location
  descriptions, or to be dereferenced to read memory. The exception is
  ``DW_CFA_val_offset`` which uses the address to set the value of a register.
  By defining the CFA DWARF expression as being a memory location description,
  it can maintain what address space it is, and that can be used to convert the
  offset address back to an address in that address space. See
  :ref:`amdgpu-dwarf-call-frame-information`.

  This approach allows all existing DWARF to have the identical semantics. It
  allows the compiler to explicitly specify the address space it is using. For
  example, a compiler could choose to access private memory in a swizzled manner
  when mapping a source language to a wavefront in a SIMT manner, or to access
  it in an unswizzled manner if mapping the same language with the wavefront
  being the thread. It also allows the compiler to mix the address space it uses
  to access private memory. For example, for SIMT it can still spill entire
  vector registers in an unswizzled manner, while using a swizzled private
  memory for SIMT variable access. This approach allows memory location
  descriptions for different address spaces to be combined using the regular
  ``DW_OP_*piece`` operations.

  Location descriptions are an abstraction of storage, they give freedom to the
  consumer on how to implement them. They allow the address space to encode lane
  information so they can be used to read memory with only the memory
  description and no extra arguments. The same set of operations can operate on
  locations independent of their kind of storage. The ``DW_OP_deref*`` therefore
  can be used on any storage kind. ``DW_OP_xderef*`` is unnecessary except to
  become a more compact way to convert a non-default address space address
  followed by dereferencing it.

  In DWARF Version 5 a location description is defined as a single location
  description or a location list. A location list is defined as either
  effectively an undefined location description or as one or more single
  location descriptions to describe an object with multiple places. The
  ``DW_OP_push_object_address`` and ``DW_OP_call*`` operations can put a
  location description on the stack. Furthermore, debugger information entry
  attributes such as ``DW_AT_data_member_location``, ``DW_AT_use_location``, and
  ``DW_AT_vtable_elem_location`` are defined as pushing a location description
  on the expression stack before evaluating the expression. However, DWARF
  Version 5 only allows the stack to contain values and so only a single memory
  address can be on the stack which makes these incapable of handling location
  descriptions with multiple places, or places other than memory. Since this
  proposal allows the stack to contain location descriptions, the operations are
  generalized to support location descriptions that can have multiple places.
  This is backwards compatible with DWARF Version 5 and allows objects with
  multiple places to be supported. For example, the expression that describes
  how to access the field of an object can be evaluated with a location
  description that has multiple places and will result in a location description
  with multiple places as expected. With this change, the separate DWARF Version
  5 sections that described DWARF expressions and location lists have been
  unified into a single section that describes DWARF expressions in general.
  This unification seems to be a natural consequence and a necessity of allowing
  location descriptions to be part of the evaluation stack.

  For those familiar with the definition of location descriptions in DWARF
  Version 5, the definition in this proposal is presented differently, but does
  in fact define the same concept with the same fundamental semantics. However,
  it does so in a way that allows the concept to extend to support address
  spaces, bit addressing, the ability for composite location descriptions to be
  composed of any kind of location description, and the ability to support
  objects located at multiple places. Collectively these changes expand the set
  of processors that can be supported and improves support for optimized code.

  Several approaches were considered, and the one proposed appears to be the
  cleanest and offers the greatest improvement of DWARF's ability to support
  optimized code. Examining the gdb debugger and LLVM compiler, it appears only
  to require modest changes as they both already have to support general use of
  location descriptions. It is anticipated that will also be the case for other
  debuggers and compilers.

  As an experiment, gdb was modified to evaluate DWARF Version 5 expressions
  with location descriptions as stack entries and implicit conversions. All gdb
  tests have passed, except one that turned out to be an invalid test by DWARF
  Version 5 rules. The code in gdb actually became simpler as all evaluation was
  on the stack and there was no longer a need to maintain a separate structure
  for the location description result. This gives confidence of the backwards
  compatibility.

  Since the AMDGPU supports languages such as OpenCL, there is a need to define
  source language address classes so they can be used in a consistent way by
  consumers. It would also be desirable to add support for using them in
  defining language types rather than the current target architecture specific
  address spaces. See :ref:`amdgpu-dwarf-segment_addresses`.

  A ``DW_AT_LLVM_augmentation`` attribute is added to a compilation unit
  debugger information entry to indicate that there is additional target
  architecture specific information in the debugging information entries of that
  compilation unit. This allows a consumer to know what extensions are present
  in the debugger information entries as is possible with the augmentation
  string of other sections. The format that should be used for the augmentation
  string in the lookup by name table and CFI Common Information Entry is also
  recommended to allow a consumer to parse the string when it contains
  information from multiple vendors.

  The AMDGPU supports programming languages that include online compilation
  where the source text may be created at runtime. Therefore, a way to embed the
  source text in the debug information is required. For example, the OpenCL
  language runtime supports online compilation. See
  :ref:`amdgpu-dwarf-line-number-information`.

  Support to allow MD5 checksums to be optionally present in the line table is
  added. This allows linking together compilation units where some have MD5
  checksums and some do not. In DWARF Version 5 the file timestamp and file size
  can be optional, but if the MD5 checksum is present it must be valid for all
  files. See :ref:`amdgpu-dwarf-line-number-information`.

  Support is added for the HIP programming language which is supported by the
  AMDGPU. See :ref:`amdgpu-dwarf-language-names`.

  The following sections provide the definitions for the additional operations,
  as well as clarifying how existing expression operations, CFI operations, and
  attributes behave with respect to generalized location descriptions that
  support address spaces and location descriptions that support multiple places.
  It has been defined such that it is backwards compatible with DWARF Version 5.
  The definitions are intended to fully define well-formed DWARF in a consistent
  style based on the DWARF Version 5 specification. Non-normative text is shown
  in *italics*.

  The names for the new operations, attributes, and constants include "\
  ``LLVM``\ " and are encoded with vendor specific codes so this proposal can be
  implemented as an LLVM vendor extension to DWARF Version 5. If accepted these
  names would not include the "\ ``LLVM``\ " and would not use encodings in the
  vendor range.

  The proposal is organized to follow the section ordering of DWARF Version 5.
  It includes notes to indicate the corresponding DWARF Version 5 sections to
  which they pertain. Other notes describe additional changes that may be worth
  considering, and to raise questions.

General Description
-------------------

Attribute Types
~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 2.2 and Table 2.2.

The following table provides the additional attributes. See
:ref:`amdgpu-dwarf-debugging-information-entry-attributes`.

.. table:: Attribute names
   :name: amdgpu-dwarf-attribute-names-table

   =========================== ====================================
   Attribute                   Usage
   =========================== ====================================
   ``DW_AT_LLVM_active_lane``  SIMD or SIMT active lanes
   ``DW_AT_LLVM_augmentation`` Compilation unit augmentation string
   ``DW_AT_LLVM_lane_pc``      SIMD or SIMT lane program location
   ``DW_AT_LLVM_lanes``        SIMD or SIMT thread lane count
   ``DW_AT_LLVM_vector_size``  Base type vector size
   =========================== ====================================

.. _amdgpu-dwarf-expressions:

DWARF Expressions
~~~~~~~~~~~~~~~~~

.. note::

  This section, and its nested sections, replaces DWARF Version 5 section 2.5 and
  section 2.6. The new proposed DWARF expression operations are defined as well
  as clarifying the extensions to already existing DWARF Version 5 operations. It is
  based on the text of the existing DWARF Version 5 standard.

DWARF expressions describe how to compute a value or specify a location.

*The evaluation of a DWARF expression can provide the location of an object, the
value of an array bound, the length of a dynamic string, the desired value
itself, and so on.*

The evaluation of a DWARF expression can either result in a value or a location
description:

*value*

  A value has a type and a literal value. It can represent a literal value of
  any supported base type of the target architecture. The base type specifies
  the size and encoding of the literal value.

  .. note::

    It may be desirable to add an implicit pointer base type encoding. It would
    be used for the type of the value that is produced when the ``DW_OP_deref*``
    operation retrieves the full contents of an implicit pointer location
    storage created by the ``DW_OP_implicit_pointer`` or
    ``DW_OP_LLVM_aspace_implicit_pointer`` operations. The literal value would
    record the debugging information entry and byte dispacement specified by the
    associated ``DW_OP_implicit_pointer`` or
    ``DW_OP_LLVM_aspace_implicit_pointer`` operations.

  Instead of a base type, a value can have a distinguished generic type, which
  is an integral type that has the size of an address in the target architecture
  default address space and unspecified signedness.

  *The generic type is the same as the unspecified type used for stack
  operations defined in DWARF Version 4 and before.*

  An integral type is a base type that has an encoding of ``DW_ATE_signed``,
  ``DW_ATE_signed_char``, ``DW_ATE_unsigned``, ``DW_ATE_unsigned_char``,
  ``DW_ATE_boolean``, or any target architecture defined integral encoding in
  the inclusive range ``DW_ATE_lo_user`` to ``DW_ATE_hi_user``.

  .. note::

    It is unclear if ``DW_ATE_address`` is an integral type. Gdb does not seem
    to consider it as integral.

*location description*

  *Debugging information must provide consumers a way to find the location of
  program variables, determine the bounds of dynamic arrays and strings, and
  possibly to find the base address of a subprogram’s stack frame or the return
  address of a subprogram. Furthermore, to meet the needs of recent computer
  architectures and optimization techniques, debugging information must be able
  to describe the location of an object whose location changes over the object’s
  lifetime, and may reside at multiple locations simultaneously during parts of
  an object's lifetime.*

  Information about the location of program objects is provided by location
  descriptions.

  Location descriptions can consist of one or more single location descriptions.

  A single location description specifies the location storage that holds a
  program object and a position within the location storage where the program
  object starts. The position within the location storage is expressed as a bit
  offset relative to the start of the location storage.

  A location storage is a linear stream of bits that can hold values. Each
  location storage has a size in bits and can be accessed using a zero-based bit
  offset. The ordering of bits within a location storage uses the bit numbering
  and direction conventions that are appropriate to the current language on the
  target architecture.

  There are five kinds of location storage:

  *memory location storage*
    Corresponds to the target architecture memory address spaces.

  *register location storage*
    Corresponds to the target architecture registers.

  *implicit location storage*
    Corresponds to fixed values that can only be read.

  *undefined location storage*
    Indicates no value is available and therefore cannot be read or written.

  *composite location storage*
    Allows a mixture of these where some bits come from one location storage and
    some from another location storage, or from disjoint parts of the same
    location storage.

  .. note::

    It may be better to add an implicit pointer location storage kind used by
    the ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_aspace_implicit_pointer``
    operations. It would specify the debugger information entry and byte offset
    provided by the operations.

  *Location descriptions are a language independent representation of addressing
  rules. They are created using DWARF operation expressions of arbitrary
  complexity. They can be the result of evaluting a debugger information entry
  attribute that specifies an operation expression. In this usage they can
  describe the location of an object as long as its lifetime is either static or
  the same as the lexical block (see DWARF Version 5 section 3.5) that owns it,
  and it does not move during its lifetime. They can be the result of evaluating
  a debugger information entry attribute that specifies a location list
  expression. In this usage they can describe the location of an object that has
  a limited lifetime, changes its location during its lifetime, or has multiple
  locations over part or all of its lifetime.*

  If a location description has more than one single location description, the
  DWARF expression is ill-formed if the object value held in each single
  location description's position within the associated location storage is not
  the same value, except for the parts of the value that are uninitialized.

  *A location description that has more than one single location description can
  only be created by a location list expression that has overlapping program
  location ranges, or certain expression operations that act on a location
  description that has more than one single location description. There are no
  operation expression operations that can directly create a location
  description with more than one single location description.*

  *A location description with more than one single location description can be
  used to describe objects that reside in more than one piece of storage at the
  same time. An object may have more than one location as a result of
  optimization. For example, a value that is only read may be promoted from
  memory to a register for some region of code, but later code may revert to
  reading the value from memory as the register may be used for other purposes.
  For the code region where the value is in a register, any change to the object
  value must be made in both the register and the memory so both regions of code
  will read the updated value.*

  *A consumer of a location description with more than one single location
  description can read the object's value from any of the single location
  descriptions (since they all refer to location storage that has the same
  value), but must write any changed value to all the single location
  descriptions.*

A DWARF expression can either be encoded as a operation expression (see
:ref:`amdgpu-dwarf-operation-expressions`), or as a location list expression
(see :ref:`amdgpu-dwarf-location-list-expressions`).

A DWARF expression is evaluated in the context of:

*A current subprogram*
  This may be used in the evaluation of register access operations to support
  virtual unwinding of the call stack (see
  :ref:`amdgpu-dwarf-call-frame-information`).

*A current program location*
  This may be used in the evaluation of location list expressions to select
  amongst multiple program location ranges. It should be the program location
  corresponding to the current subprogram. If the current subprogram was reached
  by virtual call stack unwinding, then the program location will correspond to
  the associated call site.

*An initial stack*
  This is a list of values or location descriptions that will be pushed on the
  operation expression evaluation stack in the order provided before evaluation
  of an operation expression starts.

  Some debugger information entries have attributes that evaluate their DWARF
  expression value with initial stack entries. In all other cases the initial
  stack is empty.

When a DWARF expression is evaluated, it may be specified whether a value or
location description is required as the result kind.

If a result kind is specified, and the result of the evaluation does not match
the specified result kind, then the implicit conversions described in
:ref:`amdgpu-dwarf-memory-location-description-operations` are performed if
valid. Otherwise, the DWARF expression is ill-formed.

.. _amdgpu-dwarf-operation-expressions:

DWARF Operation Expressions
+++++++++++++++++++++++++++

An operation expression is comprised of a stream of operations, each consisting
of an opcode followed by zero or more operands. The number of operands is
implied by the opcode.

Operations represent a postfix operation on a simple stack machine. Each stack
entry can hold either a value or a location description. Operations can act on
entries on the stack, including adding entries and removing entries. If the kind
of a stack entry does not match the kind required by the operation and is not
implicitly convertible to the required kind (see
:ref:`amdgpu-dwarf-memory-location-description-operations`), then the DWARF
operation expression is ill-formed.

Evaluation of an operation expression starts with an empty stack on which the
entries from the initial stack provided by the context are pushed in the order
provided. Then the operations are evaluated, starting with the first operation
of the stream, until one past the last operation of the stream is reached. The
result of the evaluation is:

* If evaluation of the DWARF expression requires a location description, then:

  * If the stack is empty, the result is a location description with one
    undefined location description.

    *This rule is for backwards compatibility with DWARF Version 5 which has no
    explicit operation to create an undefined location description, and uses an
    empty operation expression for this purpose.*

  * If the top stack entry is a location description, or can be converted
    to one, then the result is that, possibly converted, location description.
    Any other entries on the stack are discarded.

  * Otherwise the DWARF expression is ill-formed.

    .. note::

      Could define this case as returning an implicit location description as
      if the ``DW_OP_implicit`` operation is performed.

* If evaluation of the DWARF expression requires a value, then:

  * If the top stack entry is a value, or can be converted to one, then the
    result is that, possibly converted, value. Any other entries on the stack
    are discarded.

  * Otherwise the DWARF expression is ill-formed.

* If evaluation of the DWARF expression does not specify if a value or location
  description is required, then:

  * If the stack is empty, the result is a location description with one
    undefined location description.

    *This rule is for backwards compatibility with DWARF Version 5 which has no
    explicit operation to create an undefined location description, and uses an
    empty operation expression for this purpose.*

    .. note::

      This rule is consistent with the rule above for when a location
      description is requested. However, gdb appears to report this as an error
      and no gdb tests appear to cause an empty stack for this case.

  * Otherwise, the top stack entry is returned. Any other entries on the stack
    are discarded.

An operation expression is encoded as a byte block with some form of prefix that
specifies the byte count. It can be used:

* as the value of a debugging information entry attribute that is encoded using
  class ``exprloc`` (see DWARF Version 5 section 7.5.5),

* as the operand to certain operation expression operations,

* as the operand to certain call frame information operations (see
  :ref:`amdgpu-dwarf-call-frame-information`),

* and in location list entries (see
  :ref:`amdgpu-dwarf-location-list-expressions`).

.. _amdgpu-dwarf-stack-operations:

Stack Operations
################

The following operations manipulate the DWARF stack. Operations that index the
stack assume that the top of the stack (most recently added entry) has index 0.
They allow the stack entries to be either a value or location description.

If any stack entry accessed by a stack operation is an incomplete composite
location description, then the DWARF expression is ill-formed.

.. note::

  These operations now support stack entries that are values and location
  descriptions.

.. note::

  If it is desired to also make them work with incomplete composite location
  descriptions, then would need to define that the composite location storage
  specified by the incomplete composite location description is also replicated
  when a copy is pushed. This ensures that each copy of the incomplete composite
  location description can update the composite location storage they specify
  independently.

1.  ``DW_OP_dup``

    ``DW_OP_dup`` duplicates the stack entry at the top of the stack.

2.  ``DW_OP_drop``

    ``DW_OP_drop`` pops the stack entry at the top of the stack and discards it.

3.  ``DW_OP_pick``

    ``DW_OP_pick`` has a single unsigned 1-byte operand that represents an index
    I. A copy of the stack entry with index I is pushed onto the stack.

4.  ``DW_OP_over``

    ``DW_OP_over`` pushes a copy of the entry with index 1.

    *This is equivalent to a ``DW_OP_pick 1`` operation.*

5.  ``DW_OP_swap``

    ``DW_OP_swap`` swaps the top two stack entries. The entry at the top of the
    stack becomes the second stack entry, and the second stack entry becomes the
    top of the stack.

6.  ``DW_OP_rot``

    ``DW_OP_rot`` rotates the first three stack entries. The entry at the top of
    the stack becomes the third stack entry, the second entry becomes the top of
    the stack, and the third entry becomes the second entry.

.. _amdgpu-dwarf-control-flow-operations:

Control Flow Operations
#######################

The following operations provide simple control of the flow of a DWARF operation
expression.

1.  ``DW_OP_nop``

    ``DW_OP_nop`` is a place holder. It has no effect on the DWARF stack
    entries.

2.  ``DW_OP_le``, ``DW_OP_ge``, ``DW_OP_eq``, ``DW_OP_lt``, ``DW_OP_gt``,
    ``DW_OP_ne``

    .. note::

      The same as in DWARF Version 5 section 2.5.1.5.

3.  ``DW_OP_skip``

    ``DW_OP_skip`` is an unconditional branch. Its single operand is a 2-byte
    signed integer constant. The 2-byte constant is the number of bytes of the
    DWARF expression to skip forward or backward from the current operation,
    beginning after the 2-byte constant.

    If the updated position is at one past the end of the last operation, then
    the operation expression evaluation is complete.

    Otherwise, the DWARF expression is ill-formed if the updated operation
    position is not in the range of the first to last operation inclusive, or
    not at the start of an operation.

4.  ``DW_OP_bra``

    ``DW_OP_bra`` is a conditional branch. Its single operand is a 2-byte signed
    integer constant. This operation pops the top of stack. If the value popped
    is not the constant 0, the 2-byte constant operand is the number of bytes of
    the DWARF operation expression to skip forward or backward from the current
    operation, beginning after the 2-byte constant.

    If the updated position is at one past the end of the last operation, then
    the operation expression evaluation is complete.

    Otherwise, the DWARF expression is ill-formed if the updated operation
    position is not in the range of the first to last operation inclusive, or
    not at the start of an operation.

5.  ``DW_OP_call2, DW_OP_call4, DW_OP_call_ref``

    ``DW_OP_call2``, ``DW_OP_call4``, and ``DW_OP_call_ref`` perform DWARF
    procedure calls during evaluation of a DWARF expression.

    ``DW_OP_call2`` and ``DW_OP_call4``, have one operand that is a 2- or 4-byte
    unsigned offset, respectively, of a debugging information entry D in the
    current compilation unit.

    ``DW_OP_LLVM_call_ref`` has one operand that is a 4-byte unsigned value in
    the 32-bit DWARF format, or an 8-byte unsigned value in the 64-bit DWARF
    format, that represents an offset of a debugging information entry D in a
    ``.debug_info`` section, which may be contained in an executable or shared
    object file other than that containing the operation. For references from one
    executable or shared object file to another, the relocation must be
    performed by the consumer.

    *Operand interpretation of* ``DW_OP_call2``\ *,* ``DW_OP_call4``\ *, and*
    ``DW_OP_call_ref`` *is exactly like that for* ``DW_FORM_ref2``\ *,
    ``DW_FORM_ref4``\ *, and* ``DW_FORM_ref_addr``\ *, respectively.*

    The call operation is evaluated by:

    * If D has a ``DW_AT_location`` attribute that is encoded as a ``exprloc``
      that specifies an operation expression E, then execution of the current
      operation expression continues from the first operation of E. Execution
      continues until one past the last operation of E is reached, at which
      point execution continues with the operation following the call operation.
      Since E is evaluated on the same stack as the call, E can use, add, and/or
      remove entries already on the stack.

      *Values on the stack at the time of the call may be used as parameters by
      the called expression and values left on the stack by the called expression
      may be used as return values by prior agreement between the calling and
      called expressions.*

    * If D has a ``DW_AT_location`` attribute that is encoded as a ``loclist`` or
      ``loclistsptr``, then the specified location list expression E is
      evaluated, and the resulting location description is pushed on the stack.
      The evaluation of E uses a context that has the same current frame and
      current program location as the current operation expression, but an empty
      initial stack.

      .. note::

        This rule avoids having to define how to execute a matched location list
        entry operation expression on the same stack as the call when there are
        multiple matches. But it allows the call to obtain the location
        description for a variable or formal parameter which may use a location
        list expression.

        An alternative is to treat the case when D has a ``DW_AT_location``
        attribute that is encoded as a ``loclist`` or ``loclistsptr``, and the
        specified location list expression E' matches a single location list
        entry with operation expression E, the same as the ``exprloc`` case and
        evaluate on the same stack.

        But this is not attractive as if the attribute is for a variable that
        happens to end with a non-singleton stack, it will not simply put a
        location description on the stack. Presumably the intent of using
        ``DW_OP_call*`` on a variable or formal parameter debugger information
        entry is to push just one location description on the stack. That
        location description may have more than one single location description.

        The previous rule for ``exprloc`` also has the same problem as normally
        a variable or formal parameter location expression may leave multiple
        entries on the stack and only return the top entry.

        Gdb implements ``DW_OP_call*`` by always executing E on the same stack.
        If the location list has multiple matching entries, it simply picks the
        first one and ignores the rest. This seems fundementally at odds with
        the desire to supporting multiple places for variables.

        So, it feels like ``DW_OP_call*`` should both support pushing a location
        description on the stack for a variable or formal parameter, and also
        support being able to execute an operation expression on the same stack.
        Being able to specify a different operation expression for different
        program locations seems a desirable feature to retain.

        A solution to that is to have a distinct ``DW_AT_LLVM_proc`` attribute
        for the ``DW_TAG_dwarf_procedure`` debugging information entry. Then the
        ``DW_AT_location`` attribute expression is always executed separately
        and pushes a location description (that may have multiple single
        location descriptions), and the ``DW_AT_LLVM_proc`` attribute expression
        is always executed on the same stack and can leave anything on the
        stack.

        The ``DW_AT_LLVM_proc`` attribute could have the new classes
        ``exprproc``, ``loclistproc``, and ``loclistsptrproc`` to indicate that
        the expression is executed on the same stack. ``exprproc`` is the same
        encoding as ``exprloc``. ``loclistproc`` and ``loclistsptrproc`` are the
        same encoding as their non-\ ``proc`` counterparts except the DWARF is
        ill-formed if the location list does not match exactly one location list
        entry and a default entry is required. These forms indicate explicitly
        that the matched single operation expression must be executed on the
        same stack. This is better than ad hoc special rules for ``loclistproc``
        and ``loclistsptrproc`` which are currently clearly defined to always
        return a location description. The producer then explicitly indicates
        the intent through the attribute classes.

        Such a change would be a breaking change for how gdb implements
        ``DW_OP_call*``. However, are the breaking cases actually occurring in
        practice? gdb could implement the current approach for DWARF Version 5,
        and the new semantics for DWARF Version 6 which has been done for some
        other features.

        Another option is to limit the execution to be on the same stack only to
        the evaluation of an expression E that is the value of a
        ``DW_AT_location`` attribute of a ``DW_TAG_dwarf_procedure`` debugging
        information entry. The DWARF would be ill-formed if E is a location list
        expression that does not match exactly one location list entry. In all
        other cases the evaluation of an expression E that is the value of a
        ``DW_AT_location`` attribute would evaluate E with a context that has
        the same current frame and current program location as the current
        operation expression, but an empty initial stack, and push the resulting
        location description on the stack.

    * If D has a ``DW_AT_const_value`` attribute with a value V, then it is as
      if a ``DW_OP_implicit_value V`` operation was executed.

      *This allows a call operation to be used to compute the location
      description for any variable or formal parameter regardless of whether the
      producer has optimized it to a constant. This is consistent with the
      ``DW_OP_implicit_pointer`` operation.*

      .. note::

        Alternatively, could deprecate using ``DW_AT_const_value`` for
        ``DW_TAG_variable`` and ``DW_TAG_formal_parameter`` debugger information
        entries that are constants and instead use ``DW_AT_location`` with an
        operation expression that results in a location description with one
        implicit location description. Then this rule would not be required.

    * Otherwise, there is no effect and no changes are made to the stack.

      .. note::

        In DWARF Version 5, if D does not have a ``DW_AT_location`` then
        ``DW_OP_call*`` is defined to have no effect. It is unclear that this is
        the right definition as a producer should be able to rely on using
        ``DW_OP_call*`` to get a location description for any non-\
        ``DW_TAG_dwarf_procedure`` debugging information entries. Also, the
        producer should not be creating DWARF with ``DW_OP_call*`` to a
        ``DW_TAG_dwarf_procedure`` that does not have a ``DW_AT_location``
        attribute. So, should this case be defined as an ill-formed DWARF
        expression?

    *The* ``DW_TAG_dwarf_procedure`` *debugging information entry can be used to
    define DWARF procedures that can be called.*

.. _amdgpu-dwarf-value-operations:

Value Operations
################

This section describes the operations that push values on the stack.

Each value stack entry has a type and a literal value and can represent a
literal value of any supported base type of the target architecture. The base
type specifies the size and encoding of the literal value.

Instead of a base type, value stack entries can have a distinguished generic
type, which is an integral type that has the size of an address in the target
architecture default address space and unspecified signedness.

*The generic type is the same as the unspecified type used for stack operations
defined in DWARF Version 4 and before.*

An integral type is a base type that has an encoding of ``DW_ATE_signed``,
``DW_ATE_signed_char``, ``DW_ATE_unsigned``, ``DW_ATE_unsigned_char``,
``DW_ATE_boolean``, or any target architecture defined integral encoding in the
inclusive range ``DW_ATE_lo_user`` to ``DW_ATE_hi_user``.

.. note::

  Unclear if ``DW_ATE_address`` is an integral type. Gdb does not seem to
  consider it as integral.

.. _amdgpu-dwarf-literal-operations:

Literal Operations
^^^^^^^^^^^^^^^^^^

The following operations all push a literal value onto the DWARF stack.

Operations other than ``DW_OP_const_type`` push a value V with the generic type.
If V is larger than the generic type, then V is truncated to the generic type
size and the low-order bits used.

1.  ``DW_OP_lit0``, ``DW_OP_lit1``, ..., ``DW_OP_lit31``

    ``DW_OP_lit<N>`` operations encode an unsigned literal value N from 0
    through 31, inclusive. They push the value N with the generic type.

2.  ``DW_OP_const1u``, ``DW_OP_const2u``, ``DW_OP_const4u``, ``DW_OP_const8u``

    ``DW_OP_const<N>u`` operations have a single operand that is a 1, 2, 4, or
    8-byte unsigned integer constant U, respectively. They push the value U with
    the generic type.

3.  ``DW_OP_const1s``, ``DW_OP_const2s``, ``DW_OP_const4s``, ``DW_OP_const8s``

    ``DW_OP_const<N>s`` operations have a single operand that is a 1, 2, 4, or
    8-byte signed integer constant S, respectively. They push the value S with
    the generic type.

4.  ``DW_OP_constu``

    ``DW_OP_constu`` has a single unsigned LEB128 integer operand N. It pushes
    the value N with the generic type.

5.  ``DW_OP_consts``

    ``DW_OP_consts`` has a single signed LEB128 integer operand N. It pushes the
    value N with the generic type.

6.  ``DW_OP_constx``

    ``DW_OP_constx`` has a single unsigned LEB128 integer operand that
    represents a zero-based index into the ``.debug_addr`` section relative to
    the value of the ``DW_AT_addr_base`` attribute of the associated compilation
    unit. The value N in the ``.debug_addr`` section has the size of the generic
    type. It pushes the value N with the generic type.

    *The* ``DW_OP_constx`` *operation is provided for constants that require
    link-time relocation but should not be interpreted by the consumer as a
    relocatable address (for example, offsets to thread-local storage).*

9.  ``DW_OP_const_type``

    ``DW_OP_const_type`` has three operands. The first is an unsigned LEB128
    integer that represents the offset of a debugging information entry D in the
    current compilation unit, that provides the type of the constant value. The
    second is a 1-byte unsigned integral constant S. The third is a block of
    bytes B, with a length equal to S.

    T is the bit size of the type D. The least significant T bits of B are
    interpreted as a value V of the type D. It pushes the value V with the type
    D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, or if T divided by 8 and rounded up to a multiple of 8
    (the byte size) is not equal to S.

    *While the size of the byte block B can be inferred from the type D
    definition, it is encoded explicitly into the operation so that the
    operation can be parsed easily without reference to the* ``.debug_info``
    *section.*

10. ``DW_OP_LLVM_push_lane`` *New*

    ``DW_OP_LLVM_push_lane`` pushes a value with the generic type that is the
    target architecture specific lane identifier of the thread of execution for
    which a user presented expression is currently being evaluated.

    *For languages that are implemented using a SIMD or SIMT execution model,
    this is the lane number that corresponds to the source language thread of
    execution upon which the user is focused.*

.. _amdgpu-dwarf-arithmetic-logical-operations:

Arithmetic and Logical Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  This section is the same as DWARF Version 5 section 2.5.1.4.

.. _amdgpu-dwarf-type-conversions-operations:

Type Conversion Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  This section is the same as DWARF Version 5 section 2.5.1.6.

.. _amdgpu-dwarf-general-operations:

Special Value Operations
^^^^^^^^^^^^^^^^^^^^^^^^

There are these special value operations currently defined:

1.  ``DW_OP_regval_type``

    ``DW_OP_regval_type`` has two operands. The first is an unsigned LEB128
    integer that represents a register number R. The second is an unsigned
    LEB128 integer that represents the offset of a debugging information entry D
    in the current compilation unit, that provides the type of the register
    value.

    The contents of register R are interpreted as a value V of the type D. The
    value V is pushed on the stack with the type D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, or if the size of type D is not the same as the size of
    register R.

    .. note::

      Should DWARF allow the type D to be a different size to the size of the
      register R? Requiring them to be the same bit size avoids any issue of
      conversion as the bit contents of the register is simply interpreted as a
      value of the specified type. If a conversion is wanted it can be done
      explicitly using a ``DW_OP_convert`` operation.

      Gdb has a per register hook that allows a target specific conversion on a
      register by register basis. It defaults to truncation of bigger registers,
      and to actually reading bytes from the next register (or reads out of
      bounds for the last register) for smaller registers. There are no gdb
      tests that read a register out of bounds (except an illegal hand written
      assembly test).

2.  ``DW_OP_deref``

    The ``DW_OP_deref`` operation pops one stack entry that must be a location
    description L.

    A value of the bit size of the generic type is retrieved from the location
    storage specified by L. The value V retrieved is pushed on the stack with
    the generic type.

    If any bit of the value is retrieved from the undefined location storage, or
    the offset of any bit exceeds the size of the location storage specified by
    L, then the DWARF expression is ill-formed.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

    *If L, or the location description of any composite location description
    part that is a subcomponent of L, has more than one single location
    description, then any one of them can be selected as they are required to
    all have the same value. For any single location description SL, bits are
    retrieved from the associated storage location starting at the bit offset
    specified by SL. For a composite location description, the retrieved bits
    are the concatenation of the N bits from each composite location part PL,
    where N is limited to the size of PL.*

3.  ``DW_OP_deref_size``

    ``DW_OP_deref_size`` has a single 1-byte unsigned integral constant that
    represents a byte result size S.

    It pops one stack entry that must be a location description L.

    T is the smaller of the generic type size and S scaled by 8 (the byte size).
    A value V of T bits is retrieved from the location storage specified by L.
    If V is smaller than the size of the generic type, V is zero-extended to the
    generic type size. V is pushed onto the stack with the generic type.

    The DWARF expression is ill-formed if any bit of the value is retrieved from
    the undefined location storage, or if the offset of any bit exceeds the size
    of the location storage specified by L.

    .. note::

      Truncating the value when S is larger than the generic type matches what
      gdb does. This allows the generic type size to not be a integral byte
      size. It does allow S to be arbitrarily large. Should S be restricted to
      the size of the generic type rounded up to a multiple of 8?

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

4.  ``DW_OP_deref_type``

    ``DW_OP_deref_type`` has two operands. The first is a 1-byte unsigned
    integral constant S. The second is an unsigned LEB128 integer that
    represents the offset of a debugging information entry D in the current
    compilation unit, that provides the type of the result value.

    It pops one stack entry that must be a location description L. T is the bit
    size of the type D. A value V of T bits is retrieved from the location
    storage specified by L. V is pushed on the stack with the type D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, if T divided by 8 and rounded up to a multiple of 8 (the
    byte size) is not equal to S, if any bit of the value is retrieved from the
    undefined location storage, or if the offset of any bit exceeds the size of
    the location storage specified by L.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

    *While the size of the pushed value V can be inferred from the type D
    definition, it is encoded explicitly into the operation so that the
    operation can be parsed easily without reference to the* ``.debug_info``
    *section.*

    .. note::

      It is unclear why the operand S is needed. Unlike ``DW_OP_const_type``,
      the size is not needed for parsing. Any evaluation needs to get the base
      type to record with the value to know its encoding and bit size.

      This definition allows the base type to be a bit size since there seems no
      reason to restrict it.

5.  ``DW_OP_xderef`` *Deprecated*

    ``DW_OP_xderef`` pops two stack entries. The first must be an integral type
    value that represents an address A. The second must be an integral type
    value that represents a target architecture specific address space
    identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref``. The value V retrieved is left
    on the stack with the generic type.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    operation can be used and provides greater expressiveness.*

6.  ``DW_OP_xderef_size`` *Deprecated*

    ``DW_OP_xderef_size`` has a single 1-byte unsigned integral constant that
    represents a byte result size S.

    It pops two stack entries. The first must be an integral type value that
    represents an address A. The second must be an integral type value that
    represents a target architecture specific address space identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref_size S``. The zero-extended
    value V retrieved is left on the stack with the generic type.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    operation can be used and provides greater expressiveness.*

7.  ``DW_OP_xderef_type`` *Deprecated*

    ``DW_OP_xderef_type`` has two operands. The first is a 1-byte unsigned
    integral constant S. The second operand is an unsigned LEB128
    integer R that represents the offset of a debugging information entry D in
    the current compilation unit, that provides the type of the result value.

    It pops two stack entries. The first must be an integral type value that
    represents an address A. The second must be an integral type value that
    represents a target architecture specific address space identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref_type S R``. The value V
    retrieved is left on the stack with the type D.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    operation can be used and provides greater expressiveness.*

8.  ``DW_OP_entry_value`` *Deprecated*

    ``DW_OP_entry_value`` pushes the value that the described location held upon
    entering the current subprogram.

    It has two operands. The first is an unsigned LEB128 integer S. The second
    is a block of bytes, with a length equal S, interpreted as a DWARF
    operation expression E.

    E is evaluated as if it had been evaluated upon entering the current
    subprogram with an empty initial stack.

    .. note::

      It is unclear what this means. What is the current program location and
      current frame that must be used? Does this require reverse execution so
      the register and memory state are as it was on entry to the current
      subprogram?

    The DWARF expression is ill-formed if the evaluation of E executes a
    ``DW_OP_push_object_address`` operation.

    If the result of E is a location description with one register location
    description (see :ref:`amdgpu-dwarf-register-location-descriptions`),
    ``DW_OP_entry_value`` pushes the value that register had upon entering the
    current subprogram. The value entry type is the target architecture register
    base type. If the register value is undefined or the register location
    description bit offset is not 0, then the DWARF expression is ill-formed.

    *The register location description provides a more compact form for the case
    where the value was in a register on entry to the subprogram.*

    If the result of E is a value V, ``DW_OP_entry_value`` pushes V on the
    stack.

    Otherwise, the DWARF expression is ill-formed.

    *The values needed to evaluate* ``DW_OP_entry_value`` *could be obtained in
    several ways. The consumer could suspend execution on entry to the
    subprogram, record values needed by* ``DW_OP_entry_value`` *expressions
    within the subprogram, and then continue. When evaluating*
    ``DW_OP_entry_value``\ *, the consumer would use these recorded values
    rather than the current values. Or, when evaluating* ``DW_OP_entry_value``\
    *, the consumer could virtually unwind using the Call Frame Information
    (see* :ref:`amdgpu-dwarf-call-frame-information`\ *) to recover register
    values that might have been clobbered since the subprogram entry point.*

    *The* ``DW_OP_entry_value`` *operation is deprecated as its main usage is
    provided by other means. DWARF Version 5 added the*
    ``DW_TAG_call_site_parameter`` *debugger information entry for call sites
    that has* ``DW_AT_call_value``\ *,* ``DW_AT_call_data_location``\ *, and*
    ``DW_AT_call_data_value`` *attributes that provide DWARF expressions to
    compute actual parameter values at the time of the call, and requires the
    producer to ensure the expressions are valid to evaluate even when virtually
    unwound. The* ``DW_OP_LLVM_call_frame_entry_reg`` *operation provides access
    to registers in the virtually unwound calling frame.*

    .. note::

      It is unclear why this operation is defined this way. How would a consumer
      know what values have to be saved on entry to the subprogram? Does it have
      to parse every expression of every ``DW_OP_entry_value`` operation to
      capture all the possible results needed? Or does it have to implement
      reverse execution so it can evaluate the expression in the context of the
      entry of the subprogram so it can obtain the entry point register and
      memory values? Or does the compiler somehow instruct the consumer how to
      create the saved copies of the variables on entry?

      If the expression is simply using existing variables, then it is just a
      regular expression and no special operation is needed. If the main purpose
      is only to read the entry value of a register using CFI then it would be
      better to have an operation that explicitly does just that such as the
      proposed ``DW_OP_LLVM_call_frame_entry_reg`` operation.

      Gdb only seems to implement ``DW_OP_entry_value`` when E is exactly
      ``DW_OP_reg*`` or ``DW_OP_breg*; DW_OP_deref*``. It evaluates E in the
      context of the calling subprogram and the calling call site program
      location. But the wording suggests that is not the intention.

      Given these issues it is suggested ``DW_OP_entry_value`` is deprecated in
      favor of using the new facities that have well defined semantics and
      implementations.

.. _amdgpu-dwarf-location-description-operations:

Location Description Operations
###############################

This section describes the operations that push location descriptions on the
stack.

General Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.  ``DW_OP_LLVM_offset`` *New*

    ``DW_OP_LLVM_offset`` pops two stack entries. The first must be an integral
    type value that represents a byte displacement B. The second must be a
    location description L.

    It adds the value of B scaled by 8 (the byte size) to the bit offset of each
    single location description SL of L, and pushes the updated L.

    If the updated bit offset of any SL is less than 0 or greater than or equal
    to the size of the location storage specified by SL, then the DWARF
    expression is ill-formed.

2.  ``DW_OP_LLVM_offset_constu`` *New*

    ``DW_OP_LLVM_offset_constu`` has a single unsigned LEB128 integer operand
    that represents a byte displacement B.

    The operation is equivalent to performing ``DW_OP_constu B;
    DW_OP_LLVM_offset``.

    *This operation is supplied specifically to be able to encode more field
    displacements in two bytes than can be done with* ``DW_OP_lit*;
    DW_OP_LLVM_offset``\ *.*

3.  ``DW_OP_LLVM_bit_offset`` *New*

    ``DW_OP_LLVM_bit_offset`` pops two stack entries. The first must be an
    integral type value that represents a bit displacement B. The second must be
    a location description L.

    It adds the value of B to the bit offset of each single location description
    SL of L, and pushes the updated L.

    If the updated bit offset of any SL is less than 0 or greater than or equal
    to the size of the location storage specified by SL, then the DWARF
    expression is ill-formed.

4.  ``DW_OP_push_object_address``

    ``DW_OP_push_object_address`` pushes the location description L of the
    object currently being evaluated as part of evaluation of a user presented
    expression.

    This object may correspond to an independent variable described by its own
    debugging information entry or it may be a component of an array, structure,
    or class whose address has been dynamically determined by an earlier step
    during user expression evaluation.

    *This operation provides explicit functionality (especially for arrays
    involving descriptions) that is analogous to the implicit push of the base
    location description of a structure prior to evaluation of a
    ``DW_AT_data_member_location`` to access a data member of a structure.*

5.  ``DW_OP_LLVM_call_frame_entry_reg`` *New*

    ``DW_OP_LLVM_call_frame_entry_reg`` has a single unsigned LEB128 integer
    operand that represents a target architecture register number R.

    It pushes a location description L that holds the value of register R on
    entry to the current subprogram as defined by the Call Frame Information
    (see :ref:`amdgpu-dwarf-call-frame-information`).

    *If there is no Call Frame Information defined, then the default rules for
    the target architecture are used. If the register rule is* undefined\ *, then
    the undefined location description is pushed. If the register rule is* same
    value\ *, then a register location description for R is pushed.*

Undefined Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*The undefined location storage represents a piece or all of an object that is
present in the source but not in the object code (perhaps due to optimization).
Neither reading nor writing to the undefined location storage is meaningful.*

An undefined location description specifies the undefined location storage.
There is no concept of the size of the undefined location storage, nor of a bit
offset for an undefined location description. The ``DW_OP_LLVM_*offset``
operations leave an undefined location description unchanged. The
``DW_OP_*piece`` operations can explicitly or implicitly specify an undefined
location description, allowing any size and offset to be specified, and results
in a part with all undefined bits.

1.  ``DW_OP_LLVM_undefined`` *New*

    ``DW_OP_LLVM_undefined`` pushes a location description L that comprises one
    undefined location description SL.

.. _amdgpu-dwarf-memory-location-description-operations:

Memory Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each of the target architecture specific address spaces has a corresponding
memory location storage that denotes the linear addressable memory of that
address space. The size of each memory location storage corresponds to the range
of the addresses in the corresponding address space.

*It is target architecture defined how address space location storage maps to
target architecture physical memory. For example, they may be independent
memory, or more than one location storage may alias the same physical memory
possibly at different offsets and with different interleaving. The mapping may
also be dictated by the source language address classes.*

A memory location description specifies a memory location storage. The bit
offset corresponds to a bit position within a byte of the memory. Bits accessed
using a memory location description, access the corresponding target
architecture memory starting at the bit position within the byte specified by
the bit offset.

A memory location description that has a bit offset that is a multiple of 8 (the
byte size) is defined to be a byte address memory location description. It has a
memory byte address A that is equal to the bit offset divided by 8.

A memory location description that does not have a bit offset that is a multiple
of 8 (the byte size) is defined to be a bit field memory location description.
It has a bit position B equal to the bit offset modulo 8, and a memory byte
address A equal to the bit offset minus B that is then divided by 8.

The address space AS of a memory location description is defined to be the
address space that corresponds to the memory location storage associated with
the memory location description.

A location description that is comprised of one byte address memory location
description SL is defined to be a memory byte address location description. It
has a byte address equal to A and an address space equal to AS of the
corresponding SL.

``DW_ASPACE_none`` is defined as the target architecture default address space.

If a stack entry is required to be a location description, but it is a value V
with the generic type, then it is implicitly converted to a location description
L with one memory location description SL. SL specifies the memory location
storage that corresponds to the target architecture default address space with a
bit offset equal to V scaled by 8 (the byte size).

.. note::

  If it is wanted to allow any integral type value to be implicitly converted to
  a memory location description in the target architecture default address
  space:

    If a stack entry is required to be a location description, but is a value V
    with an integral type, then it is implicitly converted to a location
    description L with a one memory location description SL. If the type size of
    V is less than the generic type size, then the value V is zero extended to
    the size of the generic type. The least significant generic type size bits
    are treated as a twos-complement unsigned value to be used as an address A.
    SL specifies memory location storage corresponding to the target
    architecture default address space with a bit offset equal to A scaled by 8
    (the byte size).

  The implicit conversion could also be defined as target architecture specific.
  For example, gdb checks if V is an integral type. If it is not it gives an
  error. Otherwise, gdb zero-extends V to 64 bits. If the gdb target defines a
  hook function, then it is called. The target specific hook function can modify
  the 64-bit value, possibly sign extending based on the original value type.
  Finally, gdb treats the 64-bit value V as a memory location address.

If a stack entry is required to be a location description, but it is an implicit
pointer value IPV with the target architecture default address space, then it is
implicitly converted to a location description with one single location
description specified by IPV. See
:ref:`amdgpu-dwarf-implicit-location-descriptions`.

.. note::

  Is this rule required for DWARF Version 5 backwards compatibility? If not, it
  can be eliminated, and the producer can use
  ``DW_OP_LLVM_form_aspace_address``.

If a stack entry is required to be a value, but it is a location description L
with one memory location description SL in the target architecture default
address space with a bit offset B that is a multiple of 8, then it is implicitly
converted to a value equal to B divided by 8 (the byte size) with the generic
type.

1.  ``DW_OP_addr``

    ``DW_OP_addr`` has a single byte constant value operand, which has the size
    of the generic type, that represents an address A.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage corresponding to the
    target architecture default address space with a bit offset equal to A
    scaled by 8 (the byte size).

    *If the DWARF is part of a code object, then A may need to be relocated. For
    example, in the ELF code object format, A must be adjusted by the difference
    between the ELF segment virtual address and the virtual address at which the
    segment is loaded.*

2.  ``DW_OP_addrx``

    ``DW_OP_addrx`` has a single unsigned LEB128 integer operand that represents
    a zero-based index into the ``.debug_addr`` section relative to the value of
    the ``DW_AT_addr_base`` attribute of the associated compilation unit. The
    address value A in the ``.debug_addr`` section has the size of the generic
    type.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage corresponding to the
    target architecture default address space with a bit offset equal to A
    scaled by 8 (the byte size).

    *If the DWARF is part of a code object, then A may need to be relocated. For
    example, in the ELF code object format, A must be adjusted by the difference
    between the ELF segment virtual address and the virtual address at which the
    segment is loaded.*

3.  ``DW_OP_LLVM_form_aspace_address`` *New*

    ``DW_OP_LLVM_form_aspace_address`` pops top two stack entries. The first
    must be an integral type value that represents a target architecture
    specific address space identifier AS. The second must be an integral type
    value that represents an address A.

    The address size S is defined as the address bit size of the target
    architecture specific address space that corresponds to AS.

    A is adjusted to S bits by zero extending if necessary, and then treating the
    least significant S bits as a twos-complement unsigned value A'.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage that corresponds to
    AS with a bit offset equal to A' scaled by 8 (the byte size).

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit pointer values produced by dereferencing implicit
    location descriptions created by the ``DW_OP_implicit_pointer`` and
    ``DW_OP_LLVM_implicit_aspace_pointer`` operations.

4.  ``DW_OP_form_tls_address``

    ``DW_OP_form_tls_address`` pops one stack entry that must be an integral
    type value and treats it as a thread-local storage address T.

    It pushes a location description L with one memory location description SL
    on the stack. SL is the target architecture specific memory location
    description that corresponds to the thread-local storage address T.

    The meaning of the thread-local storage address T is defined by the run-time
    environment. If the run-time environment supports multiple thread-local
    storage blocks for a single thread, then the block corresponding to the
    executable or shared library containing this DWARF expression is used.

    *Some implementations of C, C++, Fortran, and other languages support a
    thread-local storage class. Variables with this storage class have distinct
    values and addresses in distinct threads, much as automatic variables have
    distinct values and addresses in each subprogram invocation. Typically,
    there is a single block of storage containing all thread-local variables
    declared in the main executable, and a separate block for the variables
    declared in each shared library. Each thread-local variable can then be
    accessed in its block using an identifier. This identifier is typically a
    byte offset into the block and pushed onto the DWARF stack by one of the*
    ``DW_OP_const*`` *operations prior to the* ``DW_OP_form_tls_address``
    *operation. Computing the address of the appropriate block can be complex
    (in some cases, the compiler emits a function call to do it), and difficult
    to describe using ordinary DWARF location descriptions. Instead of forcing
    complex thread-local storage calculations into the DWARF expressions, the*
    ``DW_OP_form_tls_address`` *allows the consumer to perform the computation
    based on the target architecture specific run-time environment.*

5.  ``DW_OP_call_frame_cfa``

    ``DW_OP_call_frame_cfa`` pushes the location description L of the Canonical
    Frame Address (CFA) of the current subprogram, obtained from the Call Frame
    Information on the stack. See :ref:`amdgpu-dwarf-call-frame-information`.

    *Although the value of the* ``DW_AT_frame_base`` *attribute of the debugger
    information entry corresponding to the current subprogram can be computed
    using a location list expression, in some cases this would require an
    extensive location list because the values of the registers used in
    computing the CFA change during a subprogram execution. If the Call Frame
    Information is present, then it already encodes such changes, and it is
    space efficient to reference that using the* ``DW_OP_call_frame_cfa``
    *operation.*

6.  ``DW_OP_fbreg``

    ``DW_OP_fbreg`` has a single signed LEB128 integer operand that represents a
    byte displacement B.

    The location description L for the *frame base* of the current subprogram is
    obtained from the ``DW_AT_frame_base`` attribute of the debugger information
    entry corresponding to the current subprogram as described in
    :ref:`amdgpu-dwarf-debugging-information-entry-attributes`.

    The location description L is updated as if the ``DW_OP_LLVM_offset_constu
    B`` operation was applied. The updated L is pushed on the stack.

7.  ``DW_OP_breg0``, ``DW_OP_breg1``, ..., ``DW_OP_breg31``

    The ``DW_OP_breg<N>`` operations encode the numbers of up to 32 registers,
    numbered from 0 through 31, inclusive. The register number R corresponds to
    the N in the operation name.

    They have a single signed LEB128 integer operand that represents a byte
    displacement B.

    The address space identifier AS is defined as the one corresponding to the
    target architecture specific default address space.

    The address size S is defined as the address bit size of the target
    architecture specific address space corresponding to AS.

    The contents of the register specified by R are retrieved as a
    twos-complement unsigned value and zero extended to S bits. B is added and
    the least significant S bits are treated as a twos-complement unsigned value
    to be used as an address A.

    They push a location description L comprising one memory location
    description LS on the stack. LS specifies the memory location storage that
    corresponds to AS with a bit offset equal to A scaled by 8 (the byte size).

8.  ``DW_OP_bregx``

    ``DW_OP_bregx`` has two operands. The first is an unsigned LEB128 integer
    that represents a register number R. The second is a signed LEB128
    integer that represents a byte displacement B.

    The action is the same as for ``DW_OP_breg<N>`` except that R is used as the
    register number and B is used as the byte displacement.

9.  ``DW_OP_LLVM_aspace_bregx`` *New*

    ``DW_OP_LLVM_aspace_bregx`` has two operands. The first is an unsigned
    LEB128 integer that represents a register number R. The second is a signed
    LEB128 integer that represents a byte displacement B. It pops one stack
    entry that is required to be an integral type value that represents a target
    architecture specific address space identifier AS.

    The action is the same as for ``DW_OP_breg<N>`` except that R is used as the
    register number, B is used as the byte displacement, and AS is used as the
    address space identifier.

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

    .. note::

      Could also consider adding ``DW_OP_aspace_breg0, DW_OP_aspace_breg1, ...,
      DW_OP_aspace_bref31`` which would save encoding size.

.. _amdgpu-dwarf-register-location-descriptions:

Register Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a register location storage that corresponds to each of the target
architecture registers. The size of each register location storage corresponds
to the size of the corresponding target architecture register.

A register location description specifies a register location storage. The bit
offset corresponds to a bit position within the register. Bits accessed using a
register location description access the corresponding target architecture
register starting at the specified bit offset.

1.  ``DW_OP_reg0``, ``DW_OP_reg1``, ..., ``DW_OP_reg31``

    ``DW_OP_reg<N>`` operations encode the numbers of up to 32 registers,
    numbered from 0 through 31, inclusive. The target architecture register
    number R corresponds to the N in the operation name.

    They push a location description L that specifies one register location
    description SL on the stack. SL specifies the register location storage that
    corresponds to R with a bit offset of 0.

2.  ``DW_OP_regx``

    ``DW_OP_regx`` has a single unsigned LEB128 integer operand that represents
    a target architecture register number R.

    It pushes a location description L that specifies one register location
    description SL on the stack. SL specifies the register location storage that
    corresponds to R with a bit offset of 0.

*These operations obtain a register location. To fetch the contents of a
register, it is necessary to use* ``DW_OP_regval_type``\ *, use one of the*
``DW_OP_breg*`` *register-based addressing operations, or use* ``DW_OP_deref*``
*on a register location description.*

.. _amdgpu-dwarf-implicit-location-descriptions:

Implicit Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implicit location storage represents a piece or all of an object which has no
actual location in the program but whose contents are nonetheless known, either
as a constant or can be computed from other locations and values in the program.

An implicit location description specifies an implicit location storage. The bit
offset corresponds to a bit position within the implicit location storage. Bits
accessed using an implicit location description, access the corresponding
implicit storage value starting at the bit offset.

1.  ``DW_OP_implicit_value``

    ``DW_OP_implicit_value`` has two operands. The first is an unsigned LEB128
    integer that represents a byte size S. The second is a block of bytes with a
    length equal to S treated as a literal value V.

    An implicit location storage LS is created with the literal value V and a
    size of S.

    It pushes location description L with one implicit location description SL
    on the stack. SL specifies LS with a bit offset of 0.

2.  ``DW_OP_stack_value``

    ``DW_OP_stack_value`` pops one stack entry that must be a value V.

    An implicit location storage LS is created with the literal value V and a
    size equal to V's base type size.

    It pushes a location description L with one implicit location description SL
    on the stack. SL specifies LS with a bit offset of 0.

    *The* ``DW_OP_stack_value`` *operation specifies that the object does not
    exist in memory, but its value is nonetheless known. In this form, the
    location description specifies the actual value of the object, rather than
    specifying the memory or register storage that holds the value.*

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit pointer values produced by dereferencing implicit
    location descriptions created by the ``DW_OP_implicit_pointer`` and
    ``DW_OP_LLVM_implicit_aspace_pointer`` operations.

    .. note::

      Since location descriptions are allowed on the stack, the
      ``DW_OP_stack_value`` operation no longer terminates the DWARF operation
      expression execution as in DWARF Version 5.

3.  ``DW_OP_implicit_pointer``

    *An optimizing compiler may eliminate a pointer, while still retaining the
    value that the pointer addressed.* ``DW_OP_implicit_pointer`` *allows a
    producer to describe this value.*

    ``DW_OP_implicit_pointer`` *specifies an object is a pointer to the target
    architecture default address space that cannot be represented as a real
    pointer, even though the value it would point to can be described. In this
    form, the location description specifies a debugging information entry that
    represents the actual location description of the object to which the
    pointer would point. Thus, a consumer of the debug information would be able
    to access the dereferenced pointer, even when it cannot access the pointer
    itself.*

    ``DW_OP_implicit_pointer`` has two operands. The first is a 4-byte unsigned
    value in the 32-bit DWARF format, or an 8-byte unsigned value in the 64-bit
    DWARF format, that represents a debugging information entry reference R. The
    second is a signed LEB128 integer that represents a byte displacement B.

    R is used as the offset of a debugging information entry D in a
    ``.debug_info`` section, which may be contained in an executable or shared
    object file other than that containing the operation. For references from one
    executable or shared object file to another, the relocation must be
    performed by the consumer.

    *The first operand interpretation is exactly like that for*
    ``DW_FORM_ref_addr``\ *.*

    The address space identifier AS is defined as the one corresponding to the
    target architecture specific default address space.

    The address size S is defined as the address bit size of the target
    architecture specific address space corresponding to AS.

    An implicit location storage LS is created with the debugging information
    entry D, address space AS, and size of S.

    It pushes a location description L that comprises one implicit location
    description SL on the stack. SL specifies LS with a bit offset of 0.

    If a ``DW_OP_deref*`` operation pops a location description L', and
    retrieves S bits where both:

    1.  All retrieved bits come from an implicit location description that
        refers to an implicit location storage that is the same as LS.

        *Note that all bits do not have to come from the same implicit location
        description, as L' may involve composite location descriptors.*

    2.  The bits come from consecutive ascending offsets within their respective
        implicit location storage.

    *These rules are equivalent to retrieving the complete contents of LS.*

    Then the value V pushed by the ``DW_OP_deref*`` operation is an implicit
    pointer value IPV with a target architecture specific address space of AS, a
    debugging information entry of D, and a base type of T. If AS is the target
    architecture default address space, then T is the generic type. Otherwise, T
    is a target architecture specific integral type with a bit size equal to S.

    Otherwise, if a ``DW_OP_deref*`` operation is applied to a location
    description such that some retrieved bits come from an implicit location
    storage that is the same as LS, then the DWARF expression is ill-formed.

    If IPV is either implicitly converted to a location description (only done
    if AS is the target architecture default address space) or used by
    ``DW_OP_LLVM_form_aspace_address`` (only done if the address space specified
    is AS), then the resulting location description RL is:

    * If D has a ``DW_AT_location`` attribute, the DWARF expression E from the
      ``DW_AT_location`` attribute is evaluated as a location description. The
      current subprogram and current program location of the evaluation context
      that is accessing IPV is used for the evaluation context of E, together
      with an empty initial stack. RL is the expression result.

    * If D has a ``DW_AT_const_value`` attribute, then an implicit location
      storage RLS is created from the ``DW_AT_const_value`` attribute's value
      with a size matching the size of the ``DW_AT_const_value`` attribute's
      value. RL comprises one implicit location description SRL. SRL specifies
      RLS with a bit offset of 0.

      .. note::

        If using ``DW_AT_const_value`` for variables and formal parameters is
        deprecated and instead ``DW_AT_location`` is used with an implicit
        location description, then this rule would not be required.

    * Otherwise the DWARF expression is ill-formed.

    The bit offset of RL is updated as if the ``DW_OP_LLVM_offset_constu B``
    operation was applied.

    If a ``DW_OP_stack_value`` operation pops a value that is the same as IPV,
    then it pushes a location description that is the same as L.

    The DWARF expression is ill-formed if it accesses LS or IPV in any other
    manner.

    *The restrictions on how an implicit pointer location description created
    by* ``DW_OP_implicit_pointer`` *and* ``DW_OP_LLVM_aspace_implicit_pointer``
    *can be used are to simplify the DWARF consumer. Similarly, for an implicit
    pointer value created by* ``DW_OP_deref*`` *and* ``DW_OP_stack_value``\ .*

4.  ``DW_OP_LLVM_aspace_implicit_pointer`` *New*

    ``DW_OP_LLVM_aspace_implicit_pointer`` has two operands that are the same as
    for ``DW_OP_implicit_pointer``.

    It pops one stack entry that must be an integral type value that represents
    a target architecture specific address space identifier AS.

    The location description L that is pushed on the stack is the same as for
    ``DW_OP_implicit_pointer`` except that the address space identifier used is
    AS.

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

*Typically a* ``DW_OP_implicit_pointer`` *or*
``DW_OP_LLVM_aspace_implicit_pointer`` *operation is used in a DWARF expression
E*\ :sub:`1` *of a* ``DW_TAG_variable`` *or* ``DW_TAG_formal_parameter``
*debugging information entry D*\ :sub:`1`\ *'s* ``DW_AT_location`` *attribute.
The debugging information entry referenced by the* ``DW_OP_implicit_pointer``
*or* ``DW_OP_LLVM_aspace_implicit_pointer`` *operations is typically itself a*
``DW_TAG_variable`` *or* ``DW_TAG_formal_parameter`` *debugging information
entry D*\ :sub:`2` *whose* ``DW_AT_location`` *attribute gives a second DWARF
expression E*\ :sub:`2`\ *.*

*D*\ :sub:`1` *and E*\ :sub:`1` *are describing the location of a pointer type
object. D*\ :sub:`2` *and E*\ :sub:`2` *are describing the location of the
object pointed to by that pointer object.*

*However, D*\ :sub:`2` *may be any debugging information entry that contains a*
``DW_AT_location`` *or* ``DW_AT_const_value`` *attribute (for example,*
``DW_TAG_dwarf_procedure``\ *). By using E*\ :sub:`2`\ *, a consumer can
reconstruct the value of the object when asked to dereference the pointer
described by E*\ :sub:`1` *which contains the* ``DW_OP_implicit_pointer`` or
``DW_OP_LLVM_aspace_implicit_pointer`` *operation.*

Composite Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A composite location storage represents an object or value which may be
contained in part of another location storage or contained in parts of more
than one location storage.

Each part has a part location description L and a part bit size S. L can have
one or more single location descriptions SL. If there are more than one SL then
that indicates that part is located in more than one place. The bits of each
place of the part comprise S contiguous bits from the location storage LS
specified by SL starting at the bit offset specified by SL. All the bits must
be within the size of LS or the DWARF expression is ill-formed.

A composite location storage can have zero or more parts. The parts are
contiguous such that the zero-based location storage bit index will range over
each part with no gaps between them. Therefore, the size of a composite location
storage is the sum of the size of its parts. The DWARF expression is ill-formed
if the size of the contiguous location storage is larger than the size of the
memory location storage corresponding to the largest target architecture
specific address space.

A composite location description specifies a composite location storage. The bit
offset corresponds to a bit position within the composite location storage.

There are operations that create a composite location storage.

There are other operations that allow a composite location storage to be
incrementally created. Each part is created by a separate operation. There may
be one or more operations to create the final composite location storage. A
series of such operations describes the parts of the composite location storage
that are in the order that the associated part operations are executed.

To support incremental creation, a composite location storage can be in an
incomplete state. When an incremental operation operates on an incomplete
composite location storage, it adds a new part, otherwise it creates a new
composite location storage. The ``DW_OP_LLVM_piece_end`` operation explicitly
makes an incomplete composite location storage complete.

A composite location description that specifies a composite location storage
that is incomplete is termed an incomplete composite location description. A
composite location description that specifies a composite location storage that
is complete is termed a complete composite location description.

If the top stack entry is a location description that has one incomplete
composite location description SL after the execution of an operation expression
has completed, SL is converted to a complete composite location description.

*Note that this conversion does not happen after the completion of an operation
expression that is evaluated on the same stack by the* ``DW_OP_call*``
*operations. Such executions are not a separate evaluation of an operation
expression, but rather the continued evaluation of the same operation expression
that contains the* ``DW_OP_call*`` *operation.*

If a stack entry is required to be a location description L, but L has an
incomplete composite location description, then the DWARF expression is
ill-formed. The exception is for the operations involved in incrementally
creating a composite location description as described below.

*Note that a DWARF operation expression may arbitrarily compose composite
location descriptions from any other location description, including those that
have multiple single location descriptions, and those that have composite
location descriptions.*

*The incremental composite location description operations are defined to be
compatible with the definitions in DWARF Version 5.*

1.  ``DW_OP_piece``

    ``DW_OP_piece`` has a single unsigned LEB128 integer that represents a byte
    size S.

    The action is based on the context:

    * If the stack is empty, then a location description L comprised of one
      incomplete composite location description SL is pushed on the stack.

      An incomplete composite location storage LS is created with a single part
      P. P specifies a location description PL and has a bit size of S scaled by
      8 (the byte size). PL is comprised of one undefined location description
      PSL.

      SL specifies LS with a bit offset of 0.

    * Otherwise, if the top stack entry is a location description L comprised of
      one incomplete composite location description SL, then the incomplete
      composite location storage LS that SL specifies is updated to append a new
      part P. P specifies a location description PL and has a bit size of S
      scaled by 8 (the byte size). PL is comprised of one undefined location
      description PSL. L is left on the stack.

    * Otherwise, if the top stack entry is a location description or can be
      converted to one, then it is popped and treated as a part location
      description PL. Then:

      * If the top stack entry (after popping PL) is a location description L
        comprised of one incomplete composite location description SL, then the
        incomplete composite location storage LS that SL specifies is updated to
        append a new part P. P specifies the location description PL and has a
        bit size of S scaled by 8 (the byte size). L is left on the stack.

      * Otherwise, a location description L comprised of one incomplete
        composite location description SL is pushed on the stack.

        An incomplete composite location storage LS is created with a single
        part P. P specifies the location description PL and has a bit size of S
        scaled by 8 (the byte size).

        SL specifies LS with a bit offset of 0.

    * Otherwise, the DWARF expression is ill-formed

    *Many compilers store a single variable in sets of registers or store a
    variable partially in memory and partially in registers.* ``DW_OP_piece``
    *provides a way of describing where a part of a variable is located.*

    *If a non-0 byte displacement is required, the* ``DW_OP_LLVM_offset``
    *operation can be used to update the location description before using it as
    the part location description of a* ``DW_OP_piece`` *operation.*

    *The evaluation rules for the* ``DW_OP_piece`` *operation allow it to be
    compatible with the DWARF Version 5 definition.*

    .. note::

      Since this proposal allows location descriptions to be entries on the
      stack, a simpler operation to create composite location descriptions. For
      example, just one operation that specifies how many parts, and pops pairs
      of stack entries for the part size and location description. Not only
      would this be a simpler operation and avoid the complexities of incomplete
      composite location descriptions, but it may also have a smaller encoding
      in practice. However, the desire for compatibility with DWARF Version 5 is
      likely a stronger consideration.

2.  ``DW_OP_bit_piece``

    ``DW_OP_bit_piece`` has two operands. The first is an unsigned LEB128
    integer that represents the part bit size S. The second is an unsigned
    LEB128 integer that represents a bit displacement B.

    The action is the same as for ``DW_OP_piece`` except that any part created
    has the bit size S, and the location description PL of any created part is
    updated as if the ``DW_OP_constu B; DW_OP_LLVM_bit_offset`` operations were
    applied.

    ``DW_OP_bit_piece`` *is used instead of* ``DW_OP_piece`` *when the piece to
    be assembled is not byte-sized or is not at the start of the part location
    description.*

    *If a computed bit displacement is required, the* ``DW_OP_LLVM_bit_offset``
    *operation can be used to update the location description before using it as
    the part location description of a* ``DW_OP_bit_piece`` *operation.*

    .. note::

      The bit offset operand is not needed as ``DW_OP_LLVM_bit_offset`` can be
      used on the part's location description.

3.  ``DW_OP_LLVM_piece_end`` *New*

    If the top stack entry is not a location description L comprised of one
    incomplete composite location description SL, then the DWARF expression is
    ill-formed.

    Otherwise, the incomplete composite location storage LS specified by SL is
    updated to be a complete composite location description with the same parts.

4.  ``DW_OP_LLVM_extend`` *New*

    ``DW_OP_LLVM_extend`` has two operands. The first is an unsigned LEB128
    integer that represents the element bit size S. The second is an unsigned
    LEB128 integer that represents a count C.

    It pops one stack entry that must be a location description and is treated
    as the part location description PL.

    A location description L comprised of one complete composite location
    description SL is pushed on the stack.

    A complete composite location storage LS is created with C identical parts
    P. Each P specifies PL and has a bit size of S.

    SL specifies LS with a bit offset of 0.

    The DWARF expression is ill-formed if the element bit size or count are 0.

5.  ``DW_OP_LLVM_select_bit_piece`` *New*

    ``DW_OP_LLVM_select_bit_piece`` has two operands. The first is an unsigned
    LEB128 integer that represents the element bit size S. The second is an
    unsigned LEB128 integer that represents a count C.

    It pops three stack entries. The first must be an integral type value that
    represents a bit mask value M. The second must be a location description
    that represents the one-location description L1. The third must be a
    location description that represents the zero-location description L0.

    A complete composite location storage LS is created with C parts P\ :sub:`N`
    ordered in ascending N from 0 to C-1 inclusive. Each P\ :sub:`N` specifies
    location description PL\ :sub:`N` and has a bit size of S.

    PL\ :sub:`N` is as if the ``DW_OP_LLVM_bit_offset N*S`` operation was
    applied to PLX\ :sub:`N`\ .

    PLX\ :sub:`N` is the same as L0 if the N\ :sup:`th` least significant bit of
    M is a zero, otherwise it is the same as L1.

    A location description L comprised of one complete composite location
    description SL is pushed on the stack. SL specifies LS with a bit offset of
    0.

    The DWARF expression is ill-formed if S or C are 0, or if the bit size of M
    is less than C.

.. _amdgpu-dwarf-location-list-expressions:

DWARF Location List Expressions
+++++++++++++++++++++++++++++++

*To meet the needs of recent computer architectures and optimization techniques,
debugging information must be able to describe the location of an object whose
location changes over the object’s lifetime, and may reside at multiple
locations during parts of an object's lifetime. Location list expressions are
used in place of operation expressions whenever the object whose location is
being described has these requirements.*

A location list expression consists of a series of location list entries. Each
location list entry is one of the following kinds:

*Bounded location description*

  This kind of location list entry provides an operation expression that
  evaluates to the location description of an object that is valid over a
  lifetime bounded by a starting and ending address. The starting address is the
  lowest address of the address range over which the location is valid. The
  ending address is the address of the first location past the highest address
  of the address range.

  The location list entry matches when the current program location is within
  the given range.

  There are several kinds of bounded location description entries which differ
  in the way that they specify the starting and ending addresses.

*Default location description*

  This kind of location list entry provides an operation expression that
  evaluates to the location description of an object that is valid when no
  bounded location description entry applies.

  The location list entry matches when the current program location is not
  within the range of any bounded location description entry.

*Base address*

  This kind of location list entry provides an address to be used as the base
  address for beginning and ending address offsets given in certain kinds of
  bounded location description entries. The applicable base address of a bounded
  location description entry is the address specified by the closest preceding
  base address entry in the same location list. If there is no preceding base
  address entry, then the applicable base address defaults to the base address
  of the compilation unit (see DWARF Version 5 section 3.1.1).

  In the case of a compilation unit where all of the machine code is contained
  in a single contiguous section, no base address entry is needed.

*End-of-list*

  This kind of location list entry marks the end of the location list
  expression.

The address ranges defined by the bounded location description entries of a
location list expression may overlap. When they do, they describe a situation in
which an object exists simultaneously in more than one place.

If all of the address ranges in a given location list expression do not
collectively cover the entire range over which the object in question is
defined, and there is no following default location description entry, it is
assumed that the object is not available for the portion of the range that is
not covered.

The operation expression of each matching location list entry is evaluated as a
location description and its result is returned as the result of the location
list entry. The operation expression is evaluated with the same context as the
location list expression, including the same current frame, current program
location, and initial stack.

The result of the evaluation of a DWARF location list expression is a location
description that is comprised of the union of the single location descriptions
of the location description result of each matching location list entry. If
there are no matching location list entries, then the result is a location
description that comprises one undefined location description.

A location list expression can only be used as the value of a debugger
information entry attribute that is encoded using class ``loclist`` or
``loclistsptr`` (see DWARF Version 5 section 7.5.5). The value of the attribute
provides an index into a separate object file section called ``.debug_loclists``
or ``.debug_loclists.dwo`` (for split DWARF object files) that contains the
location list entries.

A ``DW_OP_call*`` and ``DW_OP_implicit_pointer`` operation can be used to
specify a debugger information entry attribute that has a location list
expression. Several debugger information entry attributes allow DWARF
expressions that are evaluated with an initial stack that includes a location
description that may originate from the evaluation of a location list
expression.

*This location list representation, the* ``loclist`` *and* ``loclistsptr``
*class, and the related* ``DW_AT_loclists_base`` *attribute are new in DWARF
Version 5. Together they eliminate most, or all of the code object relocations
previously needed for location list expressions.*

.. note::

  The rest of this section is the same as DWARF Version 5 section 2.6.2.

.. _amdgpu-dwarf-segment_addresses:

Segmented Addresses
~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 2.12.

DWARF address classes are used for source languages that have the concept of
memory spaces. They are used in the ``DW_AT_address_class`` attribute for
pointer type, reference type, subprogram, and subprogram type debugger
information entries.

Each DWARF address class is conceptually a separate source language memory space
with its own lifetime and aliasing rules. DWARF address classes are used to
specify the source language memory spaces that pointer type and reference type
values refer, and to specify the source language memory space in which variables
are allocated.

The set of currently defined source language DWARF address classes, together
with source language mappings, is given in
:ref:`amdgpu-dwarf-address-class-table`.

Vendor defined source language address classes may be defined using codes in the
range ``DW_ADDR_LLVM_lo_user`` to ``DW_ADDR_LLVM_hi_user``.

.. table:: Address class
   :name: amdgpu-dwarf-address-class-table

   ========================= ============ ========= ========= =========
   Address Class Name        Meaning      C/C++     OpenCL    CUDA/HIP
   ========================= ============ ========= ========= =========
   ``DW_ADDR_none``          generic      *default* generic   *default*
   ``DW_ADDR_LLVM_global``   global                 global
   ``DW_ADDR_LLVM_constant`` constant               constant  constant
   ``DW_ADDR_LLVM_group``    thread-group           local     shared
   ``DW_ADDR_LLVM_private``  thread                 private
   ``DW_ADDR_LLVM_lo_user``
   ``DW_ADDR_LLVM_hi_user``
   ========================= ============ ========= ========= =========

DWARF address spaces correspond to target architecture specific linear
addressable memory areas. They are used in DWARF expression location
descriptions to describe in which target architecture specific memory area data
resides.

*Target architecture specific DWARF address spaces may correspond to hardware
supported facilities such as memory utilizing base address registers, scratchpad
memory, and memory with special interleaving. The size of addresses in these
address spaces may vary. Their access and allocation may be hardware managed
with each thread or group of threads having access to independent storage. For
these reasons they may have properties that do not allow them to be viewed as
part of the unified global virtual address space accessible by all threads.*

*It is target architecture specific whether multiple DWARF address spaces are
supported and how source language DWARF address classes map to target
architecture specific DWARF address spaces. A target architecture may map
multiple source language DWARF address classes to the same target architecture
specific DWARF address class. Optimization may determine that variable lifetime
and access pattern allows them to be allocated in faster scratchpad memory
represented by a different DWARF address space.*

Although DWARF address space identifiers are target architecture specific,
``DW_ASPACE_none`` is a common address space supported by all target
architectures.

DWARF address space identifiers are used by:

* The DWARF expession operations: ``DW_OP_LLVM_aspace_bregx``,
  ``DW_OP_LLVM_form_aspace_address``, ``DW_OP_LLVM_implicit_aspace_pointer``,
  and ``DW_OP_xderef*``.

* The CFI instructions: ``DW_CFA_def_aspace_cfa`` and
  ``DW_CFA_def_aspace_cfa_sf``.

.. note::

  With the definition of DWARF address classes and DWARF address spaces in this
  proposal, DWARF Version 5 table 2.7 needs to be updated. It seems it is an
  example of DWARF address spaces and not DWARF address classes.

.. note::

  With the expanded support for DWARF address spaces in this proposal, it may be
  worth examining if DWARF segments can be eliminated and DWARF address spaces
  used instead.

  That may involve extending DWARF address spaces to also be used to specify
  code locations. In target architectures that use different memory areas for
  code and data this would seem a natural use for DWARF address spaces. This
  would allow DWARF expression location descriptions to be used to describe the
  location of subprograms and entry points that are used in expressions
  involving subprogram pointer type values.

  Currently, DWARF expressions assume data and code resides in the same default
  DWARF address space, and only the address ranges in DWARF location list
  entries and in the ``.debug_aranges`` section for accelerated access for
  addresses allow DWARF segments to be used to distinguish.

.. note::

  Currently, DWARF defines address class values as being target architecture
  specific. It is unclear how language specific memory spaces are intended to be
  represented in DWARF using these.

  For example, OpenCL defines memory spaces (called address spaces in OpenCL)
  for ``global``, ``local``, ``constant``, and ``private``. These are part of
  the type system and are modifiers to pointer types. In addition, OpenCL
  defines ``generic`` pointers that can reference either the ``global``,
  ``local``, or ``private`` memory spaces. To support the OpenCL language the
  debugger would want to support casting pointers between the ``generic`` and
  other memory spaces, querying what memory space a ``generic`` pointer value is
  currently referencing, and possibly using pointer casting to form an address
  for a specific memory space out of an integral value.

  The method to use to dereference a pointer type or reference type value is
  defined in DWARF expressions using ``DW_OP_xderef*`` which uses a target
  architecture specific address space.

  DWARF defines the ``DW_AT_address_class`` attribute on pointer type and
  reference type debugger information entries. It specifies the method to use to
  dereference them. Why is the value of this not the same as the address space
  value used in ``DW_OP_xderef*``? In both cases it is target architecture
  specific and the architecture presumably will use the same set of methods to
  dereference pointers in both cases.

  Since ``DW_AT_address_class`` uses a target architecture specific value, it
  cannot in general capture the source language memory space type modifier
  concept. On some architectures all source language memory space modifiers may
  actually use the same method for dereferencing pointers.

  One possibility is for DWARF to add an ``DW_TAG_LLVM_address_class_type``
  debugger information entry type modifier that can be applied to a pointer type
  and reference type. The ``DW_AT_address_class`` attribute could be re-defined
  to not be target architecture specific and instead define generalized language
  values (as is proposed above for DWARF address classes in the table
  :ref:`amdgpu-dwarf-address-class-table`) that will support OpenCL and other
  languages using memory spaces. The ``DW_AT_address_class`` attribute could be
  defined to not be applied to pointer types or reference types, but instead
  only to the new ``DW_TAG_LLVM_address_class_type`` type modifier debugger
  information entry.

  If a pointer type or reference type is not modified by
  ``DW_TAG_LLVM_address_class_type`` or if ``DW_TAG_LLVM_address_class_type``
  has no ``DW_AT_address_class`` attribute, then the pointer type or reference
  type would be defined to use the ``DW_ADDR_none`` address class as currently.
  Since modifiers can be chained, it would need to be defined if multiple
  ``DW_TAG_LLVM_address_class_type`` modifiers were legal, and if so if the
  outermost one is the one that takes precedence.

  A target architecture implementation that supports multiple address spaces
  would need to map ``DW_ADDR_none`` appropriately to support CUDA-like
  languages that have no address classes in the type system but do support
  variable allocation in address classes. Such variable allocation would result
  in the variable's location description needing an address space.

  The approach proposed in :ref:`amdgpu-dwarf-address-class-table` is to define
  the default ``DW_ADDR_none`` to be the generic address class and not the
  global address class. This matches how CLANG and LLVM have added support for
  CUDA-like languages on top of existing C++ language support. This allows all
  addresses to be generic by default which matches CUDA-like languages.

  An alternative approach is to define ``DW_ADDR_none`` as being the global
  address class and then change ``DW_ADDR_LLVM_global`` to
  ``DW_ADDR_LLVM_generic``. This would match the reality that languages that do
  not support multiple memory spaces only have one default global memory space.
  Generally, in these languages if they expose that the target architecture
  supports multiple address spaces, the default one is still the global memory
  space. Then a language that does support multiple memory spaces has to
  explicitly indicate which pointers have the added ability to reference more
  than the global memory space. However, compilers generating DWARF for
  CUDA-like languages would then have to define every CUDA-like language pointer
  type or reference type using ``DW_TAG_LLVM_address_class_type`` with a
  ``DW_AT_address_class`` attribute of ``DW_ADDR_LLVM_generic`` to match the
  language semantics.

  A new ``DW_AT_LLVM_address_space`` attribute could be defined that can be
  applied to pointer type, reference type, subprogram, and subprogram type to
  describe how objects having the given type are dereferenced or called (the
  role that ``DW_AT_address_class`` currently provides). The values of
  ``DW_AT_address_space`` would be target architecture specific and the same as
  used in ``DW_OP_xderef*``.

.. _amdgpu-dwarf-debugging-information-entry-attributes:

Debugging Information Entry Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This section provides changes to existing debugger information entry
  attributes and defines attributes added by the proposal. These would be
  incorporated into the appropriate DWARF Version 5 chapter 2 sections.

1.  ``DW_AT_location``

    Any debugging information entry describing a data object (which includes
    variables and parameters) or common blocks may have a ``DW_AT_location``
    attribute, whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description in the context of the current subprogram, current program
    location, and with an empty initial stack. See
    :ref:`amdgpu-dwarf-expressions`.

    See :ref:`amdgpu-dwarf-control-flow-operations` for special evaluation rules
    used by the ``DW_OP_call*`` operations.

    .. note::

      Delete the description of how the ``DW_OP_call*`` operations evaluate a
      ``DW_AT_location`` attribute as that is now described in the operations.

    .. note::

      See the discussion about the ``DW_AT_location`` attribute in the
      ``DW_OP_call*`` operation. Having each attribute only have a single
      purpose and single execution semantics seems desirable. It makes it easier
      for the consumer that no longer have to track the context. It makes it
      easier for the producer as it can rely on a single semantics for each
      attribute.

      For that reason, limiting the ``DW_AT_location`` attribute to only
      supporting evaluating the location description of an object, and using a
      different attribute and encoding class for the evaluation of DWARF
      expression *procedures* on the same operation expression stack seems
      desirable.

2.  ``DW_AT_const_value``

    .. note::

      Could deprecate using the ``DW_AT_const_value`` attribute for
      ``DW_TAG_variable`` or ``DW_TAG_formal_parameter`` debugger information
      entries that have been optimized to a constant. Instead,
      ``DW_AT_location`` could be used with a DWARF expression that produces an
      implicit location description now that any location description can be
      used within a DWARF expression. This allows the ``DW_OP_call*`` operations
      to be used to push the location description of any variable regardless of
      how it is optimized.

3.  ``DW_AT_frame_base``

    A ``DW_TAG_subprogram`` or ``DW_TAG_entry_point`` debugger information entry
    may have a ``DW_AT_frame_base`` attribute, whose value is a DWARF expression
    E.

    The result of the attribute is obtained by evaluating E as a location
    description in the context of the current subprogram, current program
    location, and with an empty initial stack.

    The DWARF is ill-formed if E contains an ``DW_OP_fbreg`` operation, or the
    resulting location description L is not comprised of one single location
    description SL.

    If SL a register location description for register R, then L is replaced
    with the result of evaluating a ``DW_OP_bregx R, 0`` operation. This
    computes the frame base memory location description in the target
    architecture default address space.

    *This allows the more compact* ``DW_OPreg*`` *to be used instead of*
    ``DW_OP_breg* 0``\ *.*

    .. note::

      This rule could be removed and require the producer to create the required
      location description directly using ``DW_OP_call_frame_cfa``,
      ``DW_OP_breg*``, or ``DW_OP_LLVM_aspace_bregx``. This would also then
      allow a target to implement the call frames within a large register.

    Otherwise, the DWARF is ill-formed if SL is not a memory location
    description in any of the target architecture specific address spaces.

    The resulting L is the *frame base* for the subprogram or entry point.

    *Typically, E will use the* ``DW_OP_call_frame_cfa`` *operation or be a
    stack pointer register plus or minus some offset.*

4.  ``DW_AT_data_member_location``

    For a ``DW_AT_data_member_location`` attribute there are two cases:

    1.  If the attribute is an integer constant B, it provides the offset in
        bytes from the beginning of the containing entity.

        The result of the attribute is obtained by evaluating a
        ``DW_OP_LLVM_offset B`` operation with an initial stack comprising the
        location description of the beginning of the containing entity.  The
        result of the evaluation is the location description of the base of the
        member entry.

        *If the beginning of the containing entity is not byte aligned, then the
        beginning of the member entry has the same bit displacement within a
        byte.*

    2.  Otherwise, the attribute must be a DWARF expression E which is evaluated
        with a context of the current frame, current program location, and an
        initial stack comprising the location description of the beginning of
        the containing entity. The result of the evaluation is the location
        description of the base of the member entry.

    .. note::

      The beginning of the containing entity can now be any location
      description, including those with more than one single location
      description, and those with single location descriptions that are of any
      kind and have any bit offset.

5.  ``DW_AT_use_location``

    The ``DW_TAG_ptr_to_member_type`` debugging information entry has a
    ``DW_AT_use_location`` attribute whose value is a DWARF expression E. It is
    used to compute the location description of the member of the class to which
    the pointer to member entry points.

    *The method used to find the location description of a given member of a
    class, structure, or union is common to any instance of that class,
    structure, or union and to any instance of the pointer to member type. The
    method is thus associated with the pointer to member type, rather than with
    each object that has a pointer to member type.*

    The ``DW_AT_use_location`` DWARF expression is used in conjunction with the
    location description for a particular object of the given pointer to member
    type and for a particular structure or class instance.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an initial stack comprising two entries. The first entry is
    the value of the pointer to member object itself. The second entry is the
    location description of the base of the entire class, structure, or union
    instance containing the member whose location is being calculated.

6.  ``DW_AT_data_location``

    The ``DW_AT_data_location`` attribute may be used with any type that
    provides one or more levels of hidden indirection and/or run-time parameters
    in its representation. Its value is a DWARF operation expression E which
    computes the location description of the data for an object. When this
    attribute is omitted, the location description of the data is the same as
    the location description of the object.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    *E will typically involve an operation expression that begins with a*
    ``DW_OP_push_object_address`` *operation which loads the location
    description of the object which can then serve as a description in
    subsequent calculation.*

    .. note::

      Since ``DW_AT_data_member_location``, ``DW_AT_use_location``, and
      ``DW_AT_vtable_elem_location`` allow both operation expressions and
      location list expressions, why does ``DW_AT_data_location`` not allow
      both? In all cases they apply to data objects so less likely that
      optimization would cause different operation expressions for different
      program location ranges. But if supporting for some then should be for
      all.

      It seems odd this attribute is not the same as
      ``DW_AT_data_member_location`` in having an initial stack with the
      location description of the object since the expression has to need it.

7.  ``DW_AT_vtable_elem_location``

    An entry for a virtual function also has a ``DW_AT_vtable_elem_location``
    attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an initial stack comprising the location description of the
    object of the enclosing type.

    The resulting location description is the slot for the function within the
    virtual function table for the enclosing class.

8.  ``DW_AT_static_link``

    If a ``DW_TAG_subprogram`` or ``DW_TAG_entry_point`` debugger information
    entry is lexically nested, it may have a ``DW_AT_static_link`` attribute,
    whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The DWARF is ill-formed if the resulting location description L is is not
    comprised of one memory location description in any of the target
    architecture specific address spaces.

    The resulting L is the *frame base* of the relevant instance of the
    subprogram that immediately lexically encloses the subprogram or entry
    point.

9.  ``DW_AT_return_addr``

    A ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_return_addr`` attribute, whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The DWARF is ill-formed if the resulting location description L is not
    comprised one memory location description in any of the target architecture
    specific address spaces.

    The resulting L is the place where the return address for the subprogram or
    entry point is stored.

    .. note::

      It is unclear why ``DW_TAG_inlined_subroutine`` has a
      ``DW_AT_return_addr`` attribute but not a ``DW_AT_frame_base`` or
      ``DW_AT_static_link`` attribute. Seems it would either have all of them or
      none. Since inlined subprograms do not have a frame it seems they would
      have none of these attributes.

10. ``DW_AT_call_value``, ``DW_AT_call_data_location``, and ``DW_AT_call_data_value``

    A ``DW_TAG_call_site_parameter`` debugger information entry may have a
    ``DW_AT_call_value`` attribute, whose value is a DWARF operation expression
    E\ :sub:`1`\ .

    The result of the ``DW_AT_call_value`` attribute is obtained by evaluating
    E\ :sub:`1` as a value with the context of the call site subprogram, call
    site program location, and an empty initial stack.

    The call site subprogram is the subprogram containing the
    ``DW_TAG_call_site_parameter`` debugger information entry. The call site
    program location is the location of call site in the call site subprogram.

    *The consumer may have to virtually unwind to the call site in order to
    evaluate the attribute. This will provide both the call site subprogram and
    call site program location needed to evaluate the expression.*

    The resulting value V\ :sub:`1` is the value of the parameter at the time of
    the call made by the call site.

    For parameters passed by reference, where the code passes a pointer to a
    location which contains the parameter, or for reference type parameters, the
    ``DW_TAG_call_site_parameter`` debugger information entry may also have a
    ``DW_AT_call_data_location`` attribute whose value is a DWARF operation
    expression E\ :sub:`2`\ , and a ``DW_AT_call_data_value`` attribute whose
    value is a DWARF operation expression E\ :sub:`3`\ .

    The value of the ``DW_AT_call_data_location`` attribute is obtained by
    evaluating E\ :sub:`2` as a location description with the context of the
    call site subprogram, call site program location, and an empty initial
    stack.

    The resulting location description L\ :sub:`2` is the location where the
    referenced parameter lives during the call made by the call site. If E\
    :sub:`2` would just be a ``DW_OP_push_object_address``, then the
    ``DW_AT_call_data_location`` attribute may be omitted.

    The value of the ``DW_AT_call_data_value`` attribute is obtained by
    evaluating E\ :sub:`3` as a value with the context of the call site
    subprogram, call site program location, and an empty initial stack.

    The resulting value V\ :sub:`3` is the value in L\ :sub:`2` at the time of
    the call made by the call site.

    If it is not possible to avoid the expressions of these attributes from
    accessing registers or memory locations that might be clobbered by the
    subprogram being called by the call site, then the associated attribute
    should not be provided.

    *The reason for the restriction is that the parameter may need to be
    accessed during the execution of the callee. The consumer may virtually
    unwind from the called subprogram back to the caller and then evaluate the
    attribute expressions. The call frame information (see*
    :ref:`amdgpu-dwarf-call-frame-information`\ *) will not be able to restore
    registers that have been clobbered, and clobbered memory will no longer have
    the value at the time of the call.*

11. ``DW_AT_LLVM_lanes`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_LLVM_lanes`` attribute whose value is an integer constant that is
    the number of lanes per thread. This is the static number of lanes per
    thread. It is not the dynamic number of lanes with which the thread was
    initiated, for example, due to smaller or partial work-groups.

    If not present, the default value of 1 is used.

    The DWARF is ill-formed if the value is 0.

12. ``DW_AT_LLVM_lane_pc`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugging information entry may have a
    ``DW_AT_LLVM_lane_pc`` attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The resulting location description L is for a thread lane count sized vector
    of generic type elements. The thread lane count is the value of the
    ``DW_AT_LLVM_lanes`` attribute. Each element holds the conceptual program
    location of the corresponding lane, where the least significant element
    corresponds to the first target architecture specific lane identifier and so
    forth. If the lane was not active when the current subprogram was called,
    its element is an undefined location description.

    ``DW_AT_LLVM_lane_pc`` *allows the compiler to indicate conceptually where
    each lane of a SIMT thread is positioned even when it is in divergent
    control flow that is not active.*

    *Typically, the result is a location description with one composite location
    description with each part being a location description with either one
    undefined location description or one memory location description.*

    If not present, the thread is not being used in a SIMT manner, and the
    thread's current program location is used.

13. ``DW_AT_LLVM_active_lane`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_LLVM_active_lane`` attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a value with the
    context of the current subprogram, current program location, and an empty
    initial stack.

    The DWARF is ill-formed if the resulting value V is not an integral value.

    The resulting V is a bit mask of active lanes for the current program
    location. The N\ :sup:`th` least significant bit of the mask corresponds to
    the N\ :sup:`th` lane. If the bit is 1 the lane is active, otherwise it is
    inactive.

    *Some targets may update the target architecture execution mask for regions
    of code that must execute with different sets of lanes than the current
    active lanes. For example, some code must execute with all lanes made
    temporarily active.* ``DW_AT_LLVM_active_lane`` *allows the compiler to
    provide the means to determine the source language active lanes.*

    If not present and ``DW_AT_LLVM_lanes`` is greater than 1, then the target
    architecture execution mask is used.

14. ``DW_AT_LLVM_vector_size`` *New*

    A ``DW_TAG_base_type`` debugger information entry for a base type T may have
    a ``DW_AT_LLVM_vector_size`` attribute whose value is an integer constant
    that is the vector type size N.

    The representation of a vector base type is as N contiguous elements, each
    one having the representation of a base type T' that is the same as T
    without the ``DW_AT_LLVM_vector_size`` attribute.

    If a ``DW_TAG_base_type`` debugger information entry does not have a
    ``DW_AT_LLVM_vector_size`` attribute, then the base type is not a vector
    type.

    The DWARF is ill-formed if N is not greater than 0.

    .. note::

      LLVM has mention of a non-upstreamed debugger information entry that is
      intended to support vector types. However, that was not for a base type so
      would not be suitable as the type of a stack value entry. But perhaps that
      could be replaced by using this attribute.

15. ``DW_AT_LLVM_augmentation`` *New*

    A ``DW_TAG_compile_unit`` debugger information entry for a compilation unit
    may have a ``DW_AT_LLVM_augmentation`` attribute, whose value is an
    augmentation string.

    *The augmentation string allows producers to indicate that there is
    additional vendor or target specific information in the debugging
    information entries. For example, this might be information about the
    version of vendor specific extensions that are being used.*

    If not present, or if the string is empty, then the compilation unit has no
    augmentation string.

    The format for the augmentation string is:

      | ``[``\ *vendor*\ ``v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

    Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
    version number of the extensions used, and *options* is an optional string
    providing additional information about the extensions. The version number
    must conform to [SEMVER]_. The *options* string must not contain the "\
    ``]``\ " character.

    For example:

      ::

        [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

Program Scope Entities
----------------------

.. _amdgpu-dwarf-language-names:

Unit Entities
~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 3.1.1 and Table 3.1.

Additional language codes defined for use with the ``DW_AT_language`` attribute
are defined in :ref:`amdgpu-dwarf-language-names-table`.

.. table:: Language Names
   :name: amdgpu-dwarf-language-names-table

   ==================== =============================
   Language Name        Meaning
   ==================== =============================
   ``DW_LANG_LLVM_HIP`` HIP Language.
   ==================== =============================

The ``DW_LANG_LLVM_HIP`` language can be supported by extending the C++
language. See [HIP]_.

Other Debugger Information
--------------------------

Accelerated Access
~~~~~~~~~~~~~~~~~~

.. _amdgpu-dwarf-lookup-by-name:

Lookup By Name
++++++++++++++

Contents of the Name Index
##########################

.. note::

  The following provides changes to DWARF Version 5 section 6.1.1.1.

  The rule for debugger information entries included in the name index in the
  optional ``.debug_names`` section is extended to also include named
  ``DW_TAG_variable`` debugging information entries with a ``DW_AT_location``
  attribute that includes a ``DW_OP_LLVM_form_aspace_address`` operation.

The name index must contain an entry for each debugging information entry that
defines a named subprogram, label, variable, type, or namespace, subject to the
following rules:

* ``DW_TAG_variable`` debugging information entries with a ``DW_AT_location``
  attribute that includes a ``DW_OP_addr``, ``DW_OP_LLVM_form_aspace_address``,
  or ``DW_OP_form_tls_address`` operation are included; otherwise, they are
  excluded.

Data Representation of the Name Index
#####################################

Section Header
^^^^^^^^^^^^^^

.. note::

  The following provides an addition to DWARF Version 5 section 6.1.1.4.1 item
  14 ``augmentation_string``.

A null-terminated UTF-8 vendor specific augmentation string, which provides
additional information about the contents of this index. If provided, the
recommended format for augmentation string is:

  | ``[``\ *vendor*\ ``v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
version number of the extensions used in the DWARF of the compilation unit, and
*options* is an optional string providing additional information about the
extensions. The version number must conform to [SEMVER]_. The *options* string
must not contain the "\ ``]``\ " character.

For example:

  ::

    [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

.. note::

  This is different to the definition in DWARF Version 5 but is consistent with
  the other augmentation strings and allows multiple vendor extensions to be
  supported.

.. _amdgpu-dwarf-line-number-information:

Line Number Information
~~~~~~~~~~~~~~~~~~~~~~~

The Line Number Program Header
++++++++++++++++++++++++++++++

Standard Content Descriptions
#############################

.. note::

  This augments DWARF Version 5 section 6.2.4.1.

.. _amdgpu-dwarf-line-number-information-dw-lnct-llvm-source:

1.  ``DW_LNCT_LLVM_source``

    The component is a null-terminated UTF-8 source text string with "\ ``\n``\
    " line endings. This content code is paired with the same forms as
    ``DW_LNCT_path``. It can be used for file name entries.

    The value is an empty null-terminated string if no source is available. If
    the source is available but is an empty file then the value is a
    null-terminated single "\ ``\n``\ ".

    *When the source field is present, consumers can use the embedded source
    instead of attempting to discover the source on disk using the file path
    provided by the* ``DW_LNCT_path`` *field. When the source field is absent,
    consumers can access the file to get the source text.*

    *This is particularly useful for programing languages that support runtime
    compilation and runtime generation of source text. In these cases, the
    source text does not reside in any permanent file. For example, the OpenCL
    language supports online compilation.*

2.  ``DW_LNCT_LLVM_is_MD5``

    ``DW_LNCT_LLVM_is_MD5`` indicates if the ``DW_LNCT_MD5`` content kind, if
    present, is valid: when 0 it is not valid and when 1 it is valid. If
    ``DW_LNCT_LLVM_is_MD5`` content kind is not present, and ``DW_LNCT_MD5``
    content kind is present, then the MD5 checksum is valid.

    ``DW_LNCT_LLVM_is_MD5`` is always paired with the ``DW_FORM_udata`` form.

    *This allows a compilation unit to have a mixture of files with and without
    MD5 checksums. This can happen when multiple relocatable files are linked
    together.*

.. _amdgpu-dwarf-call-frame-information:

Call Frame Information
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This section provides changes to existing Call Frame Information and defines
  instructions added by the proposal. Additional support is added for address
  spaces. Register unwind DWARF expressions are generalized to allow any
  location description, including those with composite and implicit location
  descriptions.

  These changes would be incorporated into the DWARF Version 5 section 6.1.

Structure of Call Frame Information
+++++++++++++++++++++++++++++++++++

The register rules are:

*undefined*
  A register that has this rule has no recoverable value in the previous frame.
  (By convention, it is not preserved by a callee.)

*same value*
  This register has not been modified from the previous frame. (By convention,
  it is preserved by the callee, but the callee has not modified it.)

*offset(N)*
  N is a signed byte offset. The previous value of this register is saved at the
  location description computed as if the DWARF operation expression
  ``DW_OP_LLVM_offset N`` is evaluated as a location description with an initial
  stack comprising the location description of the current CFA (see
  :ref:`amdgpu-dwarf-operation-expressions`).

*val_offset(N)*
  N is a signed byte offset. The previous value of this register is the memory
  byte address of the location description computed as if the DWARF operation
  expression ``DW_OP_LLVM_offset N`` is evaluated as a location description with
  an initial stack comprising the location description of the current CFA (see
  :ref:`amdgpu-dwarf-operation-expressions`).

  The DWARF is ill-formed if the CFA location description is not a memory byte
  address location description, or if the register size does not match the size
  of an address in the address space of the current CFA location description.

  *Since the CFA location description is required to be a memory byte address
  location description, the value of val_offset(N) will also be a memory byte
  address location description since it is offsetting the CFA location
  description by N bytes. Furthermore, the value of val_offset(N) will be a
  memory byte address in the same address space as the CFA location
  description.*

  .. note::

    Should DWARF allow the address size to be a different size to the size of
    the register? Requiring them to be the same bit size avoids any issue of
    conversion as the bit contents of the register is simply interpreted as a
    value of the address.

    Gdb has a per register hook that allows a target specific conversion on a
    register by register basis. It defaults to truncation of bigger registers,
    and to actually reading bytes from the next register (or reads out of bounds
    for the last register) for smaller registers. There are no gdb tests that
    read a register out of bounds (except an illegal hand written assembly
    test).

*register(R)*
  The previous value of this register is stored in another register numbered R.

  The DWARF is ill-formed if the register sizes do not match.

*expression(E)*
  The previous value of this register is located at the location description
  produced by evaluating the DWARF operation expression E (see
  :ref:`amdgpu-dwarf-operation-expressions`).

  E is evaluated as a location description in the context of the current
  subprogram, current program location, and with an initial stack comprising the
  location description of the current CFA.

*val_expression(E)*
  The previous value of this register is the value produced by evaluating the
  DWARF operation expression E (see :ref:`amdgpu-dwarf-operation-expressions`).

  E is evaluated as a value in the context of the current subprogram, current
  program location, and with an initial stack comprising the location
  description of the current CFA.

  The DWARF is ill-formed if the resulting value type size does not match the
  register size.

  .. note::

    This has limited usefulness as the DWARF expression E can only produce
    values up to the size of the generic type. This is due to not allowing any
    operations that specify a type in a CFI operation expression. This makes it
    unusable for registers that are larger than the generic type. However,
    *expression(E)* can be used to create an implicit location description of
    any size.

*architectural*
  The rule is defined externally to this specification by the augmenter.

A Common Information Entry holds information that is shared among many Frame
Description Entries. There is at least one CIE in every non-empty
``.debug_frame`` section. A CIE contains the following fields, in order:

1.  ``length`` (initial length)

    A constant that gives the number of bytes of the CIE structure, not
    including the length field itself. The size of the length field plus the
    value of length must be an integral multiple of the address size specified
    in the ``address_size`` field.

2.  ``CIE_id`` (4 or 8 bytes, see
    :ref:`amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats`)

    A constant that is used to distinguish CIEs from FDEs.

    In the 32-bit DWARF format, the value of the CIE id in the CIE header is
    0xffffffff; in the 64-bit DWARF format, the value is 0xffffffffffffffff.

3.  ``version`` (ubyte)

    A version number. This number is specific to the call frame information and
    is independent of the DWARF version number.

    The value of the CIE version number is 4.

    .. note::

      Would this be increased to 5 to reflect the changes in the proposal?

4.  ``augmentation`` (sequence of UTF-8 characters)

    A null-terminated UTF-8 string that identifies the augmentation to this CIE
    or to the FDEs that use it. If a reader encounters an augmentation string
    that is unexpected, then only the following fields can be read:

    * CIE: length, CIE_id, version, augmentation
    * FDE: length, CIE_pointer, initial_location, address_range

    If there is no augmentation, this value is a zero byte.

    *The augmentation string allows users to indicate that there is additional
    vendor and target architecture specific information in the CIE or FDE which
    is needed to virtually unwind a stack frame. For example, this might be
    information about dynamically allocated data which needs to be freed on exit
    from the routine.*

    *Because the* ``.debug_frame`` *section is useful independently of any*
    ``.debug_info`` *section, the augmentation string always uses UTF-8
    encoding.*

    The recommended format for the augmentation string is:

      | ``[``\ *vendor*\ ``v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

    Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
    version number of the extensions used, and *options* is an optional string
    providing additional information about the extensions. The version number
    must conform to [SEMVER]_. The *options* string must not contain the "\
    ``]``\ " character.

    For example:

      ::

        [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

5.  ``address_size`` (ubyte)

    The size of a target address in this CIE and any FDEs that use it, in bytes.
    If a compilation unit exists for this frame, its address size must match the
    address size here.

6.  ``segment_selector_size`` (ubyte)

    The size of a segment selector in this CIE and any FDEs that use it, in
    bytes.

7.  ``code_alignment_factor`` (unsigned LEB128)

    A constant that is factored out of all advance location instructions (see
    :ref:`amdgpu-dwarf-row-creation-instructions`). The resulting value is
    ``(operand * code_alignment_factor)``.

8.  ``data_alignment_factor`` (signed LEB128)

    A constant that is factored out of certain offset instructions (see
    :ref:`amdgpu-dwarf-cfa-definition-instructions` and
    :ref:`amdgpu-dwarf-register-rule-instructions`). The resulting value is
    ``(operand * data_alignment_factor)``.

9.  ``return_address_register`` (unsigned LEB128)

    An unsigned LEB128 constant that indicates which column in the rule table
    represents the return address of the subprogram. Note that this column might
    not correspond to an actual machine register.

10. ``initial_instructions`` (array of ubyte)

    A sequence of rules that are interpreted to create the initial setting of
    each column in the table.

    The default rule for all columns before interpretation of the initial
    instructions is the undefined rule. However, an ABI authoring body or a
    compilation system authoring body may specify an alternate default value for
    any or all columns.

11. ``padding`` (array of ubyte)

    Enough ``DW_CFA_nop`` instructions to make the size of this entry match the
    length value above.

An FDE contains the following fields, in order:

1.  ``length`` (initial length)

    A constant that gives the number of bytes of the header and instruction
    stream for this subprogram, not including the length field itself. The size
    of the length field plus the value of length must be an integral multiple of
    the address size.

2.  ``CIE_pointer`` (4 or 8 bytes, see
    :ref:`amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats`)

    A constant offset into the ``.debug_frame`` section that denotes the CIE
    that is associated with this FDE.

3.  ``initial_location`` (segment selector and target address)

    The address of the first location associated with this table entry. If the
    segment_selector_size field of this FDE’s CIE is non-zero, the initial
    location is preceded by a segment selector of the given length.

4.  ``address_range`` (target address)

    The number of bytes of program instructions described by this entry.

5.  ``instructions`` (array of ubyte)

    A sequence of table defining instructions that are described in
    :ref:`amdgpu-dwarf-call-frame-instructions`.

6.  ``padding`` (array of ubyte)

    Enough ``DW_CFA_nop`` instructions to make the size of this entry match the
    length value above.

.. _amdgpu-dwarf-call-frame-instructions:

Call Frame Instructions
+++++++++++++++++++++++

Some call frame instructions have operands that are encoded as DWARF operation
expressions E (see :ref:`amdgpu-dwarf-operation-expressions`). The DWARF
operations that can be used in E have the following restrictions:

* ``DW_OP_addrx``, ``DW_OP_call2``, ``DW_OP_call4``, ``DW_OP_call_ref``,
  ``DW_OP_const_type``, ``DW_OP_constx``, ``DW_OP_convert``,
  ``DW_OP_deref_type``, ``DW_OP_fbreg``, ``DW_OP_implicit_pointer``,
  ``DW_OP_regval_type``, ``DW_OP_reinterpret``, and ``DW_OP_xderef_type``
  operations are not allowed because the call frame information must not depend
  on other debug sections.

* ``DW_OP_push_object_address`` is not allowed because there is no object
  context to provide a value to push.

* ``DW_OP_LLVM_push_lane`` is not allowed because the call frame instructions
  describe the actions for the whole thread, not the lanes independently.

* ``DW_OP_call_frame_cfa`` and ``DW_OP_entry_value`` are not allowed because
  their use would be circular.

* ``DW_OP_LLVM_call_frame_entry_reg`` is not allowed if evaluating E causes a
  circular dependency between ``DW_OP_LLVM_call_frame_entry_reg`` operations.

  *For example, if a register R1 has a* ``DW_CFA_def_cfa_expression``
  *instruction that evaluates a* ``DW_OP_LLVM_call_frame_entry_reg`` *operation
  that specifies register R2, and register R2 has a*
  ``DW_CFA_def_cfa_expression`` *instruction that that evaluates a*
  ``DW_OP_LLVM_call_frame_entry_reg`` *operation that specifies register R1.*

*Call frame instructions to which these restrictions apply include*
``DW_CFA_def_cfa_expression``\ *,* ``DW_CFA_expression``\ *, and*
``DW_CFA_val_expression``\ *.*

.. _amdgpu-dwarf-row-creation-instructions:

Row Creation Instructions
#########################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.1.

.. _amdgpu-dwarf-cfa-definition-instructions:

CFA Definition Instructions
###########################

1.  ``DW_CFA_def_cfa``

    The ``DW_CFA_def_cfa`` instruction takes two unsigned LEB128 operands
    representing a register number R and a (non-factored) byte displacement B.
    AS is set to the target architecture default address space identifier. The
    required action is to define the current CFA rule to be the result of
    evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B`` as a location description.

2.  ``DW_CFA_def_cfa_sf``

    The ``DW_CFA_def_cfa_sf`` instruction takes two operands: an unsigned LEB128
    value representing a register number R and a signed LEB128 factored byte
    displacement B. AS is set to the target architecture default address space
    identifier. The required action is to define the current CFA rule to be the
    result of evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B*data_alignment_factor`` as a location description.

    *The action is the same as* ``DW_CFA_def_cfa`` *except that the second
    operand is signed and factored.*

3.  ``DW_CFA_def_aspace_cfa`` *New*

    The ``DW_CFA_def_aspace_cfa`` instruction takes three unsigned LEB128
    operands representing a register number R, a (non-factored) byte
    displacement B, and a target architecture specific address space identifier
    AS. The required action is to define the current CFA rule to be the result
    of evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B`` as a location description.

    If AS is not one of the values defined by the target architecture specific
    ``DW_ASPACE_*`` values then the DWARF expression is ill-formed.

4.  ``DW_CFA_def_aspace_cfa_sf`` *New*

    The ``DW_CFA_def_cfa_sf`` instruction takes three operands: an unsigned
    LEB128 value representing a register number R, a signed LEB128 factored byte
    displacement B, and an unsigned LEB128 value representing a target
    architecture specific address space identifier AS. The required action is to
    define the current CFA rule to be the result of evaluating the DWARF
    operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R,
    B*data_alignment_factor`` as a location description.

    If AS is not one of the values defined by the target architecture specific
    ``DW_ASPACE_*`` values, then the DWARF expression is ill-formed.

    *The action is the same as* ``DW_CFA_aspace_def_cfa`` *except that the
    second operand is signed and factored.*

5.  ``DW_CFA_def_cfa_register``

    The ``DW_CFA_def_cfa_register`` instruction takes a single unsigned LEB128
    operand representing a register number R. The required action is to define
    the current CFA rule to be the result of evaluating the DWARF operation
    expression ``DW_OP_constu AS; DW_OP_aspace_bregx R, B`` as a location
    description. B and AS are the old CFA byte displacement and address space
    respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

6.  ``DW_CFA_def_cfa_offset``

    The ``DW_CFA_def_cfa_offset`` instruction takes a single unsigned LEB128
    operand representing a (non-factored) byte displacement B. The required
    action is to define the current CFA rule to be the result of evaluating the
    DWARF operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R, B`` as a
    location description. R and AS are the old CFA register number and address
    space respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

7.  ``DW_CFA_def_cfa_offset_sf``

    The ``DW_CFA_def_cfa_offset_sf`` instruction takes a signed LEB128 operand
    representing a factored byte displacement B. The required action is to
    define the current CFA rule to be the result of evaluating the DWARF
    operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R,
    B*data_alignment_factor`` as a location description. R and AS are the old
    CFA register number and address space respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

    *The action is the same as* ``DW_CFA_def_cfa_offset`` *except that the
    operand is signed and factored.*

8.  ``DW_CFA_def_cfa_expression``

    The ``DW_CFA_def_cfa_expression`` instruction takes a single operand encoded
    as a ``DW_FORM_exprloc`` value representing a DWARF operation expression E.
    The required action is to define the current CFA rule to be the result of
    evaluating E as a location description in the context of the current
    subprogram, current program location, and an empty initial stack.

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

    The DWARF is ill-formed if the result of evaluating E is not a memory byte
    address location description.

.. _amdgpu-dwarf-register-rule-instructions:

Register Rule Instructions
##########################

1.  ``DW_CFA_undefined``

    The ``DW_CFA_undefined`` instruction takes a single unsigned LEB128 operand
    that represents a register number R. The required action is to set the rule
    for the register specified by R to ``undefined``.

2.  ``DW_CFA_same_value``

    The ``DW_CFA_same_value`` instruction takes a single unsigned LEB128 operand
    that represents a register number R. The required action is to set the rule
    for the register specified by R to ``same value``.

3.  ``DW_CFA_offset``

    The ``DW_CFA_offset`` instruction takes two operands: a register number R
    (encoded with the opcode) and an unsigned LEB128 constant representing a
    factored displacement B. The required action is to change the rule for the
    register specified by R to be an *offset(B\*data_alignment_factor)* rule.

    .. note::

      Seems this should be named ``DW_CFA_offset_uf`` since the offset is
      unsigned factored.

4.  ``DW_CFA_offset_extended``

    The ``DW_CFA_offset_extended`` instruction takes two unsigned LEB128
    operands representing a register number R and a factored displacement B.
    This instruction is identical to ``DW_CFA_offset`` except for the encoding
    and size of the register operand.

    .. note::

      Seems this should be named ``DW_CFA_offset_extended_uf`` since the
      displacement is unsigned factored.

5.  ``DW_CFA_offset_extended_sf``

    The ``DW_CFA_offset_extended_sf`` instruction takes two operands: an
    unsigned LEB128 value representing a register number R and a signed LEB128
    factored displacement B. This instruction is identical to
    ``DW_CFA_offset_extended`` except that B is signed.

6.  ``DW_CFA_val_offset``

    The ``DW_CFA_val_offset`` instruction takes two unsigned LEB128 operands
    representing a register number R and a factored displacement B. The required
    action is to change the rule for the register indicated by R to be a
    *val_offset(B\*data_alignment_factor)* rule.

    .. note::

      Seems this should be named ``DW_CFA_val_offset_uf`` since the displacement
      is unsigned factored.

    .. note::

      An alternative is to define ``DW_CFA_val_offset`` to implicitly use the
      target architecture default address space, and add another operation that
      specifies the address space.

7.  ``DW_CFA_val_offset_sf``

    The ``DW_CFA_val_offset_sf`` instruction takes two operands: an unsigned
    LEB128 value representing a register number R and a signed LEB128 factored
    displacement B. This instruction is identical to ``DW_CFA_val_offset``
    except that B is signed.

8.  ``DW_CFA_register``

    The ``DW_CFA_register`` instruction takes two unsigned LEB128 operands
    representing register numbers R1 and R2 respectively. The required action is
    to set the rule for the register specified by R1 to be a *register(R2)* rule.

9.  ``DW_CFA_expression``

    The ``DW_CFA_expression`` instruction takes two operands: an unsigned LEB128
    value representing a register number R, and a ``DW_FORM_block`` value
    representing a DWARF operation expression E. The required action is to
    change the rule for the register specified by R to be an *expression(E)*
    rule.

    *That is, E computes the location description where the register value can
    be retrieved.*

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

10. ``DW_CFA_val_expression``

    The ``DW_CFA_val_expression`` instruction takes two operands: an unsigned
    LEB128 value representing a register number R, and a ``DW_FORM_block`` value
    representing a DWARF operation expression E. The required action is to
    change the rule for the register specified by R to be a *val_expression(E)*
    rule.

    *That is, E computes the value of register R.*

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

    If the result of evaluating E is not a value with a base type size that
    matches the register size, then the DWARF is ill-formed.

11. ``DW_CFA_restore``

    The ``DW_CFA_restore`` instruction takes a single operand (encoded with the
    opcode) that represents a register number R. The required action is to
    change the rule for the register specified by R to the rule assigned it by
    the ``initial_instructions`` in the CIE.

12. ``DW_CFA_restore_extended``

    The ``DW_CFA_restore_extended`` instruction takes a single unsigned LEB128
    operand that represents a register number R. This instruction is identical
    to ``DW_CFA_restore`` except for the encoding and size of the register
    operand.

Row State Instructions
######################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.4.

Padding Instruction
###################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.5.

Call Frame Instruction Usage
++++++++++++++++++++++++++++

.. note::

  The same as in DWARF Version 5 section 6.4.3.

.. _amdgpu-dwarf-call-frame-calling-address:

Call Frame Calling Address
++++++++++++++++++++++++++

.. note::

  The same as in DWARF Version 5 section 6.4.4.

Data Representation
-------------------

.. _amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats:

32-Bit and 64-Bit DWARF Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.4.

1.  Within the body of the ``.debug_info`` section, certain forms of attribute
    value depend on the choice of DWARF format as follows. For the 32-bit DWARF
    format, the value is a 4-byte unsigned integer; for the 64-bit DWARF format,
    the value is an 8-byte unsigned integer.

    .. table:: ``.debug_info`` section attribute form roles
      :name: amdgpu-dwarf-debug-info-section-attribute-form-roles-table

      ================================== ===================================
      Form                               Role
      ================================== ===================================
      DW_FORM_line_strp                  offset in ``.debug_line_str``
      DW_FORM_ref_addr                   offset in ``.debug_info``
      DW_FORM_sec_offset                 offset in a section other than
                                         ``.debug_info`` or ``.debug_str``
      DW_FORM_strp                       offset in ``.debug_str``
      DW_FORM_strp_sup                   offset in ``.debug_str`` section of
                                         supplementary object file
      DW_OP_call_ref                     offset in ``.debug_info``
      DW_OP_implicit_pointer             offset in ``.debug_info``
      DW_OP_LLVM_aspace_implicit_pointer offset in ``.debug_info``
      ================================== ===================================

Format of Debugging Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attribute Encodings
+++++++++++++++++++

.. note::

  This augments DWARF Version 5 section 7.5.4 and Table 7.5.

The following table gives the encoding of the additional debugging information
entry attributes.

.. table:: Attribute encodings
   :name: amdgpu-dwarf-attribute-encodings-table

   ================================== ===== ====================================
   Attribute Name                     Value Classes
   ================================== ===== ====================================
   DW_AT_LLVM_active_lane             *TBD* exprloc, loclist
   DW_AT_LLVM_augmentation            *TBD* string
   DW_AT_LLVM_lanes                   *TBD* constant
   DW_AT_LLVM_lane_pc                 *TBD* exprloc, loclist
   DW_AT_LLVM_vector_size             *TBD* constant
   ================================== ===== ====================================

DWARF Expressions
~~~~~~~~~~~~~~~~~

.. note::

  Rename DWARF Version 5 section 7.7 to reflect the unification of location
  descriptions into DWARF expressions.

Operation Expressions
+++++++++++++++++++++

.. note::

  Rename DWARF Version 5 section 7.7.1 and delete section 7.7.2 to reflect the
  unification of location descriptions into DWARF expressions.

  This augments DWARF Version 5 section 7.7.1 and Table 7.9.

The following table gives the encoding of the additional DWARF expression
operations.

.. table:: DWARF Operation Encodings
   :name: amdgpu-dwarf-operation-encodings-table

   ================================== ===== ======== ===============================
   Operation                          Code  Number   Notes
                                            of
                                            Operands
   ================================== ===== ======== ===============================
   DW_OP_LLVM_form_aspace_address     0xe1     0
   DW_OP_LLVM_push_lane               0xe2     0
   DW_OP_LLVM_offset                  0xe3     0
   DW_OP_LLVM_offset_constu           0xe4     1     ULEB128 byte displacement
   DW_OP_LLVM_bit_offset              0xe5     0
   DW_OP_LLVM_call_frame_entry_reg    0xe6     1     ULEB128 register number
   DW_OP_LLVM_undefined               0xe7     0
   DW_OP_LLVM_aspace_bregx            0xe8     2     ULEB128 register number,
                                                     ULEB128 byte displacement
   DW_OP_LLVM_aspace_implicit_pointer 0xe9     2     4- or 8-byte offset of DIE,
                                                     SLEB128 byte displacement
   DW_OP_LLVM_piece_end               0xea     0
   DW_OP_LLVM_extend                  0xeb     2     ULEB128 bit size,
                                                     ULEB128 count
   DW_OP_LLVM_select_bit_piece        0xec     2     ULEB128 bit size,
                                                     ULEB128 count
   ================================== ===== ======== ===============================

Location List Expressions
+++++++++++++++++++++++++

.. note::

  Rename DWARF Version 5 section 7.7.3 to reflect that location lists are a kind
  of DWARF expression.

Source Languages
~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.12 and Table 7.17.

The following table gives the encoding of the additional DWARF languages.

.. table:: Language encodings
   :name: amdgpu-dwarf-language-encodings-table

   ==================== ====== ===================
   Language Name        Value  Default Lower Bound
   ==================== ====== ===================
   ``DW_LANG_LLVM_HIP`` 0x8100 0
   ==================== ====== ===================

Address Class and Address Space Encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This replaces DWARF Version 5 section 7.13.

The encodings of the constants used for the currently defined address classes
are given in :ref:`amdgpu-dwarf-address-class-encodings-table`.

.. table:: Address class encodings
   :name: amdgpu-dwarf-address-class-encodings-table

   ========================== ======
   Address Class Name         Value
   ========================== ======
   ``DW_ADDR_none``           0x0000
   ``DW_ADDR_LLVM_global``    0x0001
   ``DW_ADDR_LLVM_constant``  0x0002
   ``DW_ADDR_LLVM_group``     0x0003
   ``DW_ADDR_LLVM_private``   0x0004
   ``DW_ADDR_LLVM_lo_user``   0x8000
   ``DW_ADDR_LLVM_hi_user``   0xffff
   ========================== ======

Line Number Information
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.22 and Table 7.27.

The following table gives the encoding of the additional line number header
entry formats.

.. table:: Line number header entry format encodings
  :name: amdgpu-dwarf-line-number-header-entry-format-encodings-table

  ====================================  ====================
  Line number header entry format name  Value
  ====================================  ====================
  ``DW_LNCT_LLVM_source``               0x2001
  ``DW_LNCT_LLVM_is_MD5``               0x2002
  ====================================  ====================

Call Frame Information
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.24 and Table 7.29.

The following table gives the encoding of the additional call frame information
instructions.

.. table:: Call frame instruction encodings
   :name: amdgpu-dwarf-call-frame-instruction-encodings-table

   ======================== ====== ====== ================ ================ ================
   Instruction              High 2 Low 6  Operand 1        Operand 2        Operand 3
                            Bits   Bits
   ======================== ====== ====== ================ ================ ================
   DW_CFA_def_aspace_cfa    0      0x2f   ULEB128 register ULEB128 offset   ULEB128 address space
   DW_CFA_def_aspace_cfa_sf 0      0x30   ULEB128 register SLEB128 offset   ULEB128 address space
   ======================== ====== ====== ================ ================ ================

Attributes by Tag Value (Informative)
-------------------------------------

.. note::

  This augments DWARF Version 5 Appendix A and Table A.1.

The following table provides the additional attributes that are applicable to
debugger information entries.

.. table:: Attributes by tag value
   :name: amdgpu-dwarf-attributes-by-tag-value-table

   ============================= =============================
   Tag Name                      Applicable Attributes
   ============================= =============================
   ``DW_TAG_base_type``          * ``DW_AT_LLVM_vector_size``
   ``DW_TAG_compile_unit``       * ``DW_AT_LLVM_augmentation``
   ``DW_TAG_entry_point``        * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ``DW_TAG_inlined_subroutine`` * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ``DW_TAG_subprogram``         * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ============================= =============================

.. _amdgpu-dwarf-debug-information:

DWARF Debug Information
=======================

.. warning::

   This section describes a **provisional proposal** for AMDGPU DWARF [DWARF]_
   that is not currently fully implemented and is subject to change.

AMDGPU generates DWARF [DWARF]_ debugging information ELF sections (see
:ref:`amdgpu-elf-code-object`) which contain information that maps the code
object executable code and data to the source language constructs. It can be
used by tools such as debuggers and profilers. It uses features defined in the
:ref:`amdgpu-dwarf-6-proposal-for-heterogeneous-debugging` that are made
available in DWARF Version 4 and DWARF Version 5 as an LLVM vendor extension.

This section defines the AMDGPU target architecture specific DWARF mappings.

.. _amdgpu-dwarf-register-identifier:

Register Identifier
-------------------

This section defines the AMDGPU target architecture register numbers used in
DWARF operation expressions (see DWARF Version 5 section 2.5 and
:ref:`amdgpu-dwarf-operation-expressions`) and Call Frame Information
instructions (see DWARF Version 5 section 6.4 and
:ref:`amdgpu-dwarf-call-frame-information`).

A single code object can contain code for kernels that have different wavefront
sizes. The vector registers and some scalar registers are based on the wavefront
size. AMDGPU defines distinct DWARF registers for each wavefront size. This
simplifies the consumer of the DWARF so that each register has a fixed size,
rather than being dynamic according to the wavefront size mode. Similarly,
distinct DWARF registers are defined for those registers that vary in size
according to the process address size. This allows a consumer to treat a
specific AMDGPU processor as a single architecture regardless of how it is
configured at run time. The compiler explicitly specifies the DWARF registers
that match the mode in which the code it is generating will be executed.

DWARF registers are encoded as numbers, which are mapped to architecture
registers. The mapping for AMDGPU is defined in
:ref:`amdgpu-dwarf-register-mapping-table`. All AMDGPU targets use the same
mapping.

.. table:: AMDGPU DWARF Register Mapping
   :name: amdgpu-dwarf-register-mapping-table

   ============== ================= ======== ==================================
   DWARF Register AMDGPU Register   Bit Size Description
   ============== ================= ======== ==================================
   0              PC_32             32       Program Counter (PC) when
                                             executing in a 32-bit process
                                             address space. Used in the CFI to
                                             describe the PC of the calling
                                             frame.
   1              EXEC_MASK_32      32       Execution Mask Register when
                                             executing in wavefront 32 mode.
   2-15           *Reserved*
   16             PC_64             64       Program Counter (PC) when
                                             executing in a 64-bit process
                                             address space. Used in the CFI to
                                             describe the PC of the calling
                                             frame.
   17             EXEC_MASK_64      64       Execution Mask Register when
                                             executing in wavefront 64 mode.
   18-31          *Reserved*
   32-95          SGPR0-SGPR63      32       Scalar General Purpose
                                             Registers.
   96-127         *Reserved*
   128-511        *Reserved*
   512-1023       *Reserved*
   1024-1087      *Reserved*
   1088-1129      SGPR64-SGPR105    32       Scalar General Purpose Registers
   1130-1535      *Reserved*
   1536-1791      VGPR0-VGPR255     32*32    Vector General Purpose Registers
                                             when executing in wavefront 32
                                             mode.
   1792-2047      *Reserved*
   2048-2303      AGPR0-AGPR255     32*32    Vector Accumulation Registers
                                             when executing in wavefront 32
                                             ode.
   2304-2559      *Reserved*
   2560-2815      VGPR0-VGPR255     64*32    Vector General Purpose Registers
                                             when executing in wavefront 64
                                             mode.
   2816-3071      *Reserved*
   3072-3327      AGPR0-AGPR255     64*32    Vector Accumulation Registers
                                             when executing in wavefront 64
                                             mode.
   3328-3583      *Reserved*
   ============== ================= ======== ==================================

The vector registers are represented as the full size for the wavefront. They
are organized as consecutive dwords (32-bits), one per lane, with the dword at
the least significant bit position corresponding to lane 0 and so forth. DWARF
location expressions involving the ``DW_OP_LLVM_offset`` and
``DW_OP_LLVM_push_lane`` operations are used to select the part of the vector
register corresponding to the lane that is executing the current thread of
execution in languages that are implemented using a SIMD or SIMT execution
model.

If the wavefront size is 32 lanes then the wavefront 32 mode register
definitions are used. If the wavefront size is 64 lanes then the wavefront 64
mode register definitions are used. Some AMDGPU targets support executing in
both wavefront 32 and wavefront 64 mode. The register definitions corresponding
to the wavefront mode of the generated code will be used.

If code is generated to execute in a 32-bit process address space, then the
32-bit process address space register definitions are used. If code is generated
to execute in a 64-bit process address space, then the 64-bit process address
space register definitions are used. The ``amdgcn`` target only supports the
64-bit process address space.

.. _amdgpu-dwarf-address-class-identifier:

Address Class Identifier
------------------------

The DWARF address class represents the source language memory space. See DWARF
Version 5 section 2.12 which is updated by the propoal in
:ref:`amdgpu-dwarf-segment_addresses`.

The DWARF address class mapping used for AMDGPU is defined in
:ref:`amdgpu-dwarf-address-class-mapping-table`.

.. table:: AMDGPU DWARF Address Class Mapping
   :name: amdgpu-dwarf-address-class-mapping-table

   ========================= ====== =================
   DWARF                            AMDGPU
   -------------------------------- -----------------
   Address Class Name        Value  Address Space
   ========================= ====== =================
   ``DW_ADDR_none``          0x0000 Generic (Flat)
   ``DW_ADDR_LLVM_global``   0x0001 Global
   ``DW_ADDR_LLVM_constant`` 0x0002 Global
   ``DW_ADDR_LLVM_group``    0x0003 Local (group/LDS)
   ``DW_ADDR_LLVM_private``  0x0004 Private (Scratch)
   ``DW_ADDR_AMDGPU_region`` 0x8000 Region (GDS)
   ========================= ====== =================

The DWARF address class values defined in the proposal at
:ref:`amdgpu-dwarf-segment_addresses` are used.

In addition, ``DW_ADDR_AMDGPU_region`` is encoded as a vendor extension. This is
available for use for the AMD extension for access to the hardware GDS memory
which is scratchpad memory allocated per device.

For AMDGPU if no ``DW_AT_address_class`` attribute is present, then the default
address class of ``DW_ADDR_none`` is used.

See :ref:`amdgpu-dwarf-address-space-identifier` for information on the AMDGPU
mapping of DWARF address classes to DWARF address spaces, including address size
and NULL value.

.. _amdgpu-dwarf-address-space-identifier:

Address Space Identifier
------------------------

DWARF address spaces correspond to target architecture specific linear
addressable memory areas. See DWARF Version 5 section 2.12 and
:ref:`amdgpu-dwarf-segment_addresses`.

The DWARF address space mapping used for AMDGPU is defined in
:ref:`amdgpu-dwarf-address-space-mapping-table`.

.. table:: AMDGPU DWARF Address Space Mapping
   :name: amdgpu-dwarf-address-space-mapping-table

   ======================================= ===== ======= ======== ================= =======================
   DWARF                                                          AMDGPU            Notes
   --------------------------------------- ----- ---------------- ----------------- -----------------------
   Address Space Name                      Value Address Bit Size Address Space
   --------------------------------------- ----- ------- -------- ----------------- -----------------------
   ..                                            64-bit  32-bit
                                                 process process
                                                 address address
                                                 space   space
   ======================================= ===== ======= ======== ================= =======================
   ``DW_ASPACE_none``                      0x00  8       4        Global            *default address space*
   ``DW_ASPACE_AMDGPU_generic``            0x01  8       4        Generic (Flat)
   ``DW_ASPACE_AMDGPU_region``             0x02  4       4        Region (GDS)
   ``DW_ASPACE_AMDGPU_local``              0x03  4       4        Local (group/LDS)
   *Reserved*                              0x04
   ``DW_ASPACE_AMDGPU_private_lane``       0x05  4       4        Private (Scratch) *focused lane*
   ``DW_ASPACE_AMDGPU_private_wave``       0x06  4       4        Private (Scratch) *unswizzled wavefront*
   *Reserved*                              0x07-
                                           0x1F
   ``DW_ASPACE_AMDGPU_private_lane<0-63>`` 0x20- 4       4        Private (Scratch) *specific lane*
                                           0x5F
   ======================================= ===== ======= ======== ================= =======================

See :ref:`amdgpu-address-spaces` for information on the AMDGPU address spaces
including address size and NULL value.

The ``DW_ASPACE_none`` address space is the default target architecture address
space used in DWARF operations that do not specify an address space. It
therefore has to map to the global address space so that the ``DW_OP_addr*`` and
related operations can refer to addresses in the program code.

The ``DW_ASPACE_AMDGPU_generic`` address space allows location expressions to
specify the flat address space. If the address corresponds to an address in the
local address space, then it corresponds to the wavefront that is executing the
focused thread of execution. If the address corresponds to an address in the
private address space, then it corresponds to the lane that is executing the
focused thread of execution for languages that are implemented using a SIMD or
SIMT execution model.

.. note::

  CUDA-like languages such as HIP that do not have address spaces in the
  language type system, but do allow variables to be allocated in different
  address spaces, need to explicitly specify the ``DW_ASPACE_AMDGPU_generic``
  address space in the DWARF expression operations as the default address space
  is the global address space.

The ``DW_ASPACE_AMDGPU_local`` address space allows location expressions to
specify the local address space corresponding to the wavefront that is executing
the focused thread of execution.

The ``DW_ASPACE_AMDGPU_private_lane`` address space allows location expressions
to specify the private address space corresponding to the lane that is executing
the focused thread of execution for languages that are implemented using a SIMD
or SIMT execution model.

The ``DW_ASPACE_AMDGPU_private_wave`` address space allows location expressions
to specify the unswizzled private address space corresponding to the wavefront
that is executing the focused thread of execution. The wavefront view of private
memory is the per wavefront unswizzled backing memory layout defined in
:ref:`amdgpu-address-spaces`, such that address 0 corresponds to the first
location for the backing memory of the wavefront (namely the address is not
offset by ``wavefront-scratch-base``). The following formula can be used to
convert from a ``DW_ASPACE_AMDGPU_private_lane`` address to a
``DW_ASPACE_AMDGPU_private_wave`` address:

::

  private-address-wavefront =
    ((private-address-lane / 4) * wavefront-size * 4) +
    (wavefront-lane-id * 4) + (private-address-lane % 4)

If the ``DW_ASPACE_AMDGPU_private_lane`` address is dword aligned, and the start
of the dwords for each lane starting with lane 0 is required, then this
simplifies to:

::

  private-address-wavefront =
    private-address-lane * wavefront-size

A compiler can use the ``DW_ASPACE_AMDGPU_private_wave`` address space to read a
complete spilled vector register back into a complete vector register in the
CFI. The frame pointer can be a private lane address which is dword aligned,
which can be shifted to multiply by the wavefront size, and then used to form a
private wavefront address that gives a location for a contiguous set of dwords,
one per lane, where the vector register dwords are spilled. The compiler knows
the wavefront size since it generates the code. Note that the type of the
address may have to be converted as the size of a
``DW_ASPACE_AMDGPU_private_lane`` address may be smaller than the size of a
``DW_ASPACE_AMDGPU_private_wave`` address.

The ``DW_ASPACE_AMDGPU_private_lane<N>`` address space allows location
expressions to specify the private address space corresponding to a specific
lane N. For example, this can be used when the compiler spills scalar registers
to scratch memory, with each scalar register being saved to a different lane's
scratch memory.

.. _amdgpu-dwarf-lane-identifier:

Lane identifier
---------------

DWARF lane identifies specify a target architecture lane position for hardware
that executes in a SIMD or SIMT manner, and on which a source language maps its
threads of execution onto those lanes. The DWARF lane identifier is pushed by
the ``DW_OP_LLVM_push_lane`` DWARF expression operation. See DWARF Version 5
section 2.5 which is updated by the proposal in
:ref:`amdgpu-dwarf-operation-expressions`.

For AMDGPU, the lane identifier corresponds to the hardware lane ID of a
wavefront. It is numbered from 0 to the wavefront size minus 1.

Operation Expressions
---------------------

DWARF expressions are used to compute program values and the locations of
program objects. See DWARF Version 5 section 2.5 and
:ref:`amdgpu-dwarf-operation-expressions`.

DWARF location descriptions describe how to access storage which includes memory
and registers. When accessing storage on AMDGPU, bytes are ordered with least
significant bytes first, and bits are ordered within bytes with least
significant bits first.

For AMDGPU CFI expressions, ``DW_OP_LLVM_select_bit_piece`` is used to describe
unwinding vector registers that are spilled under the execution mask to memory:
the zero-single location description is the vector register, and the one-single
location description is the spilled memory location description. The
``DW_OP_LLVM_form_aspace_address`` is used to specify the address space of the
memory location description.

In AMDGPU expressions, ``DW_OP_LLVM_select_bit_piece`` is used by the
``DW_AT_LLVM_lane_pc`` attribute expression where divergent control flow is
controlled by the execution mask. An undefined location description together
with ``DW_OP_LLVM_extend`` is used to indicate the lane was not active on entry
to the subprogram. See :ref:`amdgpu-dwarf-dw-at-llvm-lane-pc` for an example.

Debugger Information Entry Attributes
-------------------------------------

This section describes how certain debugger information entry attributes are
used by AMDGPU. See the sections in DWARF Version 5 section 2 which are updated
by the proposal in :ref:`amdgpu-dwarf-debugging-information-entry-attributes`.

.. _amdgpu-dwarf-dw-at-llvm-lane-pc:

``DW_AT_LLVM_lane_pc``
~~~~~~~~~~~~~~~~~~~~~~

For AMDGPU, the ``DW_AT_LLVM_lane_pc`` attribute is used to specify the program
location of the separate lanes of a SIMT thread.

If the lane is an active lane then this will be the same as the current program
location.

If the lane is inactive, but was active on entry to the subprogram, then this is
the program location in the subprogram at which execution of the lane is
conceptual positioned.

If the lane was not active on entry to the subprogram, then this will be the
undefined location. A client debugger can check if the lane is part of a valid
work-group by checking that the lane is in the range of the associated
work-group within the grid, accounting for partial work-groups. If it is not,
then the debugger can omit any information for the lane. Otherwise, the debugger
may repeatedly unwind the stack and inspect the ``DW_AT_LLVM_lane_pc`` of the
calling subprogram until it finds a non-undefined location. Conceptually the
lane only has the call frames that it has a non-undefined
``DW_AT_LLVM_lane_pc``.

The following example illustrates how the AMDGPU backend can generate a DWARF
location list expression for the nested ``IF/THEN/ELSE`` structures of the
following subprogram pseudo code for a target with 64 lanes per wavefront.

.. code::
  :number-lines:

  SUBPROGRAM X
  BEGIN
    a;
    IF (c1) THEN
      b;
      IF (c2) THEN
        c;
      ELSE
        d;
      ENDIF
      e;
    ELSE
      f;
    ENDIF
    g;
  END

The AMDGPU backend may generate the following pseudo LLVM MIR to manipulate the
execution mask (``EXEC``) to linearize the control flow. The condition is
evaluated to make a mask of the lanes for which the condition evaluates to true.
First the ``THEN`` region is executed by setting the ``EXEC`` mask to the
logical ``AND`` of the current ``EXEC`` mask with the condition mask. Then the
``ELSE`` region is executed by negating the ``EXEC`` mask and logical ``AND`` of
the saved ``EXEC`` mask at the start of the region. After the ``IF/THEN/ELSE``
region the ``EXEC`` mask is restored to the value it had at the beginning of the
region. This is shown below. Other approaches are possible, but the basic
concept is the same.

.. code::
  :number-lines:

  $lex_start:
    a;
    %1 = EXEC
    %2 = c1
  $lex_1_start:
    EXEC = %1 & %2
  $if_1_then:
      b;
      %3 = EXEC
      %4 = c2
  $lex_1_1_start:
      EXEC = %3 & %4
  $lex_1_1_then:
        c;
      EXEC = ~EXEC & %3
  $lex_1_1_else:
        d;
      EXEC = %3
  $lex_1_1_end:
      e;
    EXEC = ~EXEC & %1
  $lex_1_else:
      f;
    EXEC = %1
  $lex_1_end:
    g;
  $lex_end:

To create the DWARF location list expression that defines the location
description of a vector of lane program locations, the LLVM MIR ``DBG_VALUE``
pseudo instruction can be used to annotate the linearized control flow. This can
be done by defining an artificial variable for the lane PC. The DWARF location
list expression created for it is used as the value of the
``DW_AT_LLVM_lane_pc`` attribute on the subprogram's debugger information entry.

A DWARF procedure is defined for each well nested structured control flow region
which provides the conceptual lane program location for a lane if it is not
active (namely it is divergent). The DWARF operation expression for each region
conceptually inherits the value of the immediately enclosing region and modifies
it according to the semantics of the region.

For an ``IF/THEN/ELSE`` region the divergent program location is at the start of
the region for the ``THEN`` region since it is executed first. For the ``ELSE``
region the divergent program location is at the end of the ``IF/THEN/ELSE``
region since the ``THEN`` region has completed.

The lane PC artificial variable is assigned at each region transition. It uses
the immediately enclosing region's DWARF procedure to compute the program
location for each lane assuming they are divergent, and then modifies the result
by inserting the current program location for each lane that the ``EXEC`` mask
indicates is active.

By having separate DWARF procedures for each region, they can be reused to
define the value for any nested region. This reduces the total size of the DWARF
operation expressions.

The following provides an example using pseudo LLVM MIR.

.. code::
  :number-lines:

  $lex_start:
    DEFINE_DWARF %__uint_64 = DW_TAG_base_type[
      DW_AT_name = "__uint64";
      DW_AT_byte_size = 8;
      DW_AT_encoding = DW_ATE_unsigned;
    ];
    DEFINE_DWARF %__active_lane_pc = DW_TAG_dwarf_procedure[
      DW_AT_name = "__active_lane_pc";
      DW_AT_location = [
        DW_OP_regx PC;
        DW_OP_LLVM_extend 64, 64;
        DW_OP_regval_type EXEC, %uint_64;
        DW_OP_LLVM_select_bit_piece 64, 64;
      ];
    ];
    DEFINE_DWARF %__divergent_lane_pc = DW_TAG_dwarf_procedure[
      DW_AT_name = "__divergent_lane_pc";
      DW_AT_location = [
        DW_OP_LLVM_undefined;
        DW_OP_LLVM_extend 64, 64;
      ];
    ];
    DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
      DW_OP_call_ref %__divergent_lane_pc;
      DW_OP_call_ref %__active_lane_pc;
    ];
    a;
    %1 = EXEC;
    DBG_VALUE %1, $noreg, %__lex_1_save_exec;
    %2 = c1;
  $lex_1_start:
    EXEC = %1 & %2;
  $lex_1_then:
      DEFINE_DWARF %__divergent_lane_pc_1_then = DW_TAG_dwarf_procedure[
        DW_AT_name = "__divergent_lane_pc_1_then";
        DW_AT_location = DIExpression[
          DW_OP_call_ref %__divergent_lane_pc;
          DW_OP_xaddr &lex_1_start;
          DW_OP_stack_value;
          DW_OP_LLVM_extend 64, 64;
          DW_OP_call_ref %__lex_1_save_exec;
          DW_OP_deref_type 64, %__uint_64;
          DW_OP_LLVM_select_bit_piece 64, 64;
        ];
      ];
      DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
        DW_OP_call_ref %__divergent_lane_pc_1_then;
        DW_OP_call_ref %__active_lane_pc;
      ];
      b;
      %3 = EXEC;
      DBG_VALUE %3, %__lex_1_1_save_exec;
      %4 = c2;
  $lex_1_1_start:
      EXEC = %3 & %4;
  $lex_1_1_then:
        DEFINE_DWARF %__divergent_lane_pc_1_1_then = DW_TAG_dwarf_procedure[
          DW_AT_name = "__divergent_lane_pc_1_1_then";
          DW_AT_location = DIExpression[
            DW_OP_call_ref %__divergent_lane_pc_1_then;
            DW_OP_xaddr &lex_1_1_start;
            DW_OP_stack_value;
            DW_OP_LLVM_extend 64, 64;
            DW_OP_call_ref %__lex_1_1_save_exec;
            DW_OP_deref_type 64, %__uint_64;
            DW_OP_LLVM_select_bit_piece 64, 64;
          ];
        ];
        DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
          DW_OP_call_ref %__divergent_lane_pc_1_1_then;
          DW_OP_call_ref %__active_lane_pc;
        ];
        c;
      EXEC = ~EXEC & %3;
  $lex_1_1_else:
        DEFINE_DWARF %__divergent_lane_pc_1_1_else = DW_TAG_dwarf_procedure[
          DW_AT_name = "__divergent_lane_pc_1_1_else";
          DW_AT_location = DIExpression[
            DW_OP_call_ref %__divergent_lane_pc_1_then;
            DW_OP_xaddr &lex_1_1_end;
            DW_OP_stack_value;
            DW_OP_LLVM_extend 64, 64;
            DW_OP_call_ref %__lex_1_1_save_exec;
            DW_OP_deref_type 64, %__uint_64;
            DW_OP_LLVM_select_bit_piece 64, 64;
          ];
        ];
        DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
          DW_OP_call_ref %__divergent_lane_pc_1_1_else;
          DW_OP_call_ref %__active_lane_pc;
        ];
        d;
      EXEC = %3;
  $lex_1_1_end:
      DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
        DW_OP_call_ref %__divergent_lane_pc;
        DW_OP_call_ref %__active_lane_pc;
      ];
      e;
    EXEC = ~EXEC & %1;
  $lex_1_else:
      DEFINE_DWARF %__divergent_lane_pc_1_else = DW_TAG_dwarf_procedure[
        DW_AT_name = "__divergent_lane_pc_1_else";
        DW_AT_location = DIExpression[
          DW_OP_call_ref %__divergent_lane_pc;
          DW_OP_xaddr &lex_1_end;
          DW_OP_stack_value;
          DW_OP_LLVM_extend 64, 64;
          DW_OP_call_ref %__lex_1_save_exec;
          DW_OP_deref_type 64, %__uint_64;
          DW_OP_LLVM_select_bit_piece 64, 64;
        ];
      ];
      DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc, DIExpression[
        DW_OP_call_ref %__divergent_lane_pc_1_else;
        DW_OP_call_ref %__active_lane_pc;
      ];
      f;
    EXEC = %1;
  $lex_1_end:
    DBG_VALUE $noreg, $noreg, %DW_AT_LLVM_lane_pc DIExpression[
      DW_OP_call_ref %__divergent_lane_pc;
      DW_OP_call_ref %__active_lane_pc;
    ];
    g;
  $lex_end:

The DWARF procedure ``%__active_lane_pc`` is used to update the lane pc elements
that are active, with the current program location.

Artificial variables %__lex_1_save_exec and %__lex_1_1_save_exec are created for
the execution masks saved on entry to a region. Using the ``DBG_VALUE`` pseudo
instruction, location list entries will be created that describe where the
artificial variables are allocated at any given program location. The compiler
may allocate them to registers or spill them to memory.

The DWARF procedures for each region use the values of the saved execution mask
artificial variables to only update the lanes that are active on entry to the
region. All other lanes retain the value of the enclosing region where they were
last active. If they were not active on entry to the subprogram, then will have
the undefined location description.

Other structured control flow regions can be handled similarly. For example,
loops would set the divergent program location for the region at the end of the
loop. Any lanes active will be in the loop, and any lanes not active must have
exited the loop.

An ``IF/THEN/ELSEIF/ELSEIF/...`` region can be treated as a nest of
``IF/THEN/ELSE`` regions.

The DWARF procedures can use the active lane artificial variable described in
:ref:`amdgpu-dwarf-amdgpu-dw-at-llvm-active-lane` rather than the actual
``EXEC`` mask in order to support whole or quad wavefront mode.

.. _amdgpu-dwarf-amdgpu-dw-at-llvm-active-lane:

``DW_AT_LLVM_active_lane``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DW_AT_LLVM_active_lane`` attribute on a subprogram debugger information
entry is used to specify the lanes that are conceptually active for a SIMT
thread.

The execution mask may be modified to implement whole or quad wavefront mode
operations. For example, all lanes may need to temporarily be made active to
execute a whole wavefront operation. Such regions would save the ``EXEC`` mask,
update it to enable the necessary lanes, perform the operations, and then
restore the ``EXEC`` mask from the saved value. While executing the whole
wavefront region, the conceptual execution mask is the saved value, not the
``EXEC`` value.

This is handled by defining an artificial variable for the active lane mask. The
active lane mask artificial variable would be the actual ``EXEC`` mask for
normal regions, and the saved execution mask for regions where the mask is
temporarily updated. The location list expression created for this artificial
variable is used to define the value of the ``DW_AT_LLVM_active_lane``
attribute.

``DW_AT_LLVM_augmentation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For AMDGPU, the ``DW_AT_LLVM_augmentation`` attribute of a compilation unit
debugger information entry has the following value for the augmentation string:

::

  [amdgpu:v0.0]

The "vX.Y" specifies the major X and minor Y version number of the AMDGPU
extensions used in the DWARF of the compilation unit. The version number
conforms to [SEMVER]_.

Call Frame Information
----------------------

DWARF Call Frame Information (CFI) describes how a consumer can virtually
*unwind* call frames in a running process or core dump. See DWARF Version 5
section 6.4 and :ref:`amdgpu-dwarf-call-frame-information`.

For AMDGPU, the Common Information Entry (CIE) fields have the following values:

1.  ``augmentation`` string contains the following null-terminated UTF-8 string:

    ::

      [amd:v0.0]

    The ``vX.Y`` specifies the major X and minor Y version number of the AMDGPU
    extensions used in this CIE or to the FDEs that use it. The version number
    conforms to [SEMVER]_.

2.  ``address_size`` for the ``Global`` address space is defined in
    :ref:`amdgpu-dwarf-address-space-identifier`.

3.  ``segment_selector_size`` is 0 as AMDGPU does not use a segment selector.

4.  ``code_alignment_factor`` is 4 bytes.

    .. TODO::

       Add to :ref:`amdgpu-processor-table` table.

5.  ``data_alignment_factor`` is 4 bytes.

    .. TODO::

       Add to :ref:`amdgpu-processor-table` table.

6.  ``return_address_register`` is ``PC_32`` for 32-bit processes and ``PC_64``
    for 64-bit processes defined in :ref:`amdgpu-dwarf-register-identifier`.

7.  ``initial_instructions`` Since a subprogram X with fewer registers can be
    called from subprogram Y that has more allocated, X will not change any of
    the extra registers as it cannot access them. Therefore, the default rule
    for all columns is ``same value``.

For AMDGPU the register number follows the numbering defined in
:ref:`amdgpu-dwarf-register-identifier`.

For AMDGPU the instructions are variable size. A consumer can subtract 1 from
the return address to get the address of a byte within the call site
instructions. See DWARF Version 5 section 6.4.4.

Accelerated Access
------------------

See DWARF Version 5 section 6.1.

Lookup By Name Section Header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See DWARF Version 5 section 6.1.1.4.1 and :ref:`amdgpu-dwarf-lookup-by-name`.

For AMDGPU the lookup by name section header table:

``augmentation_string_size`` (uword)

  Set to the length of the ``augmentation_string`` value which is always a
  multiple of 4.

``augmentation_string`` (sequence of UTF-8 characters)

  Contains the following UTF-8 string null padded to a multiple of 4 bytes:

  ::

    [amdgpu:v0.0]

  The "vX.Y" specifies the major X and minor Y version number of the AMDGPU
  extensions used in the DWARF of this index. The version number conforms to
  [SEMVER]_.

  .. note::

    This is different to the DWARF Version 5 definition that requires the first
    4 characters to be the vendor ID. But this is consistent with the other
    augmentation strings and does allow multiple vendor contributions. However,
    backwards compatibility may be more desirable.

Lookup By Address Section Header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See DWARF Version 5 section 6.1.2.

For AMDGPU the lookup by address section header table:

``address_size`` (ubyte)

  Match the address size for the ``Global`` address space defined in
  :ref:`amdgpu-dwarf-address-space-identifier`.

``segment_selector_size`` (ubyte)

  AMDGPU does not use a segment selector so this is 0. The entries in the
  ``.debug_aranges`` do not have a segment selector.

Line Number Information
-----------------------

See DWARF Version 5 section 6.2 and :ref:`amdgpu-dwarf-line-number-information`.

AMDGPU does not use the ``isa`` state machine registers and always sets it to 0.
The instruction set must be obtained from the ELF file header ``e_flags`` field
in the ``EF_AMDGPU_MACH`` bit position (see :ref:`ELF Header
<amdgpu-elf-header>`). See DWARF Version 5 section 6.2.2.

.. TODO::

  Should the ``isa`` state machine register be used to indicate if the code is
  in wavefront32 or wavefront64 mode? Or used to specify the architecture ISA?

For AMDGPU the line number program header fields have the following values (see
DWARF Version 5 section 6.2.4):

``address_size`` (ubyte)
  Matches the address size for the ``Global`` address space defined in
  :ref:`amdgpu-dwarf-address-space-identifier`.

``segment_selector_size`` (ubyte)
  AMDGPU does not use a segment selector so this is 0.

``minimum_instruction_length`` (ubyte)
  For GFX9-GFX10 this is 4.

``maximum_operations_per_instruction`` (ubyte)
  For GFX9-GFX10 this is 1.

Source text for online-compiled programs (for example, those compiled by the
OpenCL language runtime) may be embedded into the DWARF Version 5 line table.
See DWARF Version 5 section 6.2.4.1 which is updated by the proposal in
:ref:`DW_LNCT_LLVM_source
<amdgpu-dwarf-line-number-information-dw-lnct-llvm-source>`.

The Clang option used to control source embedding in AMDGPU is defined in
:ref:`amdgpu-clang-debug-options-table`.

  .. table:: AMDGPU Clang Debug Options
     :name: amdgpu-clang-debug-options-table

     ==================== ==================================================
     Debug Flag           Description
     ==================== ==================================================
     -g[no-]embed-source  Enable/disable embedding source text in DWARF
                          debug sections. Useful for environments where
                          source cannot be written to disk, such as
                          when performing online compilation.
     ==================== ==================================================

For example:

``-gembed-source``
  Enable the embedded source.

``-gno-embed-source``
  Disable the embedded source.

32-Bit and 64-Bit DWARF Formats
-------------------------------

See DWARF Version 5 section 7.4 and
:ref:`amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats`.

For AMDGPU:

* For the ``amdgcn`` target architecture only the 64-bit process address space
  is supported.

* The producer can generate either 32-bit or 64-bit DWARF format. LLVM generates
  the 32-bit DWARF format.

Unit Headers
------------

For AMDGPU the following values apply for each of the unit headers described in
DWARF Version 5 sections 7.5.1.1, 7.5.1.2, and 7.5.1.3:

``address_size`` (ubyte)
  Matches the address size for the ``Global`` address space defined in
  :ref:`amdgpu-dwarf-address-space-identifier`.

.. _amdgpu-code-conventions:

Code Conventions
================

This section provides code conventions used for each supported target triple OS
(see :ref:`amdgpu-target-triples`).

AMDHSA
------

This section provides code conventions used when the target triple OS is
``amdhsa`` (see :ref:`amdgpu-target-triples`).

.. _amdgpu-amdhsa-code-object-target-identification:

Code Object Target Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMDHSA OS uses the following syntax to specify the code object
target as a single string:

  ``<Architecture>-<Vendor>-<OS>-<Environment>-<Processor><Target Features>``

Where:

  - ``<Architecture>``, ``<Vendor>``, ``<OS>`` and ``<Environment>``
    are the same as the *Target Triple* (see
    :ref:`amdgpu-target-triples`).

  - ``<Processor>`` is the same as the *Processor* (see
    :ref:`amdgpu-processors`).

  - ``<Target Features>`` is a list of the enabled *Target Features*
    (see :ref:`amdgpu-target-features`), each prefixed by a plus, that
    apply to *Processor*. The list must be in the same order as listed
    in the table :ref:`amdgpu-target-feature-table`. Note that *Target
    Features* must be included in the list if they are enabled even if
    that is the default for *Processor*.

For example:

  ``"amdgcn-amd-amdhsa--gfx902+xnack"``

.. _amdgpu-amdhsa-code-object-metadata:

Code Object Metadata
~~~~~~~~~~~~~~~~~~~~

The code object metadata specifies extensible metadata associated with the code
objects executed on HSA [HSA]_ compatible runtimes such as AMD's ROCm
[AMD-ROCm]_. The encoding and semantics of this metadata depends on the code
object version; see :ref:`amdgpu-amdhsa-code-object-metadata-v2` and
:ref:`amdgpu-amdhsa-code-object-metadata-v3`.

Code object metadata is specified in a note record (see
:ref:`amdgpu-note-records`) and is required when the target triple OS is
``amdhsa`` (see :ref:`amdgpu-target-triples`). It must contain the minimum
information necessary to support the ROCM kernel queries. For example, the
segment sizes needed in a dispatch packet. In addition, a high-level language
runtime may require other information to be included. For example, the AMD
OpenCL runtime records kernel argument information.

.. _amdgpu-amdhsa-code-object-metadata-v2:

Code Object V2 Metadata (-mattr=-code-object-v3)
++++++++++++++++++++++++++++++++++++++++++++++++

.. warning:: Code Object V2 is not the default code object version emitted by
  this version of LLVM. For a description of the metadata generated with the
  default configuration (Code Object V3) see
  :ref:`amdgpu-amdhsa-code-object-metadata-v3`.

Code object V2 metadata is specified by the ``NT_AMD_AMDGPU_METADATA`` note
record (see :ref:`amdgpu-note-records-v2`).

The metadata is specified as a YAML formatted string (see [YAML]_ and
:doc:`YamlIO`).

.. TODO::

  Is the string null terminated? It probably should not if YAML allows it to
  contain null characters, otherwise it should be.

The metadata is represented as a single YAML document comprised of the mapping
defined in table :ref:`amdgpu-amdhsa-code-object-metadata-map-table-v2` and
referenced tables.

For boolean values, the string values of ``false`` and ``true`` are used for
false and true respectively.

Additional information can be added to the mappings. To avoid conflicts, any
non-AMD key names should be prefixed by "*vendor-name*.".

  .. table:: AMDHSA Code Object V2 Metadata Map
     :name: amdgpu-amdhsa-code-object-metadata-map-table-v2

     ========== ============== ========= =======================================
     String Key Value Type     Required? Description
     ========== ============== ========= =======================================
     "Version"  sequence of    Required  - The first integer is the major
                2 integers                 version. Currently 1.
                                         - The second integer is the minor
                                           version. Currently 0.
     "Printf"   sequence of              Each string is encoded information
                strings                  about a printf function call. The
                                         encoded information is organized as
                                         fields separated by colon (':'):

                                         ``ID:N:S[0]:S[1]:...:S[N-1]:FormatString``

                                         where:

                                         ``ID``
                                           A 32-bit integer as a unique id for
                                           each printf function call

                                         ``N``
                                           A 32-bit integer equal to the number
                                           of arguments of printf function call
                                           minus 1

                                         ``S[i]`` (where i = 0, 1, ... , N-1)
                                           32-bit integers for the size in bytes
                                           of the i-th FormatString argument of
                                           the printf function call

                                         FormatString
                                           The format string passed to the
                                           printf function call.
     "Kernels"  sequence of    Required  Sequence of the mappings for each
                mapping                  kernel in the code object. See
                                         :ref:`amdgpu-amdhsa-code-object-kernel-metadata-map-table-v2`
                                         for the definition of the mapping.
     ========== ============== ========= =======================================

..

  .. table:: AMDHSA Code Object V2 Kernel Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-metadata-map-table-v2

     ================= ============== ========= ================================
     String Key        Value Type     Required? Description
     ================= ============== ========= ================================
     "Name"            string         Required  Source name of the kernel.
     "SymbolName"      string         Required  Name of the kernel
                                                descriptor ELF symbol.
     "Language"        string                   Source language of the kernel.
                                                Values include:

                                                - "OpenCL C"
                                                - "OpenCL C++"
                                                - "HCC"
                                                - "OpenMP"

     "LanguageVersion" sequence of              - The first integer is the major
                       2 integers                 version.
                                                - The second integer is the
                                                  minor version.
     "Attrs"           mapping                  Mapping of kernel attributes.
                                                See
                                                :ref:`amdgpu-amdhsa-code-object-kernel-attribute-metadata-map-table-v2`
                                                for the mapping definition.
     "Args"            sequence of              Sequence of mappings of the
                       mapping                  kernel arguments. See
                                                :ref:`amdgpu-amdhsa-code-object-kernel-argument-metadata-map-table-v2`
                                                for the definition of the mapping.
     "CodeProps"       mapping                  Mapping of properties related to
                                                the kernel code. See
                                                :ref:`amdgpu-amdhsa-code-object-kernel-code-properties-metadata-map-table-v2`
                                                for the mapping definition.
     ================= ============== ========= ================================

..

  .. table:: AMDHSA Code Object V2 Kernel Attribute Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-attribute-metadata-map-table-v2

     =================== ============== ========= ==============================
     String Key          Value Type     Required? Description
     =================== ============== ========= ==============================
     "ReqdWorkGroupSize" sequence of              If not 0, 0, 0 then all values
                         3 integers               must be >=1 and the dispatch
                                                  work-group size X, Y, Z must
                                                  correspond to the specified
                                                  values. Defaults to 0, 0, 0.

                                                  Corresponds to the OpenCL
                                                  ``reqd_work_group_size``
                                                  attribute.
     "WorkGroupSizeHint" sequence of              The dispatch work-group size
                         3 integers               X, Y, Z is likely to be the
                                                  specified values.

                                                  Corresponds to the OpenCL
                                                  ``work_group_size_hint``
                                                  attribute.
     "VecTypeHint"       string                   The name of a scalar or vector
                                                  type.

                                                  Corresponds to the OpenCL
                                                  ``vec_type_hint`` attribute.

     "RuntimeHandle"     string                   The external symbol name
                                                  associated with a kernel.
                                                  OpenCL runtime allocates a
                                                  global buffer for the symbol
                                                  and saves the kernel's address
                                                  to it, which is used for
                                                  device side enqueueing. Only
                                                  available for device side
                                                  enqueued kernels.
     =================== ============== ========= ==============================

..

  .. table:: AMDHSA Code Object V2 Kernel Argument Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-argument-metadata-map-table-v2

     ================= ============== ========= ================================
     String Key        Value Type     Required? Description
     ================= ============== ========= ================================
     "Name"            string                   Kernel argument name.
     "TypeName"        string                   Kernel argument type name.
     "Size"            integer        Required  Kernel argument size in bytes.
     "Align"           integer        Required  Kernel argument alignment in
                                                bytes. Must be a power of two.
     "ValueKind"       string         Required  Kernel argument kind that
                                                specifies how to set up the
                                                corresponding argument.
                                                Values include:

                                                "ByValue"
                                                  The argument is copied
                                                  directly into the kernarg.

                                                "GlobalBuffer"
                                                  A global address space pointer
                                                  to the buffer data is passed
                                                  in the kernarg.

                                                "DynamicSharedPointer"
                                                  A group address space pointer
                                                  to dynamically allocated LDS
                                                  is passed in the kernarg.

                                                "Sampler"
                                                  A global address space
                                                  pointer to a S# is passed in
                                                  the kernarg.

                                                "Image"
                                                  A global address space
                                                  pointer to a T# is passed in
                                                  the kernarg.

                                                "Pipe"
                                                  A global address space pointer
                                                  to an OpenCL pipe is passed in
                                                  the kernarg.

                                                "Queue"
                                                  A global address space pointer
                                                  to an OpenCL device enqueue
                                                  queue is passed in the
                                                  kernarg.

                                                "HiddenGlobalOffsetX"
                                                  The OpenCL grid dispatch
                                                  global offset for the X
                                                  dimension is passed in the
                                                  kernarg.

                                                "HiddenGlobalOffsetY"
                                                  The OpenCL grid dispatch
                                                  global offset for the Y
                                                  dimension is passed in the
                                                  kernarg.

                                                "HiddenGlobalOffsetZ"
                                                  The OpenCL grid dispatch
                                                  global offset for the Z
                                                  dimension is passed in the
                                                  kernarg.

                                                "HiddenNone"
                                                  An argument that is not used
                                                  by the kernel. Space needs to
                                                  be left for it, but it does
                                                  not need to be set up.

                                                "HiddenPrintfBuffer"
                                                  A global address space pointer
                                                  to the runtime printf buffer
                                                  is passed in kernarg.

                                                "HiddenHostcallBuffer"
                                                  A global address space pointer
                                                  to the runtime hostcall buffer
                                                  is passed in kernarg.

                                                "HiddenDefaultQueue"
                                                  A global address space pointer
                                                  to the OpenCL device enqueue
                                                  queue that should be used by
                                                  the kernel by default is
                                                  passed in the kernarg.

                                                "HiddenCompletionAction"
                                                  A global address space pointer
                                                  to help link enqueued kernels into
                                                  the ancestor tree for determining
                                                  when the parent kernel has finished.

                                                "HiddenMultiGridSyncArg"
                                                  A global address space pointer for
                                                  multi-grid synchronization is
                                                  passed in the kernarg.

     "ValueType"       string         Required  Kernel argument value type. Only
                                                present if "ValueKind" is
                                                "ByValue". For vector data
                                                types, the value is for the
                                                element type. Values include:

                                                - "Struct"
                                                - "I8"
                                                - "U8"
                                                - "I16"
                                                - "U16"
                                                - "F16"
                                                - "I32"
                                                - "U32"
                                                - "F32"
                                                - "I64"
                                                - "U64"
                                                - "F64"

                                                .. TODO::
                                                   How can it be determined if a
                                                   vector type, and what size
                                                   vector?
     "PointeeAlign"    integer                  Alignment in bytes of pointee
                                                type for pointer type kernel
                                                argument. Must be a power
                                                of 2. Only present if
                                                "ValueKind" is
                                                "DynamicSharedPointer".
     "AddrSpaceQual"   string                   Kernel argument address space
                                                qualifier. Only present if
                                                "ValueKind" is "GlobalBuffer" or
                                                "DynamicSharedPointer". Values
                                                are:

                                                - "Private"
                                                - "Global"
                                                - "Constant"
                                                - "Local"
                                                - "Generic"
                                                - "Region"

                                                .. TODO::
                                                   Is GlobalBuffer only Global
                                                   or Constant? Is
                                                   DynamicSharedPointer always
                                                   Local? Can HCC allow Generic?
                                                   How can Private or Region
                                                   ever happen?
     "AccQual"         string                   Kernel argument access
                                                qualifier. Only present if
                                                "ValueKind" is "Image" or
                                                "Pipe". Values
                                                are:

                                                - "ReadOnly"
                                                - "WriteOnly"
                                                - "ReadWrite"

                                                .. TODO::
                                                   Does this apply to
                                                   GlobalBuffer?
     "ActualAccQual"   string                   The actual memory accesses
                                                performed by the kernel on the
                                                kernel argument. Only present if
                                                "ValueKind" is "GlobalBuffer",
                                                "Image", or "Pipe". This may be
                                                more restrictive than indicated
                                                by "AccQual" to reflect what the
                                                kernel actual does. If not
                                                present then the runtime must
                                                assume what is implied by
                                                "AccQual" and "IsConst". Values
                                                are:

                                                - "ReadOnly"
                                                - "WriteOnly"
                                                - "ReadWrite"

     "IsConst"         boolean                  Indicates if the kernel argument
                                                is const qualified. Only present
                                                if "ValueKind" is
                                                "GlobalBuffer".

     "IsRestrict"      boolean                  Indicates if the kernel argument
                                                is restrict qualified. Only
                                                present if "ValueKind" is
                                                "GlobalBuffer".

     "IsVolatile"      boolean                  Indicates if the kernel argument
                                                is volatile qualified. Only
                                                present if "ValueKind" is
                                                "GlobalBuffer".

     "IsPipe"          boolean                  Indicates if the kernel argument
                                                is pipe qualified. Only present
                                                if "ValueKind" is "Pipe".

                                                .. TODO::
                                                   Can GlobalBuffer be pipe
                                                   qualified?
     ================= ============== ========= ================================

..

  .. table:: AMDHSA Code Object V2 Kernel Code Properties Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-code-properties-metadata-map-table-v2

     ============================ ============== ========= =====================
     String Key                   Value Type     Required? Description
     ============================ ============== ========= =====================
     "KernargSegmentSize"         integer        Required  The size in bytes of
                                                           the kernarg segment
                                                           that holds the values
                                                           of the arguments to
                                                           the kernel.
     "GroupSegmentFixedSize"      integer        Required  The amount of group
                                                           segment memory
                                                           required by a
                                                           work-group in
                                                           bytes. This does not
                                                           include any
                                                           dynamically allocated
                                                           group segment memory
                                                           that may be added
                                                           when the kernel is
                                                           dispatched.
     "PrivateSegmentFixedSize"    integer        Required  The amount of fixed
                                                           private address space
                                                           memory required for a
                                                           work-item in
                                                           bytes. If the kernel
                                                           uses a dynamic call
                                                           stack then additional
                                                           space must be added
                                                           to this value for the
                                                           call stack.
     "KernargSegmentAlign"        integer        Required  The maximum byte
                                                           alignment of
                                                           arguments in the
                                                           kernarg segment. Must
                                                           be a power of 2.
     "WavefrontSize"              integer        Required  Wavefront size. Must
                                                           be a power of 2.
     "NumSGPRs"                   integer        Required  Number of scalar
                                                           registers used by a
                                                           wavefront for
                                                           GFX6-GFX10. This
                                                           includes the special
                                                           SGPRs for VCC, Flat
                                                           Scratch (GFX7-GFX10)
                                                           and XNACK (for
                                                           GFX8-GFX10). It does
                                                           not include the 16
                                                           SGPR added if a trap
                                                           handler is
                                                           enabled. It is not
                                                           rounded up to the
                                                           allocation
                                                           granularity.
     "NumVGPRs"                   integer        Required  Number of vector
                                                           registers used by
                                                           each work-item for
                                                           GFX6-GFX10
     "MaxFlatWorkGroupSize"       integer        Required  Maximum flat
                                                           work-group size
                                                           supported by the
                                                           kernel in work-items.
                                                           Must be >=1 and
                                                           consistent with
                                                           ReqdWorkGroupSize if
                                                           not 0, 0, 0.
     "NumSpilledSGPRs"            integer                  Number of stores from
                                                           a scalar register to
                                                           a register allocator
                                                           created spill
                                                           location.
     "NumSpilledVGPRs"            integer                  Number of stores from
                                                           a vector register to
                                                           a register allocator
                                                           created spill
                                                           location.
     ============================ ============== ========= =====================

.. _amdgpu-amdhsa-code-object-metadata-v3:

Code Object V3 Metadata (-mattr=+code-object-v3)
++++++++++++++++++++++++++++++++++++++++++++++++

Code object V3 metadata is specified by the ``NT_AMDGPU_METADATA`` note record
(see :ref:`amdgpu-note-records-v3`).

The metadata is represented as Message Pack formatted binary data (see
[MsgPack]_). The top level is a Message Pack map that includes the
keys defined in table
:ref:`amdgpu-amdhsa-code-object-metadata-map-table-v3` and referenced
tables.

Additional information can be added to the maps. To avoid conflicts,
any key names should be prefixed by "*vendor-name*." where
``vendor-name`` can be the name of the vendor and specific vendor
tool that generates the information. The prefix is abbreviated to
simply "." when it appears within a map that has been added by the
same *vendor-name*.

  .. table:: AMDHSA Code Object V3 Metadata Map
     :name: amdgpu-amdhsa-code-object-metadata-map-table-v3

     ================= ============== ========= =======================================
     String Key        Value Type     Required? Description
     ================= ============== ========= =======================================
     "amdhsa.version"  sequence of    Required  - The first integer is the major
                       2 integers                 version. Currently 1.
                                                - The second integer is the minor
                                                  version. Currently 0.
     "amdhsa.printf"   sequence of              Each string is encoded information
                       strings                  about a printf function call. The
                                                encoded information is organized as
                                                fields separated by colon (':'):

                                                ``ID:N:S[0]:S[1]:...:S[N-1]:FormatString``

                                                where:

                                                ``ID``
                                                  A 32-bit integer as a unique id for
                                                  each printf function call

                                                ``N``
                                                  A 32-bit integer equal to the number
                                                  of arguments of printf function call
                                                  minus 1

                                                ``S[i]`` (where i = 0, 1, ... , N-1)
                                                  32-bit integers for the size in bytes
                                                  of the i-th FormatString argument of
                                                  the printf function call

                                                FormatString
                                                  The format string passed to the
                                                  printf function call.
     "amdhsa.kernels"  sequence of    Required  Sequence of the maps for each
                       map                      kernel in the code object. See
                                                :ref:`amdgpu-amdhsa-code-object-kernel-metadata-map-table-v3`
                                                for the definition of the keys included
                                                in that map.
     ================= ============== ========= =======================================

..

  .. table:: AMDHSA Code Object V3 Kernel Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-metadata-map-table-v3

     =================================== ============== ========= ================================
     String Key                          Value Type     Required? Description
     =================================== ============== ========= ================================
     ".name"                             string         Required  Source name of the kernel.
     ".symbol"                           string         Required  Name of the kernel
                                                                  descriptor ELF symbol.
     ".language"                         string                   Source language of the kernel.
                                                                  Values include:

                                                                  - "OpenCL C"
                                                                  - "OpenCL C++"
                                                                  - "HCC"
                                                                  - "HIP"
                                                                  - "OpenMP"
                                                                  - "Assembler"

     ".language_version"                 sequence of              - The first integer is the major
                                         2 integers                 version.
                                                                  - The second integer is the
                                                                    minor version.
     ".args"                             sequence of              Sequence of maps of the
                                         map                      kernel arguments. See
                                                                  :ref:`amdgpu-amdhsa-code-object-kernel-argument-metadata-map-table-v3`
                                                                  for the definition of the keys
                                                                  included in that map.
     ".reqd_workgroup_size"              sequence of              If not 0, 0, 0 then all values
                                         3 integers               must be >=1 and the dispatch
                                                                  work-group size X, Y, Z must
                                                                  correspond to the specified
                                                                  values. Defaults to 0, 0, 0.

                                                                  Corresponds to the OpenCL
                                                                  ``reqd_work_group_size``
                                                                  attribute.
     ".workgroup_size_hint"              sequence of              The dispatch work-group size
                                         3 integers               X, Y, Z is likely to be the
                                                                  specified values.

                                                                  Corresponds to the OpenCL
                                                                  ``work_group_size_hint``
                                                                  attribute.
     ".vec_type_hint"                    string                   The name of a scalar or vector
                                                                  type.

                                                                  Corresponds to the OpenCL
                                                                  ``vec_type_hint`` attribute.

     ".device_enqueue_symbol"            string                   The external symbol name
                                                                  associated with a kernel.
                                                                  OpenCL runtime allocates a
                                                                  global buffer for the symbol
                                                                  and saves the kernel's address
                                                                  to it, which is used for
                                                                  device side enqueueing. Only
                                                                  available for device side
                                                                  enqueued kernels.
     ".kernarg_segment_size"             integer        Required  The size in bytes of
                                                                  the kernarg segment
                                                                  that holds the values
                                                                  of the arguments to
                                                                  the kernel.
     ".group_segment_fixed_size"         integer        Required  The amount of group
                                                                  segment memory
                                                                  required by a
                                                                  work-group in
                                                                  bytes. This does not
                                                                  include any
                                                                  dynamically allocated
                                                                  group segment memory
                                                                  that may be added
                                                                  when the kernel is
                                                                  dispatched.
     ".private_segment_fixed_size"       integer        Required  The amount of fixed
                                                                  private address space
                                                                  memory required for a
                                                                  work-item in
                                                                  bytes. If the kernel
                                                                  uses a dynamic call
                                                                  stack then additional
                                                                  space must be added
                                                                  to this value for the
                                                                  call stack.
     ".kernarg_segment_align"            integer        Required  The maximum byte
                                                                  alignment of
                                                                  arguments in the
                                                                  kernarg segment. Must
                                                                  be a power of 2.
     ".wavefront_size"                   integer        Required  Wavefront size. Must
                                                                  be a power of 2.
     ".sgpr_count"                       integer        Required  Number of scalar
                                                                  registers required by a
                                                                  wavefront for
                                                                  GFX6-GFX9. A register
                                                                  is required if it is
                                                                  used explicitly, or
                                                                  if a higher numbered
                                                                  register is used
                                                                  explicitly. This
                                                                  includes the special
                                                                  SGPRs for VCC, Flat
                                                                  Scratch (GFX7-GFX9)
                                                                  and XNACK (for
                                                                  GFX8-GFX9). It does
                                                                  not include the 16
                                                                  SGPR added if a trap
                                                                  handler is
                                                                  enabled. It is not
                                                                  rounded up to the
                                                                  allocation
                                                                  granularity.
     ".vgpr_count"                       integer        Required  Number of vector
                                                                  registers required by
                                                                  each work-item for
                                                                  GFX6-GFX9. A register
                                                                  is required if it is
                                                                  used explicitly, or
                                                                  if a higher numbered
                                                                  register is used
                                                                  explicitly.
     ".max_flat_workgroup_size"          integer        Required  Maximum flat
                                                                  work-group size
                                                                  supported by the
                                                                  kernel in work-items.
                                                                  Must be >=1 and
                                                                  consistent with
                                                                  ReqdWorkGroupSize if
                                                                  not 0, 0, 0.
     ".sgpr_spill_count"                 integer                  Number of stores from
                                                                  a scalar register to
                                                                  a register allocator
                                                                  created spill
                                                                  location.
     ".vgpr_spill_count"                 integer                  Number of stores from
                                                                  a vector register to
                                                                  a register allocator
                                                                  created spill
                                                                  location.
     =================================== ============== ========= ================================

..

  .. table:: AMDHSA Code Object V3 Kernel Argument Metadata Map
     :name: amdgpu-amdhsa-code-object-kernel-argument-metadata-map-table-v3

     ====================== ============== ========= ================================
     String Key             Value Type     Required? Description
     ====================== ============== ========= ================================
     ".name"                string                   Kernel argument name.
     ".type_name"           string                   Kernel argument type name.
     ".size"                integer        Required  Kernel argument size in bytes.
     ".offset"              integer        Required  Kernel argument offset in
                                                     bytes. The offset must be a
                                                     multiple of the alignment
                                                     required by the argument.
     ".value_kind"          string         Required  Kernel argument kind that
                                                     specifies how to set up the
                                                     corresponding argument.
                                                     Values include:

                                                     "by_value"
                                                       The argument is copied
                                                       directly into the kernarg.

                                                     "global_buffer"
                                                       A global address space pointer
                                                       to the buffer data is passed
                                                       in the kernarg.

                                                     "dynamic_shared_pointer"
                                                       A group address space pointer
                                                       to dynamically allocated LDS
                                                       is passed in the kernarg.

                                                     "sampler"
                                                       A global address space
                                                       pointer to a S# is passed in
                                                       the kernarg.

                                                     "image"
                                                       A global address space
                                                       pointer to a T# is passed in
                                                       the kernarg.

                                                     "pipe"
                                                       A global address space pointer
                                                       to an OpenCL pipe is passed in
                                                       the kernarg.

                                                     "queue"
                                                       A global address space pointer
                                                       to an OpenCL device enqueue
                                                       queue is passed in the
                                                       kernarg.

                                                     "hidden_global_offset_x"
                                                       The OpenCL grid dispatch
                                                       global offset for the X
                                                       dimension is passed in the
                                                       kernarg.

                                                     "hidden_global_offset_y"
                                                       The OpenCL grid dispatch
                                                       global offset for the Y
                                                       dimension is passed in the
                                                       kernarg.

                                                     "hidden_global_offset_z"
                                                       The OpenCL grid dispatch
                                                       global offset for the Z
                                                       dimension is passed in the
                                                       kernarg.

                                                     "hidden_none"
                                                       An argument that is not used
                                                       by the kernel. Space needs to
                                                       be left for it, but it does
                                                       not need to be set up.

                                                     "hidden_printf_buffer"
                                                       A global address space pointer
                                                       to the runtime printf buffer
                                                       is passed in kernarg.

                                                     "hidden_hostcall_buffer"
                                                       A global address space pointer
                                                       to the runtime hostcall buffer
                                                       is passed in kernarg.

                                                     "hidden_default_queue"
                                                       A global address space pointer
                                                       to the OpenCL device enqueue
                                                       queue that should be used by
                                                       the kernel by default is
                                                       passed in the kernarg.

                                                     "hidden_completion_action"
                                                       A global address space pointer
                                                       to help link enqueued kernels into
                                                       the ancestor tree for determining
                                                       when the parent kernel has finished.

                                                     "hidden_multigrid_sync_arg"
                                                       A global address space pointer for
                                                       multi-grid synchronization is
                                                       passed in the kernarg.

     ".value_type"          string         Required  Kernel argument value type. Only
                                                     present if ".value_kind" is
                                                     "by_value". For vector data
                                                     types, the value is for the
                                                     element type. Values include:

                                                     - "struct"
                                                     - "i8"
                                                     - "u8"
                                                     - "i16"
                                                     - "u16"
                                                     - "f16"
                                                     - "i32"
                                                     - "u32"
                                                     - "f32"
                                                     - "i64"
                                                     - "u64"
                                                     - "f64"

                                                     .. TODO::
                                                        How can it be determined if a
                                                        vector type, and what size
                                                        vector?
     ".pointee_align"       integer                  Alignment in bytes of pointee
                                                     type for pointer type kernel
                                                     argument. Must be a power
                                                     of 2. Only present if
                                                     ".value_kind" is
                                                     "dynamic_shared_pointer".
     ".address_space"       string                   Kernel argument address space
                                                     qualifier. Only present if
                                                     ".value_kind" is "global_buffer" or
                                                     "dynamic_shared_pointer". Values
                                                     are:

                                                     - "private"
                                                     - "global"
                                                     - "constant"
                                                     - "local"
                                                     - "generic"
                                                     - "region"

                                                     .. TODO::
                                                        Is "global_buffer" only "global"
                                                        or "constant"? Is
                                                        "dynamic_shared_pointer" always
                                                        "local"? Can HCC allow "generic"?
                                                        How can "private" or "region"
                                                        ever happen?
     ".access"              string                   Kernel argument access
                                                     qualifier. Only present if
                                                     ".value_kind" is "image" or
                                                     "pipe". Values
                                                     are:

                                                     - "read_only"
                                                     - "write_only"
                                                     - "read_write"

                                                     .. TODO::
                                                        Does this apply to
                                                        "global_buffer"?
     ".actual_access"       string                   The actual memory accesses
                                                     performed by the kernel on the
                                                     kernel argument. Only present if
                                                     ".value_kind" is "global_buffer",
                                                     "image", or "pipe". This may be
                                                     more restrictive than indicated
                                                     by ".access" to reflect what the
                                                     kernel actual does. If not
                                                     present then the runtime must
                                                     assume what is implied by
                                                     ".access" and ".is_const"      . Values
                                                     are:

                                                     - "read_only"
                                                     - "write_only"
                                                     - "read_write"

     ".is_const"            boolean                  Indicates if the kernel argument
                                                     is const qualified. Only present
                                                     if ".value_kind" is
                                                     "global_buffer".

     ".is_restrict"         boolean                  Indicates if the kernel argument
                                                     is restrict qualified. Only
                                                     present if ".value_kind" is
                                                     "global_buffer".

     ".is_volatile"         boolean                  Indicates if the kernel argument
                                                     is volatile qualified. Only
                                                     present if ".value_kind" is
                                                     "global_buffer".

     ".is_pipe"             boolean                  Indicates if the kernel argument
                                                     is pipe qualified. Only present
                                                     if ".value_kind" is "pipe".

                                                     .. TODO::
                                                        Can "global_buffer" be pipe
                                                        qualified?
     ====================== ============== ========= ================================

..

Kernel Dispatch
~~~~~~~~~~~~~~~

The HSA architected queuing language (AQL) defines a user space memory
interface that can be used to control the dispatch of kernels, in an agent
independent way. An agent can have zero or more AQL queues created for it using
the ROCm runtime, in which AQL packets (all of which are 64 bytes) can be
placed. See the *HSA Platform System Architecture Specification* [HSA]_ for the
AQL queue mechanics and packet layouts.

The packet processor of a kernel agent is responsible for detecting and
dispatching HSA kernels from the AQL queues associated with it. For AMD GPUs the
packet processor is implemented by the hardware command processor (CP),
asynchronous dispatch controller (ADC) and shader processor input controller
(SPI).

The ROCm runtime can be used to allocate an AQL queue object. It uses the kernel
mode driver to initialize and register the AQL queue with CP.

To dispatch a kernel the following actions are performed. This can occur in the
CPU host program, or from an HSA kernel executing on a GPU.

1. A pointer to an AQL queue for the kernel agent on which the kernel is to be
   executed is obtained.
2. A pointer to the kernel descriptor (see
   :ref:`amdgpu-amdhsa-kernel-descriptor`) of the kernel to execute is obtained.
   It must be for a kernel that is contained in a code object that that was
   loaded by the ROCm runtime on the kernel agent with which the AQL queue is
   associated.
3. Space is allocated for the kernel arguments using the ROCm runtime allocator
   for a memory region with the kernarg property for the kernel agent that will
   execute the kernel. It must be at least 16-byte aligned.
4. Kernel argument values are assigned to the kernel argument memory
   allocation. The layout is defined in the *HSA Programmer's Language
   Reference* [HSA]_. For AMDGPU the kernel execution directly accesses the
   kernel argument memory in the same way constant memory is accessed. (Note
   that the HSA specification allows an implementation to copy the kernel
   argument contents to another location that is accessed by the kernel.)
5. An AQL kernel dispatch packet is created on the AQL queue. The ROCm runtime
   api uses 64-bit atomic operations to reserve space in the AQL queue for the
   packet. The packet must be set up, and the final write must use an atomic
   store release to set the packet kind to ensure the packet contents are
   visible to the kernel agent. AQL defines a doorbell signal mechanism to
   notify the kernel agent that the AQL queue has been updated. These rules, and
   the layout of the AQL queue and kernel dispatch packet is defined in the *HSA
   System Architecture Specification* [HSA]_.
6. A kernel dispatch packet includes information about the actual dispatch,
   such as grid and work-group size, together with information from the code
   object about the kernel, such as segment sizes. The ROCm runtime queries on
   the kernel symbol can be used to obtain the code object values which are
   recorded in the :ref:`amdgpu-amdhsa-code-object-metadata`.
7. CP executes micro-code and is responsible for detecting and setting up the
   GPU to execute the wavefronts of a kernel dispatch.
8. CP ensures that when the a wavefront starts executing the kernel machine
   code, the scalar general purpose registers (SGPR) and vector general purpose
   registers (VGPR) are set up as required by the machine code. The required
   setup is defined in the :ref:`amdgpu-amdhsa-kernel-descriptor`. The initial
   register state is defined in
   :ref:`amdgpu-amdhsa-initial-kernel-execution-state`.
9. The prolog of the kernel machine code (see
   :ref:`amdgpu-amdhsa-kernel-prolog`) sets up the machine state as necessary
   before continuing executing the machine code that corresponds to the kernel.
10. When the kernel dispatch has completed execution, CP signals the completion
    signal specified in the kernel dispatch packet if not 0.

Image and Samplers
~~~~~~~~~~~~~~~~~~

Image and sample handles created by the ROCm runtime are 64-bit addresses of a
hardware 32-byte V# and 48 byte S# object respectively. In order to support the
HSA ``query_sampler`` operations two extra dwords are used to store the HSA BRIG
enumeration values for the queries that are not trivially deducible from the S#
representation.

HSA Signals
~~~~~~~~~~~

HSA signal handles created by the ROCm runtime are 64-bit addresses of a
structure allocated in memory accessible from both the CPU and GPU. The
structure is defined by the ROCm runtime and subject to change between releases
(see [AMD-ROCm-github]_).

.. _amdgpu-amdhsa-hsa-aql-queue:

HSA AQL Queue
~~~~~~~~~~~~~

The HSA AQL queue structure is defined by the ROCm runtime and subject to change
between releases (see [AMD-ROCm-github]_). For some processors it contains
fields needed to implement certain language features such as the flat address
aperture bases. It also contains fields used by CP such as managing the
allocation of scratch memory.

.. _amdgpu-amdhsa-kernel-descriptor:

Kernel Descriptor
~~~~~~~~~~~~~~~~~

A kernel descriptor consists of the information needed by CP to initiate the
execution of a kernel, including the entry point address of the machine code
that implements the kernel.

Kernel Descriptor for GFX6-GFX10
++++++++++++++++++++++++++++++++

CP microcode requires the Kernel descriptor to be allocated on 64-byte
alignment.

  .. table:: Kernel Descriptor for GFX6-GFX10
     :name: amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table

     ======= ======= =============================== ============================
     Bits    Size    Field Name                      Description
     ======= ======= =============================== ============================
     31:0    4 bytes GROUP_SEGMENT_FIXED_SIZE        The amount of fixed local
                                                     address space memory
                                                     required for a work-group
                                                     in bytes. This does not
                                                     include any dynamically
                                                     allocated local address
                                                     space memory that may be
                                                     added when the kernel is
                                                     dispatched.
     63:32   4 bytes PRIVATE_SEGMENT_FIXED_SIZE      The amount of fixed
                                                     private address space
                                                     memory required for a
                                                     work-item in bytes. If
                                                     is_dynamic_callstack is 1
                                                     then additional space must
                                                     be added to this value for
                                                     the call stack.
     127:64  8 bytes                                 Reserved, must be 0.
     191:128 8 bytes KERNEL_CODE_ENTRY_BYTE_OFFSET   Byte offset (possibly
                                                     negative) from base
                                                     address of kernel
                                                     descriptor to kernel's
                                                     entry point instruction
                                                     which must be 256 byte
                                                     aligned.
     351:272 20                                      Reserved, must be 0.
             bytes
     383:352 4 bytes COMPUTE_PGM_RSRC3               GFX6-9
                                                       Reserved, must be 0.
                                                     GFX10
                                                       Compute Shader (CS)
                                                       program settings used by
                                                       CP to set up
                                                       ``COMPUTE_PGM_RSRC3``
                                                       configuration
                                                       register. See
                                                       :ref:`amdgpu-amdhsa-compute_pgm_rsrc3-gfx10-table`.
     415:384 4 bytes COMPUTE_PGM_RSRC1               Compute Shader (CS)
                                                     program settings used by
                                                     CP to set up
                                                     ``COMPUTE_PGM_RSRC1``
                                                     configuration
                                                     register. See
                                                     :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     447:416 4 bytes COMPUTE_PGM_RSRC2               Compute Shader (CS)
                                                     program settings used by
                                                     CP to set up
                                                     ``COMPUTE_PGM_RSRC2``
                                                     configuration
                                                     register. See
                                                     :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     448     1 bit   ENABLE_SGPR_PRIVATE_SEGMENT     Enable the setup of the
                     _BUFFER                         SGPR user data registers
                                                     (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     The total number of SGPR
                                                     user data registers
                                                     requested must not exceed
                                                     16 and match value in
                                                     ``compute_pgm_rsrc2.user_sgpr.user_sgpr_count``.
                                                     Any requests beyond 16
                                                     will be ignored.
     449     1 bit   ENABLE_SGPR_DISPATCH_PTR        *see above*
     450     1 bit   ENABLE_SGPR_QUEUE_PTR           *see above*
     451     1 bit   ENABLE_SGPR_KERNARG_SEGMENT_PTR *see above*
     452     1 bit   ENABLE_SGPR_DISPATCH_ID         *see above*
     453     1 bit   ENABLE_SGPR_FLAT_SCRATCH_INIT   *see above*
     454     1 bit   ENABLE_SGPR_PRIVATE_SEGMENT     *see above*
                     _SIZE
     457:455 3 bits                                  Reserved, must be 0.
     458     1 bit   ENABLE_WAVEFRONT_SIZE32         GFX6-9
                                                       Reserved, must be 0.
                                                     GFX10
                                                       - If 0 execute in
                                                         wavefront size 64 mode.
                                                       - If 1 execute in
                                                         native wavefront size
                                                         32 mode.
     463:459 5 bits                                  Reserved, must be 0.
     511:464 6 bytes                                 Reserved, must be 0.
     512     **Total size 64 bytes.**
     ======= ====================================================================

..

  .. table:: compute_pgm_rsrc1 for GFX6-GFX10
     :name: amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table

     ======= ======= =============================== ===========================================================================
     Bits    Size    Field Name                      Description
     ======= ======= =============================== ===========================================================================
     5:0     6 bits  GRANULATED_WORKITEM_VGPR_COUNT  Number of vector register
                                                     blocks used by each work-item;
                                                     granularity is device
                                                     specific:

                                                     GFX6-GFX9
                                                       - vgprs_used 0..256
                                                       - max(0, ceil(vgprs_used / 4) - 1)
                                                     GFX10 (wavefront size 64)
                                                       - max_vgpr 1..256
                                                       - max(0, ceil(vgprs_used / 4) - 1)
                                                     GFX10 (wavefront size 32)
                                                       - max_vgpr 1..256
                                                       - max(0, ceil(vgprs_used / 8) - 1)

                                                     Where vgprs_used is defined
                                                     as the highest VGPR number
                                                     explicitly referenced plus
                                                     one.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.VGPRS``.

                                                     The
                                                     :ref:`amdgpu-assembler`
                                                     calculates this
                                                     automatically for the
                                                     selected processor from
                                                     values provided to the
                                                     `.amdhsa_kernel` directive
                                                     by the
                                                     `.amdhsa_next_free_vgpr`
                                                     nested directive (see
                                                     :ref:`amdhsa-kernel-directives-table`).
     9:6     4 bits  GRANULATED_WAVEFRONT_SGPR_COUNT Number of scalar register
                                                     blocks used by a wavefront;
                                                     granularity is device
                                                     specific:

                                                     GFX6-GFX8
                                                       - sgprs_used 0..112
                                                       - max(0, ceil(sgprs_used / 8) - 1)
                                                     GFX9
                                                       - sgprs_used 0..112
                                                       - 2 * max(0, ceil(sgprs_used / 16) - 1)
                                                     GFX10
                                                       Reserved, must be 0.
                                                       (128 SGPRs always
                                                       allocated.)

                                                     Where sgprs_used is
                                                     defined as the highest
                                                     SGPR number explicitly
                                                     referenced plus one, plus
                                                     a target specific number
                                                     of additional special
                                                     SGPRs for VCC,
                                                     FLAT_SCRATCH (GFX7+) and
                                                     XNACK_MASK (GFX8+), and
                                                     any additional
                                                     target specific
                                                     limitations. It does not
                                                     include the 16 SGPRs added
                                                     if a trap handler is
                                                     enabled.

                                                     The target specific
                                                     limitations and special
                                                     SGPR layout are defined in
                                                     the hardware
                                                     documentation, which can
                                                     be found in the
                                                     :ref:`amdgpu-processors`
                                                     table.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.SGPRS``.

                                                     The
                                                     :ref:`amdgpu-assembler`
                                                     calculates this
                                                     automatically for the
                                                     selected processor from
                                                     values provided to the
                                                     `.amdhsa_kernel` directive
                                                     by the
                                                     `.amdhsa_next_free_sgpr`
                                                     and `.amdhsa_reserve_*`
                                                     nested directives (see
                                                     :ref:`amdhsa-kernel-directives-table`).
     11:10   2 bits  PRIORITY                        Must be 0.

                                                     Start executing wavefront
                                                     at the specified priority.

                                                     CP is responsible for
                                                     filling in
                                                     ``COMPUTE_PGM_RSRC1.PRIORITY``.
     13:12   2 bits  FLOAT_ROUND_MODE_32             Wavefront starts execution
                                                     with specified rounding
                                                     mode for single (32
                                                     bit) floating point
                                                     precision floating point
                                                     operations.

                                                     Floating point rounding
                                                     mode values are defined in
                                                     :ref:`amdgpu-amdhsa-floating-point-rounding-mode-enumeration-values-table`.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.FLOAT_MODE``.
     15:14   2 bits  FLOAT_ROUND_MODE_16_64          Wavefront starts execution
                                                     with specified rounding
                                                     denorm mode for half/double (16
                                                     and 64-bit) floating point
                                                     precision floating point
                                                     operations.

                                                     Floating point rounding
                                                     mode values are defined in
                                                     :ref:`amdgpu-amdhsa-floating-point-rounding-mode-enumeration-values-table`.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.FLOAT_MODE``.
     17:16   2 bits  FLOAT_DENORM_MODE_32            Wavefront starts execution
                                                     with specified denorm mode
                                                     for single (32
                                                     bit)  floating point
                                                     precision floating point
                                                     operations.

                                                     Floating point denorm mode
                                                     values are defined in
                                                     :ref:`amdgpu-amdhsa-floating-point-denorm-mode-enumeration-values-table`.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.FLOAT_MODE``.
     19:18   2 bits  FLOAT_DENORM_MODE_16_64         Wavefront starts execution
                                                     with specified denorm mode
                                                     for half/double (16
                                                     and 64-bit) floating point
                                                     precision floating point
                                                     operations.

                                                     Floating point denorm mode
                                                     values are defined in
                                                     :ref:`amdgpu-amdhsa-floating-point-denorm-mode-enumeration-values-table`.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.FLOAT_MODE``.
     20      1 bit   PRIV                            Must be 0.

                                                     Start executing wavefront
                                                     in privilege trap handler
                                                     mode.

                                                     CP is responsible for
                                                     filling in
                                                     ``COMPUTE_PGM_RSRC1.PRIV``.
     21      1 bit   ENABLE_DX10_CLAMP               Wavefront starts execution
                                                     with DX10 clamp mode
                                                     enabled. Used by the vector
                                                     ALU to force DX10 style
                                                     treatment of NaN's (when
                                                     set, clamp NaN to zero,
                                                     otherwise pass NaN
                                                     through).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.DX10_CLAMP``.
     22      1 bit   DEBUG_MODE                      Must be 0.

                                                     Start executing wavefront
                                                     in single step mode.

                                                     CP is responsible for
                                                     filling in
                                                     ``COMPUTE_PGM_RSRC1.DEBUG_MODE``.
     23      1 bit   ENABLE_IEEE_MODE                Wavefront starts execution
                                                     with IEEE mode
                                                     enabled. Floating point
                                                     opcodes that support
                                                     exception flag gathering
                                                     will quiet and propagate
                                                     signaling-NaN inputs per
                                                     IEEE 754-2008. Min_dx10 and
                                                     max_dx10 become IEEE
                                                     754-2008 compliant due to
                                                     signaling-NaN propagation
                                                     and quieting.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC1.IEEE_MODE``.
     24      1 bit   BULKY                           Must be 0.

                                                     Only one work-group allowed
                                                     to execute on a compute
                                                     unit.

                                                     CP is responsible for
                                                     filling in
                                                     ``COMPUTE_PGM_RSRC1.BULKY``.
     25      1 bit   CDBG_USER                       Must be 0.

                                                     Flag that can be used to
                                                     control debugging code.

                                                     CP is responsible for
                                                     filling in
                                                     ``COMPUTE_PGM_RSRC1.CDBG_USER``.
     26      1 bit   FP16_OVFL                       GFX6-GFX8
                                                       Reserved, must be 0.
                                                     GFX9-GFX10
                                                       Wavefront starts execution
                                                       with specified fp16 overflow
                                                       mode.

                                                       - If 0, fp16 overflow generates
                                                         +/-INF values.
                                                       - If 1, fp16 overflow that is the
                                                         result of an +/-INF input value
                                                         or divide by 0 produces a +/-INF,
                                                         otherwise clamps computed
                                                         overflow to +/-MAX_FP16 as
                                                         appropriate.

                                                       Used by CP to set up
                                                       ``COMPUTE_PGM_RSRC1.FP16_OVFL``.
     28:27   2 bits                                  Reserved, must be 0.
     29      1 bit    WGP_MODE                       GFX6-GFX9
                                                       Reserved, must be 0.
                                                     GFX10
                                                       - If 0 execute work-groups in
                                                         CU wavefront execution mode.
                                                       - If 1 execute work-groups on
                                                         in WGP wavefront execution mode.

                                                       See :ref:`amdgpu-amdhsa-memory-model`.

                                                       Used by CP to set up
                                                       ``COMPUTE_PGM_RSRC1.WGP_MODE``.
     30      1 bit    MEM_ORDERED                    GFX6-9
                                                       Reserved, must be 0.
                                                     GFX10
                                                       Controls the behavior of the
                                                       waitcnt's vmcnt and vscnt
                                                       counters.

                                                       - If 0 vmcnt reports completion
                                                         of load and atomic with return
                                                         out of order with sample
                                                         instructions, and the vscnt
                                                         reports the completion of
                                                         store and atomic without
                                                         return in order.
                                                       - If 1 vmcnt reports completion
                                                         of load, atomic with return
                                                         and sample instructions in
                                                         order, and the vscnt reports
                                                         the completion of store and
                                                         atomic without return in order.

                                                       Used by CP to set up
                                                       ``COMPUTE_PGM_RSRC1.MEM_ORDERED``.
     31      1 bit    FWD_PROGRESS                   GFX6-9
                                                       Reserved, must be 0.
                                                     GFX10
                                                       - If 0 execute SIMD wavefronts
                                                         using oldest first policy.
                                                       - If 1 execute SIMD wavefronts to
                                                         ensure wavefronts will make some
                                                         forward progress.

                                                       Used by CP to set up
                                                       ``COMPUTE_PGM_RSRC1.FWD_PROGRESS``.
     32      **Total size 4 bytes**
     ======= ===================================================================================================================

..

  .. table:: compute_pgm_rsrc2 for GFX6-GFX10
     :name: amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table

     ======= ======= =============================== ===========================================================================
     Bits    Size    Field Name                      Description
     ======= ======= =============================== ===========================================================================
     0       1 bit   ENABLE_SGPR_PRIVATE_SEGMENT     Enable the setup of the
                     _WAVEFRONT_OFFSET               SGPR wavefront scratch offset
                                                     system register (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.SCRATCH_EN``.
     5:1     5 bits  USER_SGPR_COUNT                 The total number of SGPR
                                                     user data registers
                                                     requested. This number must
                                                     match the number of user
                                                     data registers enabled.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.USER_SGPR``.
     6       1 bit   ENABLE_TRAP_HANDLER             Must be 0.

                                                     This bit represents
                                                     ``COMPUTE_PGM_RSRC2.TRAP_PRESENT``,
                                                     which is set by the CP if
                                                     the runtime has installed a
                                                     trap handler.
     7       1 bit   ENABLE_SGPR_WORKGROUP_ID_X      Enable the setup of the
                                                     system SGPR register for
                                                     the work-group id in the X
                                                     dimension (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.TGID_X_EN``.
     8       1 bit   ENABLE_SGPR_WORKGROUP_ID_Y      Enable the setup of the
                                                     system SGPR register for
                                                     the work-group id in the Y
                                                     dimension (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.TGID_Y_EN``.
     9       1 bit   ENABLE_SGPR_WORKGROUP_ID_Z      Enable the setup of the
                                                     system SGPR register for
                                                     the work-group id in the Z
                                                     dimension (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.TGID_Z_EN``.
     10      1 bit   ENABLE_SGPR_WORKGROUP_INFO      Enable the setup of the
                                                     system SGPR register for
                                                     work-group information (see
                                                     :ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.TGID_SIZE_EN``.
     12:11   2 bits  ENABLE_VGPR_WORKITEM_ID         Enable the setup of the
                                                     VGPR system registers used
                                                     for the work-item ID.
                                                     :ref:`amdgpu-amdhsa-system-vgpr-work-item-id-enumeration-values-table`
                                                     defines the values.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.TIDIG_CMP_CNT``.
     13      1 bit   ENABLE_EXCEPTION_ADDRESS_WATCH  Must be 0.

                                                     Wavefront starts execution
                                                     with address watch
                                                     exceptions enabled which
                                                     are generated when L1 has
                                                     witnessed a thread access
                                                     an *address of
                                                     interest*.

                                                     CP is responsible for
                                                     filling in the address
                                                     watch bit in
                                                     ``COMPUTE_PGM_RSRC2.EXCP_EN_MSB``
                                                     according to what the
                                                     runtime requests.
     14      1 bit   ENABLE_EXCEPTION_MEMORY         Must be 0.

                                                     Wavefront starts execution
                                                     with memory violation
                                                     exceptions exceptions
                                                     enabled which are generated
                                                     when a memory violation has
                                                     occurred for this wavefront from
                                                     L1 or LDS
                                                     (write-to-read-only-memory,
                                                     mis-aligned atomic, LDS
                                                     address out of range,
                                                     illegal address, etc.).

                                                     CP sets the memory
                                                     violation bit in
                                                     ``COMPUTE_PGM_RSRC2.EXCP_EN_MSB``
                                                     according to what the
                                                     runtime requests.
     23:15   9 bits  GRANULATED_LDS_SIZE             Must be 0.

                                                     CP uses the rounded value
                                                     from the dispatch packet,
                                                     not this value, as the
                                                     dispatch may contain
                                                     dynamically allocated group
                                                     segment memory. CP writes
                                                     directly to
                                                     ``COMPUTE_PGM_RSRC2.LDS_SIZE``.

                                                     Amount of group segment
                                                     (LDS) to allocate for each
                                                     work-group. Granularity is
                                                     device specific:

                                                     GFX6:
                                                       roundup(lds-size / (64 * 4))
                                                     GFX7-GFX10:
                                                       roundup(lds-size / (128 * 4))

     24      1 bit   ENABLE_EXCEPTION_IEEE_754_FP    Wavefront starts execution
                     _INVALID_OPERATION              with specified exceptions
                                                     enabled.

                                                     Used by CP to set up
                                                     ``COMPUTE_PGM_RSRC2.EXCP_EN``
                                                     (set from bits 0..6).

                                                     IEEE 754 FP Invalid
                                                     Operation
     25      1 bit   ENABLE_EXCEPTION_FP_DENORMAL    FP Denormal one or more
                     _SOURCE                         input operands is a
                                                     denormal number
     26      1 bit   ENABLE_EXCEPTION_IEEE_754_FP    IEEE 754 FP Division by
                     _DIVISION_BY_ZERO               Zero
     27      1 bit   ENABLE_EXCEPTION_IEEE_754_FP    IEEE 754 FP FP Overflow
                     _OVERFLOW
     28      1 bit   ENABLE_EXCEPTION_IEEE_754_FP    IEEE 754 FP Underflow
                     _UNDERFLOW
     29      1 bit   ENABLE_EXCEPTION_IEEE_754_FP    IEEE 754 FP Inexact
                     _INEXACT
     30      1 bit   ENABLE_EXCEPTION_INT_DIVIDE_BY  Integer Division by Zero
                     _ZERO                           (rcp_iflag_f32 instruction
                                                     only)
     31      1 bit                                   Reserved, must be 0.
     32      **Total size 4 bytes.**
     ======= ===================================================================================================================

..

  .. table:: compute_pgm_rsrc3 for GFX10
     :name: amdgpu-amdhsa-compute_pgm_rsrc3-gfx10-table

     ======= ======= =============================== ===========================================================================
     Bits    Size    Field Name                      Description
     ======= ======= =============================== ===========================================================================
     3:0     4 bits  SHARED_VGPR_COUNT               Number of shared VGPRs for wavefront size 64. Granularity 8. Value 0-120.
                                                     compute_pgm_rsrc1.vgprs + shared_vgpr_cnt cannot exceed 64.
     31:4    28                                      Reserved, must be 0.
             bits
     32      **Total size 4 bytes.**
     ======= ===================================================================================================================

..

  .. table:: Floating Point Rounding Mode Enumeration Values
     :name: amdgpu-amdhsa-floating-point-rounding-mode-enumeration-values-table

     ====================================== ===== ==============================
     Enumeration Name                       Value Description
     ====================================== ===== ==============================
     FLOAT_ROUND_MODE_NEAR_EVEN             0     Round Ties To Even
     FLOAT_ROUND_MODE_PLUS_INFINITY         1     Round Toward +infinity
     FLOAT_ROUND_MODE_MINUS_INFINITY        2     Round Toward -infinity
     FLOAT_ROUND_MODE_ZERO                  3     Round Toward 0
     ====================================== ===== ==============================

..

  .. table:: Floating Point Denorm Mode Enumeration Values
     :name: amdgpu-amdhsa-floating-point-denorm-mode-enumeration-values-table

     ====================================== ===== ==============================
     Enumeration Name                       Value Description
     ====================================== ===== ==============================
     FLOAT_DENORM_MODE_FLUSH_SRC_DST        0     Flush Source and Destination
                                                  Denorms
     FLOAT_DENORM_MODE_FLUSH_DST            1     Flush Output Denorms
     FLOAT_DENORM_MODE_FLUSH_SRC            2     Flush Source Denorms
     FLOAT_DENORM_MODE_FLUSH_NONE           3     No Flush
     ====================================== ===== ==============================

..

  .. table:: System VGPR Work-Item ID Enumeration Values
     :name: amdgpu-amdhsa-system-vgpr-work-item-id-enumeration-values-table

     ======================================== ===== ============================
     Enumeration Name                         Value Description
     ======================================== ===== ============================
     SYSTEM_VGPR_WORKITEM_ID_X                0     Set work-item X dimension
                                                    ID.
     SYSTEM_VGPR_WORKITEM_ID_X_Y              1     Set work-item X and Y
                                                    dimensions ID.
     SYSTEM_VGPR_WORKITEM_ID_X_Y_Z            2     Set work-item X, Y and Z
                                                    dimensions ID.
     SYSTEM_VGPR_WORKITEM_ID_UNDEFINED        3     Undefined.
     ======================================== ===== ============================

.. _amdgpu-amdhsa-initial-kernel-execution-state:

Initial Kernel Execution State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section defines the register state that will be set up by the packet
processor prior to the start of execution of every wavefront. This is limited by
the constraints of the hardware controllers of CP/ADC/SPI.

The order of the SGPR registers is defined, but the compiler can specify which
ones are actually setup in the kernel descriptor using the ``enable_sgpr_*`` bit
fields (see :ref:`amdgpu-amdhsa-kernel-descriptor`). The register numbers used
for enabled registers are dense starting at SGPR0: the first enabled register is
SGPR0, the next enabled register is SGPR1 etc.; disabled registers do not have
an SGPR number.

The initial SGPRs comprise up to 16 User SRGPs that are set by CP and apply to
all wavefronts of the grid. It is possible to specify more than 16 User SGPRs
using the ``enable_sgpr_*`` bit fields, in which case only the first 16 are
actually initialized. These are then immediately followed by the System SGPRs
that are set up by ADC/SPI and can have different values for each wavefront of
the grid dispatch.

SGPR register initial state is defined in
:ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

  .. table:: SGPR Register Set Up Order
     :name: amdgpu-amdhsa-sgpr-register-set-up-order-table

     ========== ========================== ====== ==============================
     SGPR Order Name                       Number Description
                (kernel descriptor enable  of
                field)                     SGPRs
     ========== ========================== ====== ==============================
     First      Private Segment Buffer     4      V# that can be used, together
                (enable_sgpr_private              with Scratch Wavefront Offset
                _segment_buffer)                  as an offset, to access the
                                                  private address space using a
                                                  segment address.

                                                  CP uses the value provided by
                                                  the runtime.
     then       Dispatch Ptr               2      64-bit address of AQL dispatch
                (enable_sgpr_dispatch_ptr)        packet for kernel dispatch
                                                  actually executing.
     then       Queue Ptr                  2      64-bit address of amd_queue_t
                (enable_sgpr_queue_ptr)           object for AQL queue on which
                                                  the dispatch packet was
                                                  queued.
     then       Kernarg Segment Ptr        2      64-bit address of Kernarg
                (enable_sgpr_kernarg              segment. This is directly
                _segment_ptr)                     copied from the
                                                  kernarg_address in the kernel
                                                  dispatch packet.

                                                  Having CP load it once avoids
                                                  loading it at the beginning of
                                                  every wavefront.
     then       Dispatch Id                2      64-bit Dispatch ID of the
                (enable_sgpr_dispatch_id)         dispatch packet being
                                                  executed.
     then       Flat Scratch Init          2      This is 2 SGPRs:
                (enable_sgpr_flat_scratch
                _init)                            GFX6
                                                    Not supported.
                                                  GFX7-GFX8
                                                    The first SGPR is a 32-bit
                                                    byte offset from
                                                    ``SH_HIDDEN_PRIVATE_BASE_VIMID``
                                                    to per SPI base of memory
                                                    for scratch for the queue
                                                    executing the kernel
                                                    dispatch. CP obtains this
                                                    from the runtime. (The
                                                    Scratch Segment Buffer base
                                                    address is
                                                    ``SH_HIDDEN_PRIVATE_BASE_VIMID``
                                                    plus this offset.) The value
                                                    of Scratch Wavefront Offset must
                                                    be added to this offset by
                                                    the kernel machine code,
                                                    right shifted by 8, and
                                                    moved to the FLAT_SCRATCH_HI
                                                    SGPR register.
                                                    FLAT_SCRATCH_HI corresponds
                                                    to SGPRn-4 on GFX7, and
                                                    SGPRn-6 on GFX8 (where SGPRn
                                                    is the highest numbered SGPR
                                                    allocated to the wavefront).
                                                    FLAT_SCRATCH_HI is
                                                    multiplied by 256 (as it is
                                                    in units of 256 bytes) and
                                                    added to
                                                    ``SH_HIDDEN_PRIVATE_BASE_VIMID``
                                                    to calculate the per wavefront
                                                    FLAT SCRATCH BASE in flat
                                                    memory instructions that
                                                    access the scratch
                                                    aperture.

                                                    The second SGPR is 32-bit
                                                    byte size of a single
                                                    work-item's scratch memory
                                                    usage. CP obtains this from
                                                    the runtime, and it is
                                                    always a multiple of DWORD.
                                                    CP checks that the value in
                                                    the kernel dispatch packet
                                                    Private Segment Byte Size is
                                                    not larger and requests the
                                                    runtime to increase the
                                                    queue's scratch size if
                                                    necessary. The kernel code
                                                    must move it to
                                                    FLAT_SCRATCH_LO which is
                                                    SGPRn-3 on GFX7 and SGPRn-5
                                                    on GFX8. FLAT_SCRATCH_LO is
                                                    used as the FLAT SCRATCH
                                                    SIZE in flat memory
                                                    instructions. Having CP load
                                                    it once avoids loading it at
                                                    the beginning of every
                                                    wavefront.
                                                  GFX9-GFX10
                                                    This is the
                                                    64-bit base address of the
                                                    per SPI scratch backing
                                                    memory managed by SPI for
                                                    the queue executing the
                                                    kernel dispatch. CP obtains
                                                    this from the runtime (and
                                                    divides it if there are
                                                    multiple Shader Arrays each
                                                    with its own SPI). The value
                                                    of Scratch Wavefront Offset must
                                                    be added by the kernel
                                                    machine code and the result
                                                    moved to the FLAT_SCRATCH
                                                    SGPR which is SGPRn-6 and
                                                    SGPRn-5. It is used as the
                                                    FLAT SCRATCH BASE in flat
                                                    memory instructions.
     then       Private Segment Size       1      The 32-bit byte size of a
                                                  (enable_sgpr_private single
                                                  work-item's
                                                  scratch_segment_size) memory
                                                  allocation. This is the
                                                  value from the kernel
                                                  dispatch packet Private
                                                  Segment Byte Size rounded up
                                                  by CP to a multiple of
                                                  DWORD.

                                                  Having CP load it once avoids
                                                  loading it at the beginning of
                                                  every wavefront.

                                                  This is not used for
                                                  GFX7-GFX8 since it is the same
                                                  value as the second SGPR of
                                                  Flat Scratch Init. However, it
                                                  may be needed for GFX9-GFX10 which
                                                  changes the meaning of the
                                                  Flat Scratch Init value.
     then       Grid Work-Group Count X    1      32-bit count of the number of
                (enable_sgpr_grid                 work-groups in the X dimension
                _workgroup_count_X)               for the grid being
                                                  executed. Computed from the
                                                  fields in the kernel dispatch
                                                  packet as ((grid_size.x +
                                                  workgroup_size.x - 1) /
                                                  workgroup_size.x).
     then       Grid Work-Group Count Y    1      32-bit count of the number of
                (enable_sgpr_grid                 work-groups in the Y dimension
                _workgroup_count_Y &&             for the grid being
                less than 16 previous             executed. Computed from the
                SGPRs)                            fields in the kernel dispatch
                                                  packet as ((grid_size.y +
                                                  workgroup_size.y - 1) /
                                                  workgroupSize.y).

                                                  Only initialized if <16
                                                  previous SGPRs initialized.
     then       Grid Work-Group Count Z    1      32-bit count of the number of
                (enable_sgpr_grid                 work-groups in the Z dimension
                _workgroup_count_Z &&             for the grid being
                less than 16 previous             executed. Computed from the
                SGPRs)                            fields in the kernel dispatch
                                                  packet as ((grid_size.z +
                                                  workgroup_size.z - 1) /
                                                  workgroupSize.z).

                                                  Only initialized if <16
                                                  previous SGPRs initialized.
     then       Work-Group Id X            1      32-bit work-group id in X
                (enable_sgpr_workgroup_id         dimension of grid for
                _X)                               wavefront.
     then       Work-Group Id Y            1      32-bit work-group id in Y
                (enable_sgpr_workgroup_id         dimension of grid for
                _Y)                               wavefront.
     then       Work-Group Id Z            1      32-bit work-group id in Z
                (enable_sgpr_workgroup_id         dimension of grid for
                _Z)                               wavefront.
     then       Work-Group Info            1      {first_wavefront, 14'b0000,
                (enable_sgpr_workgroup            ordered_append_term[10:0],
                _info)                            threadgroup_size_in_wavefronts[5:0]}
     then       Scratch Wavefront Offset   1      32-bit byte offset from base
                (enable_sgpr_private              of scratch base of queue
                _segment_wavefront_offset)        executing the kernel
                                                  dispatch. Must be used as an
                                                  offset with Private
                                                  segment address when using
                                                  Scratch Segment Buffer. It
                                                  must be used to set up FLAT
                                                  SCRATCH for flat addressing
                                                  (see
                                                  :ref:`amdgpu-amdhsa-kernel-prolog-flat-scratch`).
     ========== ========================== ====== ==============================

The order of the VGPR registers is defined, but the compiler can specify which
ones are actually setup in the kernel descriptor using the ``enable_vgpr*`` bit
fields (see :ref:`amdgpu-amdhsa-kernel-descriptor`). The register numbers used
for enabled registers are dense starting at VGPR0: the first enabled register is
VGPR0, the next enabled register is VGPR1 etc.; disabled registers do not have a
VGPR number.

VGPR register initial state is defined in
:ref:`amdgpu-amdhsa-vgpr-register-set-up-order-table`.

  .. table:: VGPR Register Set Up Order
     :name: amdgpu-amdhsa-vgpr-register-set-up-order-table

     ========== ========================== ====== ==============================
     VGPR Order Name                       Number Description
                (kernel descriptor enable  of
                field)                     VGPRs
     ========== ========================== ====== ==============================
     First      Work-Item Id X             1      32-bit work item id in X
                (Always initialized)              dimension of work-group for
                                                  wavefront lane.
     then       Work-Item Id Y             1      32-bit work item id in Y
                (enable_vgpr_workitem_id          dimension of work-group for
                > 0)                              wavefront lane.
     then       Work-Item Id Z             1      32-bit work item id in Z
                (enable_vgpr_workitem_id          dimension of work-group for
                > 1)                              wavefront lane.
     ========== ========================== ====== ==============================

The setting of registers is done by GPU CP/ADC/SPI hardware as follows:

1. SGPRs before the Work-Group Ids are set by CP using the 16 User Data
   registers.
2. Work-group Id registers X, Y, Z are set by ADC which supports any
   combination including none.
3. Scratch Wavefront Offset is set by SPI in a per wavefront basis which is why
   its value cannot be included with the flat scratch init value which is per
   queue.
4. The VGPRs are set by SPI which only supports specifying either (X), (X, Y)
   or (X, Y, Z).

Flat Scratch register pair are adjacent SGRRs so they can be moved as a 64-bit
value to the hardware required SGPRn-3 and SGPRn-4 respectively.

The global segment can be accessed either using buffer instructions (GFX6 which
has V# 64-bit address support), flat instructions (GFX7-GFX10), or global
instructions (GFX9-GFX10).

If buffer operations are used, then the compiler can generate a V# with the
following properties:

* base address of 0
* no swizzle
* ATC: 1 if IOMMU present (such as APU)
* ptr64: 1
* MTYPE set to support memory coherence that matches the runtime (such as CC for
  APU and NC for dGPU).

.. _amdgpu-amdhsa-kernel-prolog:

Kernel Prolog
~~~~~~~~~~~~~

The compiler performs initialization in the kernel prologue depending on the
target and information about things like stack usage in the kernel and called
functions. Some of this initialization requires the compiler to request certain
User and System SGPRs be present in the
:ref:`amdgpu-amdhsa-initial-kernel-execution-state` via the
:ref:`amdgpu-amdhsa-kernel-descriptor`.

.. _amdgpu-amdhsa-kernel-prolog-cfi:

CFI
+++

1.  The CFI return address is undefined.

2.  The CFI CFA is defined using an expression which evaluates to a location
    description that comprises one memory location description for the
    ``DW_ASPACE_AMDGPU_private_lane`` address space address ``0``.

.. _amdgpu-amdhsa-kernel-prolog-m0:

M0
++

GFX6-GFX8
  The M0 register must be initialized with a value at least the total LDS size
  if the kernel may access LDS via DS or flat operations. Total LDS size is
  available in dispatch packet. For M0, it is also possible to use maximum
  possible value of LDS for given target (0x7FFF for GFX6 and 0xFFFF for
  GFX7-GFX8).
GFX9-GFX10
  The M0 register is not used for range checking LDS accesses and so does not
  need to be initialized in the prolog.

.. _amdgpu-amdhsa-kernel-prolog-stack-pointer:

Stack Pointer
+++++++++++++

If the kernel has function calls it must set up the ABI stack pointer described
in :ref:`amdgpu-amdhsa-function-call-convention-non-kernel-functions` by setting
SGPR32 to the unswizzled scratch offset of the address past the last local
allocation.

.. _amdgpu-amdhsa-kernel-prolog-frame-pointer:

Frame Pointer
+++++++++++++

If the kernel needs a frame pointer for the reasons defined in
``SIFrameLowering`` then SGPR33 is used and is always set to ``0`` in the
kernel prolog. If a frame pointer is not required then all uses of the frame
pointer are replaced with immediate ``0`` offsets.

.. _amdgpu-amdhsa-kernel-prolog-flat-scratch:

Flat Scratch
++++++++++++

If the kernel or any function it calls may use flat operations to access
scratch memory, the prolog code must set up the FLAT_SCRATCH register pair
(FLAT_SCRATCH_LO/FLAT_SCRATCH_HI which are in SGPRn-4/SGPRn-3). Initialization
uses Flat Scratch Init and Scratch Wavefront Offset SGPR registers (see
:ref:`amdgpu-amdhsa-initial-kernel-execution-state`):

GFX6
  Flat scratch is not supported.

GFX7-GFX8

  1. The low word of Flat Scratch Init is 32-bit byte offset from
     ``SH_HIDDEN_PRIVATE_BASE_VIMID`` to the base of scratch backing memory
     being managed by SPI for the queue executing the kernel dispatch. This is
     the same value used in the Scratch Segment Buffer V# base address. The
     prolog must add the value of Scratch Wavefront Offset to get the
     wavefront's byte scratch backing memory offset from
     ``SH_HIDDEN_PRIVATE_BASE_VIMID``. Since FLAT_SCRATCH_LO is in units of 256
     bytes, the offset must be right shifted by 8 before moving into
     FLAT_SCRATCH_LO.
  2. The second word of Flat Scratch Init is 32-bit byte size of a single
     work-items scratch memory usage. This is directly loaded from the kernel
     dispatch packet Private Segment Byte Size and rounded up to a multiple of
     DWORD. Having CP load it once avoids loading it at the beginning of every
     wavefront. The prolog must move it to FLAT_SCRATCH_LO for use as FLAT
     SCRATCH SIZE.

GFX9-GFX10
  The Flat Scratch Init is the 64-bit address of the base of scratch backing
  memory being managed by SPI for the queue executing the kernel dispatch. The
  prolog must add the value of Scratch Wavefront Offset and moved to the
  FLAT_SCRATCH pair for use as the flat scratch base in flat memory
  instructions.

.. _amdgpu-amdhsa-kernel-prolog-private-segment-buffer:

Private Segment Buffer
++++++++++++++++++++++

A set of four SGPRs beginning at a four-aligned SGPR index are always selected
to serve as the scratch V# for the kernel as follows:

  - If it is known during instruction selection that there is stack usage,
    SGPR0-3 is reserved for use as the scratch V#.  Stack usage is assumed if
    optimizations are disabled (``-O0``), if stack objects already exist (for
    locals, etc.), or if there are any function calls.

  - Otherwise, four high numbered SGPRs beginning at a four-aligned SGPR index
    are reserved for the tentative scratch V#. These will be used if it is
    determined that spilling is needed.

    - If no use is made of the tentative scratch V#, then it is unreserved,
      and the register count is determined ignoring it.
    - If use is made of the tentative scratch V#, then its register numbers
      are shifted to the first four-aligned SGPR index after the highest one
      allocated by the register allocator, and all uses are updated. The
      register count includes them in the shifted location.
    - In either case, if the processor has the SGPR allocation bug, the
      tentative allocation is not shifted or unreserved in order to ensure
      the register count is higher to workaround the bug.

    .. note::

      This approach of using a tentative scratch V# and shifting the register
      numbers if used avoids having to perform register allocation a second
      time if the tentative V# is eliminated. This is more efficient and
      avoids the problem that the second register allocation may perform
      spilling which will fail as there is no longer a scratch V#.

When the kernel prolog code is being emitted it is known whether the scratch V#
described above is actually used. If it is, the prolog code must set it up by
copying the Private Segment Buffer to the scratch V# registers and then adding
the Private Segment Wavefront Offset to the queue base address in the V#. The
result is a V# with a base address pointing to the beginning of the wavefront
scratch backing memory.

The Private Segment Buffer is always requested, but the Private Segment
Wavefront Offset is only requested if it is used (see
:ref:`amdgpu-amdhsa-initial-kernel-execution-state`).

.. _amdgpu-amdhsa-memory-model:

Memory Model
~~~~~~~~~~~~

This section describes the mapping of LLVM memory model onto AMDGPU machine code
(see :ref:`memmodel`).

The AMDGPU backend supports the memory synchronization scopes specified in
:ref:`amdgpu-memory-scopes`.

The code sequences used to implement the memory model are defined in table
:ref:`amdgpu-amdhsa-memory-model-code-sequences-gfx6-gfx10-table`.

The sequences specify the order of instructions that a single thread must
execute. The ``s_waitcnt`` and ``buffer_wbinvl1_vol`` are defined with respect
to other memory instructions executed by the same thread. This allows them to be
moved earlier or later which can allow them to be combined with other instances
of the same instruction, or hoisted/sunk out of loops to improve
performance. Only the instructions related to the memory model are given;
additional ``s_waitcnt`` instructions are required to ensure registers are
defined before being used. These may be able to be combined with the memory
model ``s_waitcnt`` instructions as described above.

The AMDGPU backend supports the following memory models:

  HSA Memory Model [HSA]_
    The HSA memory model uses a single happens-before relation for all address
    spaces (see :ref:`amdgpu-address-spaces`).
  OpenCL Memory Model [OpenCL]_
    The OpenCL memory model which has separate happens-before relations for the
    global and local address spaces. Only a fence specifying both global and
    local address space, and seq_cst instructions join the relationships. Since
    the LLVM ``memfence`` instruction does not allow an address space to be
    specified the OpenCL fence has to conservatively assume both local and
    global address space was specified. However, optimizations can often be
    done to eliminate the additional ``s_waitcnt`` instructions when there are
    no intervening memory instructions which access the corresponding address
    space. The code sequences in the table indicate what can be omitted for the
    OpenCL memory. The target triple environment is used to determine if the
    source language is OpenCL (see :ref:`amdgpu-opencl`).

``ds/flat_load/store/atomic`` instructions to local memory are termed LDS
operations.

``buffer/global/flat_load/store/atomic`` instructions to global memory are
termed vector memory operations.

For GFX6-GFX9:

* Each agent has multiple shader arrays (SA).
* Each SA has multiple compute units (CU).
* Each CU has multiple SIMDs that execute wavefronts.
* The wavefronts for a single work-group are executed in the same CU but may be
  executed by different SIMDs.
* Each CU has a single LDS memory shared by the wavefronts of the work-groups
  executing on it.
* All LDS operations of a CU are performed as wavefront wide operations in a
  global order and involve no caching. Completion is reported to a wavefront in
  execution order.
* The LDS memory has multiple request queues shared by the SIMDs of a
  CU. Therefore, the LDS operations performed by different wavefronts of a
  work-group can be reordered relative to each other, which can result in
  reordering the visibility of vector memory operations with respect to LDS
  operations of other wavefronts in the same work-group. A ``s_waitcnt
  lgkmcnt(0)`` is required to ensure synchronization between LDS operations and
  vector memory operations between wavefronts of a work-group, but not between
  operations performed by the same wavefront.
* The vector memory operations are performed as wavefront wide operations and
  completion is reported to a wavefront in execution order. The exception is
  that for GFX7-GFX9 ``flat_load/store/atomic`` instructions can report out of
  vector memory order if they access LDS memory, and out of LDS operation order
  if they access global memory.
* The vector memory operations access a single vector L1 cache shared by all
  SIMDs a CU. Therefore, no special action is required for coherence between the
  lanes of a single wavefront, or for coherence between wavefronts in the same
  work-group. A ``buffer_wbinvl1_vol`` is required for coherence between
  wavefronts executing in different work-groups as they may be executing on
  different CUs.
* The scalar memory operations access a scalar L1 cache shared by all wavefronts
  on a group of CUs. The scalar and vector L1 caches are not coherent. However,
  scalar operations are used in a restricted way so do not impact the memory
  model. See :ref:`amdgpu-address-spaces`.
* The vector and scalar memory operations use an L2 cache shared by all CUs on
  the same agent.
* The L2 cache has independent channels to service disjoint ranges of virtual
  addresses.
* Each CU has a separate request queue per channel. Therefore, the vector and
  scalar memory operations performed by wavefronts executing in different
  work-groups (which may be executing on different CUs) of an agent can be
  reordered relative to each other. A ``s_waitcnt vmcnt(0)`` is required to
  ensure synchronization between vector memory operations of different CUs. It
  ensures a previous vector memory operation has completed before executing a
  subsequent vector memory or LDS operation and so can be used to meet the
  requirements of acquire and release.
* The L2 cache can be kept coherent with other agents on some targets, or ranges
  of virtual addresses can be set up to bypass it to ensure system coherence.

For GFX10:

* Each agent has multiple shader arrays (SA).
* Each SA has multiple work-group processors (WGP).
* Each WGP has multiple compute units (CU).
* Each CU has multiple SIMDs that execute wavefronts.
* The wavefronts for a single work-group are executed in the same
  WGP. In CU wavefront execution mode the wavefronts may be executed by
  different SIMDs in the same CU. In WGP wavefront execution mode the
  wavefronts may be executed by different SIMDs in different CUs in the same
  WGP.
* Each WGP has a single LDS memory shared by the wavefronts of the work-groups
  executing on it.
* All LDS operations of a WGP are performed as wavefront wide operations in a
  global order and involve no caching. Completion is reported to a wavefront in
  execution order.
* The LDS memory has multiple request queues shared by the SIMDs of a
  WGP. Therefore, the LDS operations performed by different wavefronts of a
  work-group can be reordered relative to each other, which can result in
  reordering the visibility of vector memory operations with respect to LDS
  operations of other wavefronts in the same work-group. A ``s_waitcnt
  lgkmcnt(0)`` is required to ensure synchronization between LDS operations and
  vector memory operations between wavefronts of a work-group, but not between
  operations performed by the same wavefront.
* The vector memory operations are performed as wavefront wide operations.
  Completion of load/store/sample operations are reported to a wavefront in
  execution order of other load/store/sample operations performed by that
  wavefront.
* The vector memory operations access a vector L0 cache. There is a single L0
  cache per CU. Each SIMD of a CU accesses the same L0 cache. Therefore, no
  special action is required for coherence between the lanes of a single
  wavefront. However, a ``BUFFER_GL0_INV`` is required for coherence between
  wavefronts executing in the same work-group as they may be executing on SIMDs
  of different CUs that access different L0s. A ``BUFFER_GL0_INV`` is also
  required for coherence between wavefronts executing in different work-groups
  as they may be executing on different WGPs.
* The scalar memory operations access a scalar L0 cache shared by all wavefronts
  on a WGP. The scalar and vector L0 caches are not coherent. However, scalar
  operations are used in a restricted way so do not impact the memory model. See
  :ref:`amdgpu-address-spaces`.
* The vector and scalar memory L0 caches use an L1 cache shared by all WGPs on
  the same SA. Therefore, no special action is required for coherence between
  the wavefronts of a single work-group. However, a ``BUFFER_GL1_INV`` is
  required for coherence between wavefronts executing in different work-groups
  as they may be executing on different SAs that access different L1s.
* The L1 caches have independent quadrants to service disjoint ranges of virtual
  addresses.
* Each L0 cache has a separate request queue per L1 quadrant. Therefore, the
  vector and scalar memory operations performed by different wavefronts, whether
  executing in the same or different work-groups (which may be executing on
  different CUs accessing different L0s), can be reordered relative to each
  other. A ``s_waitcnt vmcnt(0) & vscnt(0)`` is required to ensure
  synchronization between vector memory operations of different wavefronts. It
  ensures a previous vector memory operation has completed before executing a
  subsequent vector memory or LDS operation and so can be used to meet the
  requirements of acquire, release and sequential consistency.
* The L1 caches use an L2 cache shared by all SAs on the same agent.
* The L2 cache has independent channels to service disjoint ranges of virtual
  addresses.
* Each L1 quadrant of a single SA accesses a different L2 channel. Each L1
  quadrant has a separate request queue per L2 channel. Therefore, the vector
  and scalar memory operations performed by wavefronts executing in different
  work-groups (which may be executing on different SAs) of an agent can be
  reordered relative to each other. A ``s_waitcnt vmcnt(0) & vscnt(0)`` is
  required to ensure synchronization between vector memory operations of
  different SAs. It ensures a previous vector memory operation has completed
  before executing a subsequent vector memory and so can be used to meet the
  requirements of acquire, release and sequential consistency.
* The L2 cache can be kept coherent with other agents on some targets, or ranges
  of virtual addresses can be set up to bypass it to ensure system coherence.

Private address space uses ``buffer_load/store`` using the scratch V#
(GFX6-GFX8), or ``scratch_load/store`` (GFX9-GFX10). Since only a single thread
is accessing the memory, atomic memory orderings are not meaningful, and all
accesses are treated as non-atomic.

Constant address space uses ``buffer/global_load`` instructions (or equivalent
scalar memory instructions). Since the constant address space contents do not
change during the execution of a kernel dispatch it is not legal to perform
stores, and atomic memory orderings are not meaningful, and all access are
treated as non-atomic.

A memory synchronization scope wider than work-group is not meaningful for the
group (LDS) address space and is treated as work-group.

The memory model does not support the region address space which is treated as
non-atomic.

Acquire memory ordering is not meaningful on store atomic instructions and is
treated as non-atomic.

Release memory ordering is not meaningful on load atomic instructions and is
treated a non-atomic.

Acquire-release memory ordering is not meaningful on load or store atomic
instructions and is treated as acquire and release respectively.

AMDGPU backend only uses scalar memory operations to access memory that is
proven to not change during the execution of the kernel dispatch. This includes
constant address space and global address space for program scope const
variables. Therefore, the kernel machine code does not have to maintain the
scalar L1 cache to ensure it is coherent with the vector L1 cache. The scalar
and vector L1 caches are invalidated between kernel dispatches by CP since
constant address space data may change between kernel dispatch executions. See
:ref:`amdgpu-address-spaces`.

The one exception is if scalar writes are used to spill SGPR registers. In this
case the AMDGPU backend ensures the memory location used to spill is never
accessed by vector memory operations at the same time. If scalar writes are used
then a ``s_dcache_wb`` is inserted before the ``s_endpgm`` and before a function
return since the locations may be used for vector memory instructions by a
future wavefront that uses the same scratch area, or a function call that
creates a frame at the same address, respectively. There is no need for a
``s_dcache_inv`` as all scalar writes are write-before-read in the same thread.

For GFX6-GFX9, scratch backing memory (which is used for the private address
space) is accessed with MTYPE NC_NV (non-coherent non-volatile). Since the
private address space is only accessed by a single thread, and is always
write-before-read, there is never a need to invalidate these entries from the L1
cache. Hence all cache invalidates are done as ``*_vol`` to only invalidate the
volatile cache lines.

For GFX10, scratch backing memory (which is used for the private address space)
is accessed with MTYPE NC (non-coherent). Since the private address space is
only accessed by a single thread, and is always write-before-read, there is
never a need to invalidate these entries from the L0 or L1 caches.

For GFX10, wavefronts are executed in native mode with in-order reporting of
loads and sample instructions. In this mode vmcnt reports completion of load,
atomic with return and sample instructions in order, and the vscnt reports the
completion of store and atomic without return in order. See ``MEM_ORDERED``
field in :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.

In GFX10, wavefronts can be executed in WGP or CU wavefront execution mode:

* In WGP wavefront execution mode the wavefronts of a work-group are executed
  on the SIMDs of both CUs of the WGP. Therefore, explicit management of the per
  CU L0 caches is required for work-group synchronization. Also accesses to L1
  at work-group scope need to be explicitly ordered as the accesses from
  different CUs are not ordered.
* In CU wavefront execution mode the wavefronts of a work-group are executed on
  the SIMDs of a single CU of the WGP. Therefore, all global memory access by
  the work-group access the same L0 which in turn ensures L1 accesses are
  ordered and so do not require explicit management of the caches for
  work-group synchronization.

See ``WGP_MODE`` field in
:ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table` and
:ref:`amdgpu-target-features`.

On dGPU the kernarg backing memory is accessed as UC (uncached) to avoid needing
to invalidate the L2 cache. For GFX6-GFX9, this also causes it to be treated as
non-volatile and so is not invalidated by ``*_vol``. On APU it is accessed as CC
(cache coherent) and so the L2 cache will be coherent with the CPU and other
agents.

  .. table:: AMDHSA Memory Model Code Sequences GFX6-GFX10
     :name: amdgpu-amdhsa-memory-model-code-sequences-gfx6-gfx10-table

     ============ ============ ============== ========== =============================== ==================================
     LLVM Instr   LLVM Memory  LLVM Memory    AMDGPU     AMDGPU Machine Code             AMDGPU Machine Code
                  Ordering     Sync Scope     Address    GFX6-9                          GFX10
                                              Space
     ============ ============ ============== ========== =============================== ==================================
     **Non-Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     load         *none*       *none*         - global   - !volatile & !nontemporal      - !volatile & !nontemporal
                                              - generic
                                              - private    1. buffer/global/flat_load      1. buffer/global/flat_load
                                              - constant
                                                         - volatile & !nontemporal       - volatile & !nontemporal

                                                           1. buffer/global/flat_load      1. buffer/global/flat_load
                                                              glc=1                           glc=1 dlc=1

                                                         - nontemporal                   - nontemporal

                                                           1. buffer/global/flat_load      1. buffer/global/flat_load
                                                              glc=1 slc=1                     slc=1

     load         *none*       *none*         - local    1. ds_load                      1. ds_load
     store        *none*       *none*         - global   - !nontemporal                  - !nontemporal
                                              - generic
                                              - private    1. buffer/global/flat_store     1. buffer/global/flat_store
                                              - constant
                                                         - nontemporal                   - nontemporal

                                                           1. buffer/global/flat_store      1. buffer/global/flat_store
                                                              glc=1 slc=1                      slc=1

     store        *none*       *none*         - local    1. ds_store                     1. ds_store
     **Unordered Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     load atomic  unordered    *any*          *any*      *Same as non-atomic*.           *Same as non-atomic*.
     store atomic unordered    *any*          *any*      *Same as non-atomic*.           *Same as non-atomic*.
     atomicrmw    unordered    *any*          *any*      *Same as monotonic              *Same as monotonic
                                                         atomic*.                        atomic*.
     **Monotonic Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     load atomic  monotonic    - singlethread - global   1. buffer/global/flat_load      1. buffer/global/flat_load
                               - wavefront    - generic
     load atomic  monotonic    - workgroup    - global   1. buffer/global/flat_load      1. buffer/global/flat_load
                                              - generic                                     glc=1

                                                                                           - If CU wavefront execution mode, omit glc=1.

     load atomic  monotonic    - singlethread - local    1. ds_load                      1. ds_load
                               - wavefront
                               - workgroup
     load atomic  monotonic    - agent        - global   1. buffer/global/flat_load      1. buffer/global/flat_load
                               - system       - generic     glc=1                           glc=1 dlc=1
     store atomic monotonic    - singlethread - global   1. buffer/global/flat_store     1. buffer/global/flat_store
                               - wavefront    - generic
                               - workgroup
                               - agent
                               - system
     store atomic monotonic    - singlethread - local    1. ds_store                     1. ds_store
                               - wavefront
                               - workgroup
     atomicrmw    monotonic    - singlethread - global   1. buffer/global/flat_atomic    1. buffer/global/flat_atomic
                               - wavefront    - generic
                               - workgroup
                               - agent
                               - system
     atomicrmw    monotonic    - singlethread - local    1. ds_atomic                    1. ds_atomic
                               - wavefront
                               - workgroup
     **Acquire Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     load atomic  acquire      - singlethread - global   1. buffer/global/ds/flat_load   1. buffer/global/ds/flat_load
                               - wavefront    - local
                                              - generic
     load atomic  acquire      - workgroup    - global   1. buffer/global/flat_load      1. buffer/global_load glc=1

                                                                                           - If CU wavefront execution mode, omit glc=1.

                                                                                         2. s_waitcnt vmcnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Must happen before
                                                                                             the following buffer_gl0_inv
                                                                                             and before any following
                                                                                             global/generic
                                                                                             load/load
                                                                                             atomic/store/store
                                                                                             atomic/atomicrmw.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     load atomic  acquire      - workgroup    - local    1. ds_load                      1. ds_load
                                                         2. s_waitcnt lgkmcnt(0)         2. s_waitcnt lgkmcnt(0)

                                                           - If OpenCL, omit.              - If OpenCL, omit.
                                                           - Must happen before            - Must happen before
                                                             any following                   the following buffer_gl0_inv
                                                             global/generic                  and before any following
                                                             load/load                       global/generic load/load
                                                             atomic/store/store              atomic/store/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the load             older than the load
                                                             atomic value being              atomic value being
                                                             acquired.                       acquired.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - If OpenCL, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     load atomic  acquire      - workgroup    - generic  1. flat_load                    1. flat_load glc=1

                                                                                           - If CU wavefront execution mode, omit glc=1.

                                                         2. s_waitcnt lgkmcnt(0)         2. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             any following                   the following
                                                             global/generic                  buffer_gl0_inv and any
                                                             load/load                       following global/generic
                                                             atomic/store/store              load/load
                                                             atomic/atomicrmw.               atomic/store/store
                                                                                             atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the load             older than the load
                                                             atomic value being              atomic value being
                                                             acquired.                       acquired.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     load atomic  acquire      - agent        - global   1. buffer/global/flat_load      1. buffer/global_load
                               - system                     glc=1                           glc=1 dlc=1
                                                         2. s_waitcnt vmcnt(0)           2. s_waitcnt vmcnt(0)

                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures the load              - Ensures the load
                                                             has completed                   has completed
                                                             before invalidating             before invalidating
                                                             the cache.                      the caches.

                                                         3. buffer_wbinvl1_vol           3. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following                       following
                                                             loads will not see              loads will not see
                                                             stale global data.              stale global data.

     load atomic  acquire      - agent        - generic  1. flat_load glc=1              1. flat_load glc=1 dlc=1
                               - system                  2. s_waitcnt vmcnt(0) &         2. s_waitcnt vmcnt(0) &
                                                            lgkmcnt(0)                      lgkmcnt(0)

                                                           - If OpenCL omit                - If OpenCL omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_invl.
                                                           - Ensures the flat_load         - Ensures the flat_load
                                                             has completed                   has completed
                                                             before invalidating             before invalidating
                                                             the cache.                      the caches.

                                                         3. buffer_wbinvl1_vol           3. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     atomicrmw    acquire      - singlethread - global   1. buffer/global/ds/flat_atomic 1. buffer/global/ds/flat_atomic
                               - wavefront    - local
                                              - generic
     atomicrmw    acquire      - workgroup    - global   1. buffer/global/flat_atomic    1. buffer/global_atomic
                                                                                         2. s_waitcnt vm/vscnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                                                           - Must happen before
                                                                                             the following buffer_gl0_inv
                                                                                             and before any following
                                                                                             global/generic
                                                                                             load/load
                                                                                             atomic/store/store
                                                                                             atomic/atomicrmw.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acquire      - workgroup    - local    1. ds_atomic                    1. ds_atomic
                                                         2. waitcnt lgkmcnt(0)           2. waitcnt lgkmcnt(0)

                                                           - If OpenCL, omit.              - If OpenCL, omit.
                                                           - Must happen before            - Must happen before
                                                             any following                   the following
                                                             global/generic                  buffer_gl0_inv.
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the                  older than the
                                                             atomicrmw value                 atomicrmw value
                                                             being acquired.                 being acquired.

                                                                                         3. buffer_gl0_inv

                                                                                           - If OpenCL omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acquire      - workgroup    - generic  1. flat_atomic                  1. flat_atomic
                                                         2. waitcnt lgkmcnt(0)           2. waitcnt lgkmcnt(0) &
                                                                                            vm/vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vm/vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             waitcnt lgkmcnt(0)..
                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             any following                   the following
                                                             global/generic                  buffer_gl0_inv.
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the                  older than the
                                                             atomicrmw value                 atomicrmw value
                                                             being acquired.                 being acquired.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acquire      - agent        - global   1. buffer/global/flat_atomic    1. buffer/global_atomic
                               - system                  2. s_waitcnt vmcnt(0)           2. s_waitcnt vm/vscnt(0)

                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures the                   - Ensures the
                                                             atomicrmw has                   atomicrmw has
                                                             completed before                completed before
                                                             invalidating the                invalidating the
                                                             cache.                          caches.

                                                         3. buffer_wbinvl1_vol           3. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     atomicrmw    acquire      - agent        - generic  1. flat_atomic                  1. flat_atomic
                               - system                  2. s_waitcnt vmcnt(0) &         2. s_waitcnt vm/vscnt(0) &
                                                            lgkmcnt(0)                      lgkmcnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures the                   - Ensures the
                                                             atomicrmw has                   atomicrmw has
                                                             completed before                completed before
                                                             invalidating the                invalidating the
                                                             cache.                          caches.

                                                         3. buffer_wbinvl1_vol           3. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     fence        acquire      - singlethread *none*     *none*                          *none*
                               - wavefront
     fence        acquire      - workgroup    *none*     1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit.              not generic, omit
                                                                                             lgkmcnt(0).
                                                                                           - If OpenCL and
                                                                                             address space is
                                                                                             local, omit
                                                                                             vmcnt(0) and vscnt(0).
                                                           - However, since LLVM           - However, since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate. If             always generate. If
                                                             fence had an                    fence had an
                                                             address space then              address space then
                                                             set to address                  set to address
                                                             space of OpenCL                 space of OpenCL
                                                             fence flag, or to               fence flag, or to
                                                             generic if both                 generic if both
                                                             local and global                local and global
                                                             flags are                       flags are
                                                             specified.                      specified.
                                                           - Must happen after
                                                             any preceding
                                                             local/generic load
                                                             atomic/atomicrmw
                                                             with an equal or
                                                             wider sync scope
                                                             and memory ordering
                                                             stronger than
                                                             unordered (this is
                                                             termed the
                                                             fence-paired-atomic).
                                                           - Must happen before
                                                             any following
                                                             global/generic
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                           - Ensures any
                                                             following global
                                                             data read is no
                                                             older than the
                                                             value read by the
                                                             fence-paired-atomic.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             atomicrmw-no-return-value
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load
                                                                                             atomic/atomicrmw
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - Must happen before
                                                                                             the following
                                                                                             buffer_gl0_inv.
                                                                                           - Ensures that the
                                                                                             fence-paired atomic
                                                                                             has completed
                                                                                             before invalidating
                                                                                             the
                                                                                             cache. Therefore
                                                                                             any following
                                                                                             locations read must
                                                                                             be no older than
                                                                                             the value read by
                                                                                             the
                                                                                             fence-paired-atomic.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     fence        acquire      - agent        *none*     1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system                     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit               not generic, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                                                           - If OpenCL and
                                                                                             address space is
                                                                                             local, omit
                                                                                             vmcnt(0) and vscnt(0).
                                                           - However, since LLVM           - However, since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate                 always generate
                                                             (see comment for                (see comment for
                                                             previous fence).                previous fence).
                                                           - Could be split into
                                                             separate s_waitcnt
                                                             vmcnt(0) and
                                                             s_waitcnt
                                                             lgkmcnt(0) to allow
                                                             them to be
                                                             independently moved
                                                             according to the
                                                             following rules.
                                                           - s_waitcnt vmcnt(0)
                                                             must happen after
                                                             any preceding
                                                             global/generic load
                                                             atomic/atomicrmw
                                                             with an equal or
                                                             wider sync scope
                                                             and memory ordering
                                                             stronger than
                                                             unordered (this is
                                                             termed the
                                                             fence-paired-atomic).
                                                           - s_waitcnt lgkmcnt(0)
                                                             must happen after
                                                             any preceding
                                                             local/generic load
                                                             atomic/atomicrmw
                                                             with an equal or
                                                             wider sync scope
                                                             and memory ordering
                                                             stronger than
                                                             unordered (this is
                                                             termed the
                                                             fence-paired-atomic).
                                                           - Must happen before
                                                             the following
                                                             buffer_wbinvl1_vol.
                                                           - Ensures that the
                                                             fence-paired atomic
                                                             has completed
                                                             before invalidating
                                                             the
                                                             cache. Therefore
                                                             any following
                                                             locations read must
                                                             be no older than
                                                             the value read by
                                                             the
                                                             fence-paired-atomic.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             atomicrmw-no-return-value
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load
                                                                                             atomic/atomicrmw
                                                                                             with an equal or
                                                                                             wider sync scope
                                                                                             and memory ordering
                                                                                             stronger than
                                                                                             unordered (this is
                                                                                             termed the
                                                                                             fence-paired-atomic).
                                                                                           - Must happen before
                                                                                             the following
                                                                                             buffer_gl*_inv.
                                                                                           - Ensures that the
                                                                                             fence-paired atomic
                                                                                             has completed
                                                                                             before invalidating
                                                                                             the
                                                                                             caches. Therefore
                                                                                             any following
                                                                                             locations read must
                                                                                             be no older than
                                                                                             the value read by
                                                                                             the
                                                                                             fence-paired-atomic.

                                                         2. buffer_wbinvl1_vol           2. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before any        - Must happen before any
                                                             following global/generic        following global/generic
                                                             load/load                       load/load
                                                             atomic/store/store              atomic/store/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     **Release Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     store atomic release      - singlethread - global   1. buffer/global/ds/flat_store  1. buffer/global/ds/flat_store
                               - wavefront    - local
                                              - generic
     store atomic release      - workgroup    - global   1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             lgkmcnt(0).
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/store/load
                                                             atomic/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic
                                                                                             load/store/load
                                                                                             atomic/store
                                                                                             atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             store.                          store.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             store that is being             store that is being
                                                             released.                       released.

                                                         2. buffer/global/flat_store     2. buffer/global_store
     store atomic release      - workgroup    - local                                    1. waitcnt vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - If OpenCL, omit.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0) and s_waitcnt
                                                                                             vscnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - Must happen before
                                                                                             the following
                                                                                             store.
                                                                                           - Ensures that all
                                                                                             global memory
                                                                                             operations have
                                                                                             completed before
                                                                                             performing the
                                                                                             store that is being
                                                                                             released.

                                                         1. ds_store                     2. ds_store
     store atomic release      - workgroup    - generic  1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             lgkmcnt(0).
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/store/load
                                                             atomic/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load/store/load
                                                                                             atomic/store atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             store.                          store.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             store that is being             store that is being
                                                             released.                       released.

                                                         2. flat_store                   2. flat_store
     store atomic release      - agent        - global   1. s_waitcnt lgkmcnt(0) &         1. s_waitcnt lgkmcnt(0) &
                               - system       - generic     vmcnt(0)                          vmcnt(0) & vscnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt vscnt(0)
                                                             s_waitcnt                       and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load
                                                             atomic/store                    atomic/
                                                             atomic/atomicrmw.               atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             store.                          store.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to memory have                  to memory have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             store that is being             store that is being
                                                             released.                       released.

                                                         2. buffer/global/ds/flat_store  2. buffer/global/ds/flat_store
     atomicrmw    release      - singlethread - global   1. buffer/global/ds/flat_atomic 1. buffer/global/ds/flat_atomic
                               - wavefront    - local
                                              - generic
     atomicrmw    release      - workgroup    - global   1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.

                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/store/load
                                                             atomic/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic
                                                                                             load/store/load
                                                                                             atomic/store
                                                                                             atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. buffer/global/flat_atomic    2. buffer/global_atomic
     atomicrmw    release      - workgroup    - local                                    1. waitcnt vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - If OpenCL, omit.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0) and s_waitcnt
                                                                                             vscnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - Must happen before
                                                                                             the following
                                                                                             store.
                                                                                           - Ensures that all
                                                                                             global memory
                                                                                             operations have
                                                                                             completed before
                                                                                             performing the
                                                                                             store that is being
                                                                                             released.

                                                         1. ds_atomic                    2. ds_atomic
     atomicrmw    release      - workgroup    - generic  1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/store/load
                                                             atomic/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load/store/load
                                                                                             atomic/store atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. flat_atomic                  2. flat_atomic
     atomicrmw    release      - agent        - global   1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lkkmcnt(0) &
                               - system       - generic     vmcnt(0)                         vmcnt(0) & vscnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt
                                                             s_waitcnt                       vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load atomic/
                                                             atomic/store                    atomicrmw-with-return-value.
                                                             atomic/atomicrmw.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to global and local             to global and local
                                                             have completed                  have completed
                                                             before performing               before performing
                                                             the atomicrmw that              the atomicrmw that
                                                             is being released.              is being released.

                                                         2. buffer/global/ds/flat_atomic 2. buffer/global/ds/flat_atomic
     fence        release      - singlethread *none*     *none*                          *none*
                               - wavefront
     fence        release      - workgroup    *none*     1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit.              not generic, omit
                                                                                             lgkmcnt(0).
                                                                                           - If OpenCL and
                                                                                             address space is
                                                                                             local, omit
                                                                                             vmcnt(0) and vscnt(0).
                                                           - However, since LLVM           - However, since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate. If             always generate. If
                                                             fence had an                    fence had an
                                                             address space then              address space then
                                                             set to address                  set to address
                                                             space of OpenCL                 space of OpenCL
                                                             fence flag, or to               fence flag, or to
                                                             generic if both                 generic if both
                                                             local and global                local and global
                                                             flags are                       flags are
                                                             specified.                      specified.
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic
                                                                                             load/store/load
                                                                                             atomic/store atomic/
                                                                                             atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             any following store             any following store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             fence-paired-atomic).           fence-paired-atomic).
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             following                       following
                                                             fence-paired-atomic.            fence-paired-atomic.

     fence        release      - agent        *none*     1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system                     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit               not generic, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             local, omit                     local, omit
                                                             vmcnt(0).                       vmcnt(0) and vscnt(0).
                                                           - However, since LLVM           - However, since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate. If             always generate. If
                                                             fence had an                    fence had an
                                                             address space then              address space then
                                                             set to address                  set to address
                                                             space of OpenCL                 space of OpenCL
                                                             fence flag, or to               fence flag, or to
                                                             generic if both                 generic if both
                                                             local and global                local and global
                                                             flags are                       flags are
                                                             specified.                      specified.
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt
                                                             s_waitcnt                       vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load atomic/
                                                             atomic/store                    atomicrmw-with-return-value.
                                                             atomic/atomicrmw.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             any following store             any following store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             fence-paired-atomic).           fence-paired-atomic).
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             have                            have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             following                       following
                                                             fence-paired-atomic.            fence-paired-atomic.

     **Acquire-Release Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     atomicrmw    acq_rel      - singlethread - global   1. buffer/global/ds/flat_atomic 1. buffer/global/ds/flat_atomic
                               - wavefront    - local
                                              - generic
     atomicrmw    acq_rel      - workgroup    - global   1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             s_waitcnt lgkmcnt(0).
                                                           - Must happen after             - Must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load/store/load
                                                                                             atomic/store atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. buffer/global/flat_atomic    2. buffer/global_atomic
                                                                                         3. s_waitcnt vm/vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vm/vscnt.
                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                                                             waitcnt lgkmcnt(0).
                                                                                           - Must happen before
                                                                                             the following
                                                                                             buffer_gl0_inv.
                                                                                           - Ensures any
                                                                                             following global
                                                                                             data read is no
                                                                                             older than the
                                                                                             atomicrmw value
                                                                                             being acquired.

                                                                                         4. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acq_rel      - workgroup    - local                                    1. waitcnt vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - If OpenCL, omit.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0) and s_waitcnt
                                                                                             vscnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - Must happen before
                                                                                             the following
                                                                                             store.
                                                                                           - Ensures that all
                                                                                             global memory
                                                                                             operations have
                                                                                             completed before
                                                                                             performing the
                                                                                             store that is being
                                                                                             released.

                                                         1. ds_atomic                    2. ds_atomic
                                                         2. s_waitcnt lgkmcnt(0)         3. s_waitcnt lgkmcnt(0)

                                                           - If OpenCL, omit.              - If OpenCL, omit.
                                                           - Must happen before            - Must happen before
                                                             any following                   the following
                                                             global/generic                  buffer_gl0_inv.
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the load             older than the load
                                                             atomic value being              atomic value being
                                                             acquired.                       acquired.

                                                                                         4. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - If OpenCL omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acq_rel      - workgroup    - generic  1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/store/load
                                                             atomic/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic load/store/load
                                                                                             atomic/store atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. flat_atomic                  2. flat_atomic
                                                         3. s_waitcnt lgkmcnt(0)         3. s_waitcnt lgkmcnt(0) &
                                                                                            vm/vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vm/vscnt.
                                                           - If OpenCL, omit.              - If OpenCL, omit
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             any following                   the following
                                                             global/generic                  buffer_gl0_inv.
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                           - Ensures any                   - Ensures any
                                                             following global                following global
                                                             data read is no                 data read is no
                                                             older than the load             older than the load
                                                             atomic value being              atomic value being
                                                             acquired.                       acquired.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     atomicrmw    acq_rel      - agent        - global   1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system                     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt
                                                             s_waitcnt                       vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load atomic/
                                                             atomic/store                    atomicrmw-with-return-value.
                                                             atomic/atomicrmw.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to global have                  to global have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. buffer/global/flat_atomic    2. buffer/global_atomic
                                                         3. s_waitcnt vmcnt(0)           3. s_waitcnt vm/vscnt(0)

                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                                                             waitcnt lgkmcnt(0).
                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures the                   - Ensures the
                                                             atomicrmw has                   atomicrmw has
                                                             completed before                completed before
                                                             invalidating the                invalidating the
                                                             cache.                          caches.

                                                         4. buffer_wbinvl1_vol           4. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     atomicrmw    acq_rel      - agent        - generic  1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system                     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt
                                                             s_waitcnt                       vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load atomic
                                                             atomic/store                    atomicrmw-with-return-value.
                                                             atomic/atomicrmw.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             atomicrmw.                      atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to global have                  have
                                                             completed before                completed before
                                                             performing the                  performing the
                                                             atomicrmw that is               atomicrmw that is
                                                             being released.                 being released.

                                                         2. flat_atomic                  2. flat_atomic
                                                         3. s_waitcnt vmcnt(0) &         3. s_waitcnt vm/vscnt(0) &
                                                            lgkmcnt(0)                      lgkmcnt(0)

                                                           - If OpenCL, omit               - If OpenCL, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                                                           - Use vmcnt if atomic with
                                                                                             return and vscnt if atomic
                                                                                             with no-return.
                                                           - Must happen before            - Must happen before
                                                             following                       following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures the                   - Ensures the
                                                             atomicrmw has                   atomicrmw has
                                                             completed before                completed before
                                                             invalidating the                invalidating the
                                                             cache.                          caches.

                                                         4. buffer_wbinvl1_vol           4. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data.                    global data.

     fence        acq_rel      - singlethread *none*     *none*                          *none*
                               - wavefront
     fence        acq_rel      - workgroup    *none*     1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                                                                            vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit.              not generic, omit
                                                                                             lgkmcnt(0).
                                                                                           - If OpenCL and
                                                                                             address space is
                                                                                             local, omit
                                                                                             vmcnt(0) and vscnt(0).
                                                           - However,                      - However,
                                                             since LLVM                      since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate                 always generate
                                                             (see comment for                (see comment for
                                                             previous fence).                previous fence).
                                                           - Must happen after
                                                             any preceding
                                                             local/generic
                                                             load/load
                                                             atomic/store/store
                                                             atomic/atomicrmw.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - s_waitcnt vmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             load/load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                                                           - s_waitcnt lgkmcnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             local/generic
                                                                                             load/store/load
                                                                                             atomic/store atomic/
                                                                                             atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/store/store              atomic/store/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that all              - Ensures that all
                                                             memory operations               memory operations
                                                             to local have                   have
                                                             completed before                completed before
                                                             performing any                  performing any
                                                             following global                following global
                                                             memory operations.              memory operations.
                                                           - Ensures that the              - Ensures that the
                                                             preceding                       preceding
                                                             local/generic load              local/generic load
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             acquire-fence-paired-atomic     acquire-fence-paired-atomic
                                                             ) has completed                 ) has completed
                                                             before following                before following
                                                             global memory                   global memory
                                                             operations. This                operations. This
                                                             satisfies the                   satisfies the
                                                             requirements of                 requirements of
                                                             acquire.                        acquire.
                                                           - Ensures that all              - Ensures that all
                                                             previous memory                 previous memory
                                                             operations have                 operations have
                                                             completed before a              completed before a
                                                             following                       following
                                                             local/generic store             local/generic store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             release-fence-paired-atomic     release-fence-paired-atomic
                                                             ). This satisfies the           ). This satisfies the
                                                             requirements of                 requirements of
                                                             release.                        release.
                                                                                           - Must happen before
                                                                                             the following
                                                                                             buffer_gl0_inv.
                                                                                           - Ensures that the
                                                                                             acquire-fence-paired
                                                                                             atomic has completed
                                                                                             before invalidating
                                                                                             the
                                                                                             cache. Therefore
                                                                                             any following
                                                                                             locations read must
                                                                                             be no older than
                                                                                             the value read by
                                                                                             the
                                                                                             acquire-fence-paired-atomic.

                                                                                         3. buffer_gl0_inv

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Ensures that
                                                                                             following
                                                                                             loads will not see
                                                                                             stale data.

     fence        acq_rel      - agent        *none*     1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system                     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - If OpenCL and                 - If OpenCL and
                                                             address space is                address space is
                                                             not generic, omit               not generic, omit
                                                             lgkmcnt(0).                     lgkmcnt(0).
                                                                                           - If OpenCL and
                                                                                             address space is
                                                                                             local, omit
                                                                                             vmcnt(0) and vscnt(0).
                                                           - However, since LLVM           - However, since LLVM
                                                             currently has no                currently has no
                                                             address space on                address space on
                                                             the fence need to               the fence need to
                                                             conservatively                  conservatively
                                                             always generate                 always generate
                                                             (see comment for                (see comment for
                                                             previous fence).                previous fence).
                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0) and                    vmcnt(0), s_waitcnt
                                                             s_waitcnt                       vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - s_waitcnt vmcnt(0)            - s_waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             global/generic                  global/generic
                                                             load/store/load                 load/load
                                                             atomic/store                    atomic/
                                                             atomic/atomicrmw.               atomicrmw-with-return-value.
                                                                                           - s_waitcnt vscnt(0)
                                                                                             must happen after
                                                                                             any preceding
                                                                                             global/generic
                                                                                             store/store atomic/
                                                                                             atomicrmw-no-return-value.
                                                           - s_waitcnt lgkmcnt(0)          - s_waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             any preceding                   any preceding
                                                             local/generic                   local/generic
                                                             load/store/load                 load/store/load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Must happen before            - Must happen before
                                                             the following                   the following
                                                             buffer_wbinvl1_vol.             buffer_gl*_inv.
                                                           - Ensures that the              - Ensures that the
                                                             preceding                       preceding
                                                             global/local/generic            global/local/generic
                                                             load                            load
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             acquire-fence-paired-atomic     acquire-fence-paired-atomic
                                                             ) has completed                 ) has completed
                                                             before invalidating             before invalidating
                                                             the cache. This                 the caches. This
                                                             satisfies the                   satisfies the
                                                             requirements of                 requirements of
                                                             acquire.                        acquire.
                                                           - Ensures that all              - Ensures that all
                                                             previous memory                 previous memory
                                                             operations have                 operations have
                                                             completed before a              completed before a
                                                             following                       following
                                                             global/local/generic            global/local/generic
                                                             store                           store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with an equal or                with an equal or
                                                             wider sync scope                wider sync scope
                                                             and memory ordering             and memory ordering
                                                             stronger than                   stronger than
                                                             unordered (this is              unordered (this is
                                                             termed the                      termed the
                                                             release-fence-paired-atomic     release-fence-paired-atomic
                                                             ). This satisfies the           ). This satisfies the
                                                             requirements of                 requirements of
                                                             release.                        release.

                                                         2. buffer_wbinvl1_vol           2. buffer_gl0_inv;
                                                                                            buffer_gl1_inv

                                                           - Must happen before            - Must happen before
                                                             any following                   any following
                                                             global/generic                  global/generic
                                                             load/load                       load/load
                                                             atomic/store/store              atomic/store/store
                                                             atomic/atomicrmw.               atomic/atomicrmw.
                                                           - Ensures that                  - Ensures that
                                                             following loads                 following loads
                                                             will not see stale              will not see stale
                                                             global data. This               global data. This
                                                             satisfies the                   satisfies the
                                                             requirements of                 requirements of
                                                             acquire.                        acquire.

     **Sequential Consistent Atomic**
     ----------------------------------------------------------------------------------------------------------------------
     load atomic  seq_cst      - singlethread - global   *Same as corresponding          *Same as corresponding
                               - wavefront    - local    load atomic acquire,            load atomic acquire,
                                              - generic  except must generated           except must generated
                                                         all instructions even           all instructions even
                                                         for OpenCL.*                    for OpenCL.*
     load atomic  seq_cst      - workgroup    - global   1. s_waitcnt lgkmcnt(0)         1. s_waitcnt lgkmcnt(0) &
                                              - generic                                     vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit vmcnt and
                                                                                             vscnt.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0), s_waitcnt
                                                                                             vscnt(0) and s_waitcnt
                                                                                             lgkmcnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                           - Must                          - waitcnt lgkmcnt(0) must
                                                             happen after                    happen after
                                                             preceding                       preceding
                                                             global/generic load             local load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with memory                     with memory
                                                             ordering of seq_cst             ordering of seq_cst
                                                             and with equal or               and with equal or
                                                             wider sync scope.               wider sync scope.
                                                             (Note that seq_cst              (Note that seq_cst
                                                             fences have their               fences have their
                                                             own s_waitcnt                   own s_waitcnt
                                                             lgkmcnt(0) and so do            lgkmcnt(0) and so do
                                                             not need to be                  not need to be
                                                             considered.)                    considered.)
                                                                                           - waitcnt vmcnt(0)
                                                                                             Must happen after
                                                                                             preceding
                                                                                             global/generic load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value
                                                                                             with memory
                                                                                             ordering of seq_cst
                                                                                             and with equal or
                                                                                             wider sync scope.
                                                                                             (Note that seq_cst
                                                                                             fences have their
                                                                                             own s_waitcnt
                                                                                             vmcnt(0) and so do
                                                                                             not need to be
                                                                                             considered.)
                                                                                           - waitcnt vscnt(0)
                                                                                             Must happen after
                                                                                             preceding
                                                                                             global/generic store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value
                                                                                             with memory
                                                                                             ordering of seq_cst
                                                                                             and with equal or
                                                                                             wider sync scope.
                                                                                             (Note that seq_cst
                                                                                             fences have their
                                                                                             own s_waitcnt
                                                                                             vscnt(0) and so do
                                                                                             not need to be
                                                                                             considered.)
                                                           - Ensures any                   - Ensures any
                                                             preceding                       preceding
                                                             sequential                      sequential
                                                             consistent local                consistent global/local
                                                             memory instructions             memory instructions
                                                             have completed                  have completed
                                                             before executing                before executing
                                                             this sequentially               this sequentially
                                                             consistent                      consistent
                                                             instruction. This               instruction. This
                                                             prevents reordering             prevents reordering
                                                             a seq_cst store                 a seq_cst store
                                                             followed by a                   followed by a
                                                             seq_cst load. (Note             seq_cst load. (Note
                                                             that seq_cst is                 that seq_cst is
                                                             stronger than                   stronger than
                                                             acquire/release as              acquire/release as
                                                             the reordering of               the reordering of
                                                             load acquire                    load acquire
                                                             followed by a store             followed by a store
                                                             release is                      release is
                                                             prevented by the                prevented by the
                                                             waitcnt of                      waitcnt of
                                                             the release, but                the release, but
                                                             there is nothing                there is nothing
                                                             preventing a store              preventing a store
                                                             release followed by             release followed by
                                                             load acquire from               load acquire from
                                                             competing out of                competing out of
                                                             order.)                         order.)

                                                         2. *Following                   2. *Following
                                                            instructions same as            instructions same as
                                                            corresponding load              corresponding load
                                                            atomic acquire,                 atomic acquire,
                                                            except must generated           except must generated
                                                            all instructions even           all instructions even
                                                            for OpenCL.*                    for OpenCL.*
     load atomic  seq_cst      - workgroup    - local    *Same as corresponding
                                                         load atomic acquire,
                                                         except must generated
                                                         all instructions even
                                                         for OpenCL.*

                                                                                         1. s_waitcnt vmcnt(0) & vscnt(0)

                                                                                           - If CU wavefront execution mode, omit.
                                                                                           - Could be split into
                                                                                             separate s_waitcnt
                                                                                             vmcnt(0) and s_waitcnt
                                                                                             vscnt(0) to allow
                                                                                             them to be
                                                                                             independently moved
                                                                                             according to the
                                                                                             following rules.
                                                                                           - waitcnt vmcnt(0)
                                                                                             Must happen after
                                                                                             preceding
                                                                                             global/generic load
                                                                                             atomic/
                                                                                             atomicrmw-with-return-value
                                                                                             with memory
                                                                                             ordering of seq_cst
                                                                                             and with equal or
                                                                                             wider sync scope.
                                                                                             (Note that seq_cst
                                                                                             fences have their
                                                                                             own s_waitcnt
                                                                                             vmcnt(0) and so do
                                                                                             not need to be
                                                                                             considered.)
                                                                                           - waitcnt vscnt(0)
                                                                                             Must happen after
                                                                                             preceding
                                                                                             global/generic store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value
                                                                                             with memory
                                                                                             ordering of seq_cst
                                                                                             and with equal or
                                                                                             wider sync scope.
                                                                                             (Note that seq_cst
                                                                                             fences have their
                                                                                             own s_waitcnt
                                                                                             vscnt(0) and so do
                                                                                             not need to be
                                                                                             considered.)
                                                                                           - Ensures any
                                                                                             preceding
                                                                                             sequential
                                                                                             consistent global
                                                                                             memory instructions
                                                                                             have completed
                                                                                             before executing
                                                                                             this sequentially
                                                                                             consistent
                                                                                             instruction. This
                                                                                             prevents reordering
                                                                                             a seq_cst store
                                                                                             followed by a
                                                                                             seq_cst load. (Note
                                                                                             that seq_cst is
                                                                                             stronger than
                                                                                             acquire/release as
                                                                                             the reordering of
                                                                                             load acquire
                                                                                             followed by a store
                                                                                             release is
                                                                                             prevented by the
                                                                                             waitcnt of
                                                                                             the release, but
                                                                                             there is nothing
                                                                                             preventing a store
                                                                                             release followed by
                                                                                             load acquire from
                                                                                             competing out of
                                                                                             order.)

                                                                                         2. *Following
                                                                                            instructions same as
                                                                                            corresponding load
                                                                                            atomic acquire,
                                                                                            except must generated
                                                                                            all instructions even
                                                                                            for OpenCL.*

     load atomic  seq_cst      - agent        - global   1. s_waitcnt lgkmcnt(0) &       1. s_waitcnt lgkmcnt(0) &
                               - system       - generic     vmcnt(0)                        vmcnt(0) & vscnt(0)

                                                           - Could be split into           - Could be split into
                                                             separate s_waitcnt              separate s_waitcnt
                                                             vmcnt(0)                        vmcnt(0), s_waitcnt
                                                             and s_waitcnt                   vscnt(0) and s_waitcnt
                                                             lgkmcnt(0) to allow             lgkmcnt(0) to allow
                                                             them to be                      them to be
                                                             independently moved             independently moved
                                                             according to the                according to the
                                                             following rules.                following rules.
                                                           - waitcnt lgkmcnt(0)            - waitcnt lgkmcnt(0)
                                                             must happen after               must happen after
                                                             preceding                       preceding
                                                             global/generic load             local load
                                                             atomic/store                    atomic/store
                                                             atomic/atomicrmw                atomic/atomicrmw
                                                             with memory                     with memory
                                                             ordering of seq_cst             ordering of seq_cst
                                                             and with equal or               and with equal or
                                                             wider sync scope.               wider sync scope.
                                                             (Note that seq_cst              (Note that seq_cst
                                                             fences have their               fences have their
                                                             own s_waitcnt                   own s_waitcnt
                                                             lgkmcnt(0) and so do            lgkmcnt(0) and so do
                                                             not need to be                  not need to be
                                                             considered.)                    considered.)
                                                           - waitcnt vmcnt(0)              - waitcnt vmcnt(0)
                                                             must happen after               must happen after
                                                             preceding                       preceding
                                                             global/generic load             global/generic load
                                                             atomic/store                    atomic/
                                                             atomic/atomicrmw                atomicrmw-with-return-value
                                                             with memory                     with memory
                                                             ordering of seq_cst             ordering of seq_cst
                                                             and with equal or               and with equal or
                                                             wider sync scope.               wider sync scope.
                                                             (Note that seq_cst              (Note that seq_cst
                                                             fences have their               fences have their
                                                             own s_waitcnt                   own s_waitcnt
                                                             vmcnt(0) and so do              vmcnt(0) and so do
                                                             not need to be                  not need to be
                                                             considered.)                    considered.)
                                                                                           - waitcnt vscnt(0)
                                                                                             Must happen after
                                                                                             preceding
                                                                                             global/generic store
                                                                                             atomic/
                                                                                             atomicrmw-no-return-value
                                                                                             with memory
                                                                                             ordering of seq_cst
                                                                                             and with equal or
                                                                                             wider sync scope.
                                                                                             (Note that seq_cst
                                                                                             fences have their
                                                                                             own s_waitcnt
                                                                                             vscnt(0) and so do
                                                                                             not need to be
                                                                                             considered.)
                                                           - Ensures any                   - Ensures any
                                                             preceding                       preceding
                                                             sequential                      sequential
                                                             consistent global               consistent global
                                                             memory instructions             memory instructions
                                                             have completed                  have completed
                                                             before executing                before executing
                                                             this sequentially               this sequentially
                                                             consistent                      consistent
                                                             instruction. This               instruction. This
                                                             prevents reordering             prevents reordering
                                                             a seq_cst store                 a seq_cst store
                                                             followed by a                   followed by a
                                                             seq_cst load. (Note             seq_cst load. (Note
                                                             that seq_cst is                 that seq_cst is
                                                             stronger than                   stronger than
                                                             acquire/release as              acquire/release as
                                                             the reordering of               the reordering of
                                                             load acquire                    load acquire
                                                             followed by a store             followed by a store
                                                             release is                      release is
                                                             prevented by the                prevented by the
                                                             waitcnt of                      waitcnt of
                                                             the release, but                the release, but
                                                             there is nothing                there is nothing
                                                             preventing a store              preventing a store
                                                             release followed by             release followed by
                                                             load acquire from               load acquire from
                                                             competing out of                competing out of
                                                             order.)                         order.)

                                                         2. *Following                   2. *Following
                                                            instructions same as            instructions same as
                                                            corresponding load              corresponding load
                                                            atomic acquire,                 atomic acquire,
                                                            except must generated           except must generated
                                                            all instructions even           all instructions even
                                                            for OpenCL.*                    for OpenCL.*
     store atomic seq_cst      - singlethread - global   *Same as corresponding          *Same as corresponding
                               - wavefront    - local    store atomic release,           store atomic release,
                               - workgroup    - generic  except must generated           except must generated
                                                         all instructions even           all instructions even
                                                         for OpenCL.*                    for OpenCL.*
     store atomic seq_cst      - agent        - global   *Same as corresponding          *Same as corresponding
                               - system       - generic  store atomic release,           store atomic release,
                                                         except must generated           except must generated
                                                         all instructions even           all instructions even
                                                         for OpenCL.*                    for OpenCL.*
     atomicrmw    seq_cst      - singlethread - global   *Same as corresponding          *Same as corresponding
                               - wavefront    - local    atomicrmw acq_rel,              atomicrmw acq_rel,
                               - workgroup    - generic  except must generated           except must generated
                                                         all instructions even           all instructions even
                                                         for OpenCL.*                    for OpenCL.*
     atomicrmw    seq_cst      - agent        - global   *Same as corresponding          *Same as corresponding
                               - system       - generic  atomicrmw acq_rel,              atomicrmw acq_rel,
                                                         except must generated           except must generated
                                                         all instructions even           all instructions even
                                                         for OpenCL.*                    for OpenCL.*
     fence        seq_cst      - singlethread *none*     *Same as corresponding          *Same as corresponding
                               - wavefront               fence acq_rel,                  fence acq_rel,
                               - workgroup               except must generated           except must generated
                               - agent                   all instructions even           all instructions even
                               - system                  for OpenCL.*                    for OpenCL.*
     ============ ============ ============== ========== =============================== ==================================

The memory order also adds the single thread optimization constrains defined in
table
:ref:`amdgpu-amdhsa-memory-model-single-thread-optimization-constraints-gfx6-gfx10-table`.

  .. table:: AMDHSA Memory Model Single Thread Optimization Constraints GFX6-GFX10
     :name: amdgpu-amdhsa-memory-model-single-thread-optimization-constraints-gfx6-gfx10-table

     ============ ==============================================================
     LLVM Memory  Optimization Constraints
     Ordering
     ============ ==============================================================
     unordered    *none*
     monotonic    *none*
     acquire      - If a load atomic/atomicrmw then no following load/load
                    atomic/store/ store atomic/atomicrmw/fence instruction can
                    be moved before the acquire.
                  - If a fence then same as load atomic, plus no preceding
                    associated fence-paired-atomic can be moved after the fence.
     release      - If a store atomic/atomicrmw then no preceding load/load
                    atomic/store/ store atomic/atomicrmw/fence instruction can
                    be moved after the release.
                  - If a fence then same as store atomic, plus no following
                    associated fence-paired-atomic can be moved before the
                    fence.
     acq_rel      Same constraints as both acquire and release.
     seq_cst      - If a load atomic then same constraints as acquire, plus no
                    preceding sequentially consistent load atomic/store
                    atomic/atomicrmw/fence instruction can be moved after the
                    seq_cst.
                  - If a store atomic then the same constraints as release, plus
                    no following sequentially consistent load atomic/store
                    atomic/atomicrmw/fence instruction can be moved before the
                    seq_cst.
                  - If an atomicrmw/fence then same constraints as acq_rel.
     ============ ==============================================================

Trap Handler ABI
~~~~~~~~~~~~~~~~

For code objects generated by AMDGPU backend for HSA [HSA]_ compatible runtimes
(such as ROCm [AMD-ROCm]_), the runtime installs a trap handler that supports
the ``s_trap`` instruction with the following usage:

  .. table:: AMDGPU Trap Handler for AMDHSA OS
     :name: amdgpu-trap-handler-for-amdhsa-os-table

     =================== =============== =============== =======================
     Usage               Code Sequence   Trap Handler    Description
                                         Inputs
     =================== =============== =============== =======================
     reserved            ``s_trap 0x00``                 Reserved by hardware.
     ``debugtrap(arg)``  ``s_trap 0x01`` ``SGPR0-1``:    Reserved for HSA
                                           ``queue_ptr`` ``debugtrap``
                                         ``VGPR0``:      intrinsic (not
                                           ``arg``       implemented).
     ``llvm.trap``       ``s_trap 0x02`` ``SGPR0-1``:    Causes dispatch to be
                                           ``queue_ptr`` terminated and its
                                                         associated queue put
                                                         into the error state.
     ``llvm.debugtrap``  ``s_trap 0x03``                 - If debugger not
                                                           installed then
                                                           behaves as a
                                                           no-operation. The
                                                           trap handler is
                                                           entered and
                                                           immediately returns
                                                           to continue
                                                           execution of the
                                                           wavefront.
                                                         - If the debugger is
                                                           installed, causes
                                                           the debug trap to be
                                                           reported by the
                                                           debugger and the
                                                           wavefront is put in
                                                           the halt state until
                                                           resumed by the
                                                           debugger.
     reserved            ``s_trap 0x04``                 Reserved.
     reserved            ``s_trap 0x05``                 Reserved.
     reserved            ``s_trap 0x06``                 Reserved.
     debugger breakpoint ``s_trap 0x07``                 Reserved for debugger
                                                         breakpoints.
     reserved            ``s_trap 0x08``                 Reserved.
     reserved            ``s_trap 0xfe``                 Reserved.
     reserved            ``s_trap 0xff``                 Reserved.
     =================== =============== =============== =======================

.. _amdgpu-amdhsa-function-call-convention:

Call Convention
~~~~~~~~~~~~~~~

.. note::

  This section is currently incomplete and has inakkuracies. It is WIP that will
  be updated as information is determined.

See :ref:`amdgpu-dwarf-address-space-identifier` for information on swizzled
addresses. Unswizzled addresses are normal linear addresses.

.. _amdgpu-amdhsa-function-call-convention-kernel-functions:

Kernel Functions
++++++++++++++++

This section describes the call convention ABI for the outer kernel function.

See :ref:`amdgpu-amdhsa-initial-kernel-execution-state` for the kernel call
convention.

The following is not part of the AMDGPU kernel calling convention but describes
how the AMDGPU implements function calls:

1.  Clang decides the kernarg layout to match the *HSA Programmer's Language
    Reference* [HSA]_.

    - All structs are passed directly.
    - Lambda values are passed *TBA*.

    .. TODO::

      - Does this really follow HSA rules? Or are structs >16 bytes passed
        by-value struct?
      - What is ABI for lambda values?

4.  The kernel performs certain setup in its prolog, as described in
    :ref:`amdgpu-amdhsa-kernel-prolog`.

.. _amdgpu-amdhsa-function-call-convention-non-kernel-functions:

Non-Kernel Functions
++++++++++++++++++++

This section describes the call convention ABI for functions other than the
outer kernel function.

If a kernel has function calls then scratch is always allocated and used for
the call stack which grows from low address to high address using the swizzled
scratch address space.

On entry to a function:

1.  SGPR0-3 contain a V# with the following properties (see
    :ref:`amdgpu-amdhsa-kernel-prolog-private-segment-buffer`):

    * Base address pointing to the beginning of the wavefront scratch backing
      memory.
    * Swizzled with dword element size and stride of wavefront size elements.

2.  The FLAT_SCRATCH register pair is setup. See
    :ref:`amdgpu-amdhsa-kernel-prolog-flat-scratch`.
3.  GFX6-8: M0 register set to the size of LDS in bytes. See
    :ref:`amdgpu-amdhsa-kernel-prolog-m0`.
4.  The EXEC register is set to the lanes active on entry to the function.
5.  MODE register: *TBD*
6.  VGPR0-31 and SGPR4-29 are used to pass function input arguments as described
    below.
7.  SGPR30-31 return address (RA). The code address that the function must
    return to when it completes. The value is undefined if the function is *no
    return*.
8.  SGPR32 is used for the stack pointer (SP). It is an unswizzled scratch
    offset relative to the beginning of the wavefront scratch backing memory.

    The unswizzled SP can be used with buffer instructions as an unswizzled SGPR
    offset with the scratch V# in SGPR0-3 to access the stack in a swizzled
    manner.

    The unswizzled SP value can be converted into the swizzled SP value by:

      | swizzled SP = unswizzled SP / wavefront size

    This may be used to obtain the private address space address of stack
    objects and to convert this address to a flat address by adding the flat
    scratch aperture base address.

    The swizzled SP value is always 4 bytes aligned for the ``r600``
    architecture and 16 byte aligned for the ``amdgcn`` architecture.

    .. note::

      The ``amdgcn`` value is selected to avoid dynamic stack alignment for the
      OpenCL language which has the largest base type defined as 16 bytes.

    On entry, the swizzled SP value is the address of the first function
    argument passed on the stack. Other stack passed arguments are positive
    offsets from the entry swizzled SP value.

    The function may use positive offsets beyond the last stack passed argument
    for stack allocated local variables and register spill slots. If necessary,
    the function may align these to greater alignment than 16 bytes. After these
    the function may dynamically allocate space for such things as runtime sized
    ``alloca`` local allocations.

    If the function calls another function, it will place any stack allocated
    arguments after the last local allocation and adjust SGPR32 to the address
    after the last local allocation.

9.  All other registers are unspecified.
10. Any necessary ``waitcnt`` has been performed to ensure memory is available
    to the function.

On exit from a function:

1.  VGPR0-31 and SGPR4-29 are used to pass function result arguments as
    described below. Any registers used are considered clobbered registers.
2.  The following registers are preserved and have the same value as on entry:

    * FLAT_SCRATCH
    * EXEC
    * GFX6-8: M0
    * All SGPR and VGPR registers except the clobbered registers of SGPR4-31 and
      VGPR0-31.

      For the AMDGPU backend, an inter-procedural register allocation (IPRA)
      optimization may mark some of clobbered SGPR4-31 and VGPR0-31 registers as
      preserved if it can be determined that the called function does not change
      their value.

2.  The PC is set to the RA provided on entry.
3.  MODE register: *TBD*.
4.  All other registers are clobbered.
5.  Any necessary ``waitcnt`` has been performed to ensure memory accessed by
    function is available to the caller.

.. TODO::

  - On gfx908 are all ACC registers clobbered?

  - How are function results returned? The address of structured types is passed
    by reference, but what about other types?

The function input arguments are made up of the formal arguments explicitly
declared by the source language function plus the implicit input arguments used
by the implementation.

The source language input arguments are:

1. Any source language implicit ``this`` or ``self`` argument comes first as a
   pointer type.
2. Followed by the function formal arguments in left to right source order.

The source language result arguments are:

1. The function result argument.

The source language input or result struct type arguments that are less than or
equal to 16 bytes, are decomposed recursively into their base type fields, and
each field is passed as if a separate argument. For input arguments, if the
called function requires the struct to be in memory, for example because its
address is taken, then the function body is responsible for allocating a stack
location and copying the field arguments into it. Clang terms this *direct
struct*.

The source language input struct type arguments that are greater than 16 bytes,
are passed by reference. The caller is responsible for allocating a stack
location to make a copy of the struct value and pass the address as the input
argument. The called function is responsible to perform the dereference when
accessing the input argument. Clang terms this *by-value struct*.

A source language result struct type argument that is greater than 16 bytes, is
returned by reference. The caller is responsible for allocating a stack location
to hold the result value and passes the address as the last input argument
(before the implicit input arguments). In this case there are no result
arguments. The called function is responsible to perform the dereference when
storing the result value. Clang terms this *structured return (sret)*.

*TODO: correct the ``sret`` definition.*

.. TODO::

  Is this definition correct? Or is ``sret`` only used if passing in registers, and
  pass as non-decomposed struct as stack argument? Or something else? Is the
  memory location in the caller stack frame, or a stack memory argument and so
  no address is passed as the caller can directly write to the argument stack
  location? But then the stack location is still live after return. If an
  argument stack location is it the first stack argument or the last one?

Lambda argument types are treated as struct types with an implementation defined
set of fields.

.. TODO::

  Need to specify the ABI for lambda types for AMDGPU.

For AMDGPU backend all source language arguments (including the decomposed
struct type arguments) are passed in VGPRs unless marked ``inreg`` in which case
they are passed in SGPRs.

The AMDGPU backend walks the function call graph from the leaves to determine
which implicit input arguments are used, propagating to each caller of the
function. The used implicit arguments are appended to the function arguments
after the source language arguments in the following order:

.. TODO::

  Is recursion or external functions supported?

1.  Work-Item ID (1 VGPR)

    The X, Y and Z work-item ID are packed into a single VGRP with the following
    layout. Only fields actually used by the function are set. The other bits
    are undefined.

    The values come from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-vgpr-register-set-up-order-table`.

    .. table:: Work-item implict argument layout
      :name: amdgpu-amdhsa-workitem-implict-argument-layout-table

      ======= ======= ==============
      Bits    Size    Field Name
      ======= ======= ==============
      9:0     10 bits X Work-Item ID
      19:10   10 bits Y Work-Item ID
      29:20   10 bits Z Work-Item ID
      31:30   2 bits  Unused
      ======= ======= ==============

2.  Dispatch Ptr (2 SGPRs)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

3.  Queue Ptr (2 SGPRs)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

4.  Kernarg Segment Ptr (2 SGPRs)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

5.  Dispatch id (2 SGPRs)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

6.  Work-Group ID X (1 SGPR)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

7.  Work-Group ID Y (1 SGPR)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

8.  Work-Group ID Z (1 SGPR)

    The value comes from the initial kernel execution state. See
    :ref:`amdgpu-amdhsa-sgpr-register-set-up-order-table`.

9.  Implicit Argument Ptr (2 SGPRs)

    The value is computed by adding an offset to Kernarg Segment Ptr to get the
    global address space pointer to the first kernarg implicit argument.

The input and result arguments are assigned in order in the following manner:

..note::

  There are likely some errors and omissions in the following description that
  need correction.

  ..TODO::

    Check the clang source code to decipher how function arguments and return
    results are handled. Also see the AMDGPU specific values used.

* VGPR arguments are assigned to consecutive VGPRs starting at VGPR0 up to
  VGPR31.

  If there are more arguments than will fit in these registers, the remaining
  arguments are allocated on the stack in order on naturally aligned
  addresses.

  .. TODO::

    How are overly aligned structures allocated on the stack?

* SGPR arguments are assigned to consecutive SGPRs starting at SGPR0 up to
  SGPR29.

  If there are more arguments than will fit in these registers, the remaining
  arguments are allocated on the stack in order on naturally aligned
  addresses.

Note that decomposed struct type arguments may have some fields passed in
registers and some in memory.

.. TODO::

  So, a struct which can pass some fields as decomposed register arguments, will
  pass the rest as decomposed stack elements? But an argument that will not start
  in registers will not be decomposed and will be passed as a non-decomposed
  stack value?

The following is not part of the AMDGPU function calling convention but
describes how the AMDGPU implements function calls:

1.  SGPR33 is used as a frame pointer (FP) if necessary. Like the SP it is an
    unswizzled scratch address. It is only needed if runtime sized ``alloca``
    are used, or for the reasons defined in ``SIFrameLowering``.
2.  Runtime stack alignment is not currently supported.

    .. TODO::

      - If runtime stack alignment is supported, then will an extra argument
        pointer register be used?

2.  Allocating SGPR arguments on the stack are not supported.

3.  No CFI is currently generated. See
    :ref:`amdgpu-dwarf-call-frame-information`.

    ..note::

      CFI will be generated that defines the CFA as the unswizzled address
      relative to the wave scratch base in the unswizzled private address space
      of the lowest address stack allocated local variable.

      ``DW_AT_frame_base`` will be defined as the swizzled address in the
      swizzled private address space by dividing the CFA by the wavefront size
      (since CFA is always at least dword aligned which matches the scratch
      swizzle element size).

      If no dynamic stack alignment was performed, the stack allocated arguments
      are accessed as negative offsets relative to ``DW_AT_frame_base``, and the
      local variables and register spill slots are accessed as positive offsets
      relative to ``DW_AT_frame_base``.

4.  Function argument passing is implemented by copying the input physical
    registers to virtual registers on entry. The register allocator can spill if
    necessary. These are copied back to physical registers at call sites. The
    net effect is that each function call can have these values in entirely
    distinct locations. The IPRA can help avoid shuffling argument registers.
5.  Call sites are implemented by setting up the arguments at positive offsets
    from SP. Then SP is incremented to account for the known frame size before
    the call and decremented after the call.

    ..note::

      The CFI will reflect the changed calculation needed to compute the CFA
      from SP.

6.  4 byte spill slots are used in the stack frame. One slot is allocated for an
    emergency spill slot. Buffer instructions are used for stack accesses and
    not the ``flat_scratch`` instruction.

    ..TODO::

      Explain when the emergency spill slot is used.

.. TODO::

  Possible broken issues:

  - Stack arguments must be aligned to required alignment.
  - Stack is aligned to max(16, max formal argument alignment)
  - Direct argument < 64 bits should check register budget.
  - Register budget calculation should respect ``inreg`` for SGPR.
  - SGPR overflow is not handled.
  - struct with 1 member unpeeling is not checking size of member.
  - ``sret`` is after ``this`` pointer.
  - Caller is not implementing stack realignment: need an extra pointer.
  - Should say AMDGPU passes FP rather than SP.
  - Should CFI define CFA as address of locals or arguments. Difference is
    apparent when have implemented dynamic alignment.
  - If ``SCRATCH`` instruction could allow negative offsets, then can make FP be
    highest address of stack frame and use negative offset for locals. Would
    allow SP to be the same as FP and could support signal-handler-like as now
    have a real SP for the top of the stack.
  - How is ``sret`` passed on the stack? In argument stack area? Can it overlay
    arguments?

AMDPAL
------

This section provides code conventions used when the target triple OS is
``amdpal`` (see :ref:`amdgpu-target-triples`) for passing runtime parameters
from the application/runtime to each invocation of a hardware shader. These
parameters include both generic, application-controlled parameters called
*user data* as well as system-generated parameters that are a product of the
draw or dispatch execution.

User Data
~~~~~~~~~

Each hardware stage has a set of 32-bit *user data registers* which can be
written from a command buffer and then loaded into SGPRs when waves are launched
via a subsequent dispatch or draw operation. This is the way most arguments are
passed from the application/runtime to a hardware shader.

Compute User Data
~~~~~~~~~~~~~~~~~

Compute shader user data mappings are simpler than graphics shaders and have a
fixed mapping.

Note that there are always 10 available *user data entries* in registers -
entries beyond that limit must be fetched from memory (via the spill table
pointer) by the shader.

  .. table:: PAL Compute Shader User Data Registers
     :name: pal-compute-user-data-registers

     ============= ================================
     User Register Description
     ============= ================================
     0             Global Internal Table (32-bit pointer)
     1             Per-Shader Internal Table (32-bit pointer)
     2 - 11        Application-Controlled User Data (10 32-bit values)
     12            Spill Table (32-bit pointer)
     13 - 14       Thread Group Count (64-bit pointer)
     15            GDS Range
     ============= ================================

Graphics User Data
~~~~~~~~~~~~~~~~~~

Graphics pipelines support a much more flexible user data mapping:

  .. table:: PAL Graphics Shader User Data Registers
     :name: pal-graphics-user-data-registers

     ============= ================================
     User Register Description
     ============= ================================
     0             Global Internal Table (32-bit pointer)
     +             Per-Shader Internal Table (32-bit pointer)
     + 1-15        Application Controlled User Data
                   (1-15 Contiguous 32-bit Values in Registers)
     +             Spill Table (32-bit pointer)
     +             Draw Index (First Stage Only)
     +             Vertex Offset (First Stage Only)
     +             Instance Offset (First Stage Only)
     ============= ================================

  The placement of the global internal table remains fixed in the first *user
  data SGPR register*. Otherwise all parameters are optional, and can be mapped
  to any desired *user data SGPR register*, with the following restrictions:

  * Draw Index, Vertex Offset, and Instance Offset can only be used by the first
    active hardware stage in a graphics pipeline (i.e. where the API vertex
    shader runs).

  * Application-controlled user data must be mapped into a contiguous range of
    user data registers.

  * The application-controlled user data range supports compaction remapping, so
    only *entries* that are actually consumed by the shader must be assigned to
    corresponding *registers*. Note that in order to support an efficient runtime
    implementation, the remapping must pack *registers* in the same order as
    *entries*, with unused *entries* removed.

.. _pal_global_internal_table:

Global Internal Table
~~~~~~~~~~~~~~~~~~~~~

The global internal table is a table of *shader resource descriptors* (SRDs)
that define how certain engine-wide, runtime-managed resources should be
accessed from a shader. The majority of these resources have HW-defined formats,
and it is up to the compiler to write/read data as required by the target
hardware.

The following table illustrates the required format:

  .. table:: PAL Global Internal Table
     :name: pal-git-table

     ============= ================================
     Offset        Description
     ============= ================================
     0-3           Graphics Scratch SRD
     4-7           Compute Scratch SRD
     8-11          ES/GS Ring Output SRD
     12-15         ES/GS Ring Input SRD
     16-19         GS/VS Ring Output #0
     20-23         GS/VS Ring Output #1
     24-27         GS/VS Ring Output #2
     28-31         GS/VS Ring Output #3
     32-35         GS/VS Ring Input SRD
     36-39         Tessellation Factor Buffer SRD
     40-43         Off-Chip LDS Buffer SRD
     44-47         Off-Chip Param Cache Buffer SRD
     48-51         Sample Position Buffer SRD
     52            vaRange::ShadowDescriptorTable High Bits
     ============= ================================

  The pointer to the global internal table passed to the shader as user data
  is a 32-bit pointer. The top 32 bits should be assumed to be the same as
  the top 32 bits of the pipeline, so the shader may use the program
  counter's top 32 bits.

Unspecified OS
--------------

This section provides code conventions used when the target triple OS is
empty (see :ref:`amdgpu-target-triples`).

Trap Handler ABI
~~~~~~~~~~~~~~~~

For code objects generated by AMDGPU backend for non-amdhsa OS, the runtime does
not install a trap handler. The ``llvm.trap`` and ``llvm.debugtrap``
instructions are handled as follows:

  .. table:: AMDGPU Trap Handler for Non-AMDHSA OS
     :name: amdgpu-trap-handler-for-non-amdhsa-os-table

     =============== =============== ===========================================
     Usage           Code Sequence   Description
     =============== =============== ===========================================
     llvm.trap       s_endpgm        Causes wavefront to be terminated.
     llvm.debugtrap  *none*          Compiler warning given that there is no
                                     trap handler installed.
     =============== =============== ===========================================

Source Languages
================

.. _amdgpu-opencl:

OpenCL
------

When the language is OpenCL the following differences occur:

1. The OpenCL memory model is used (see :ref:`amdgpu-amdhsa-memory-model`).
2. The AMDGPU backend appends additional arguments to the kernel's explicit
   arguments for the AMDHSA OS (see
   :ref:`opencl-kernel-implicit-arguments-appended-for-amdhsa-os-table`).
3. Additional metadata is generated
   (see :ref:`amdgpu-amdhsa-code-object-metadata`).

  .. table:: OpenCL kernel implicit arguments appended for AMDHSA OS
     :name: opencl-kernel-implicit-arguments-appended-for-amdhsa-os-table

     ======== ==== ========= ===========================================
     Position Byte Byte      Description
              Size Alignment
     ======== ==== ========= ===========================================
     1        8    8         OpenCL Global Offset X
     2        8    8         OpenCL Global Offset Y
     3        8    8         OpenCL Global Offset Z
     4        8    8         OpenCL address of printf buffer
     5        8    8         OpenCL address of virtual queue used by
                             enqueue_kernel.
     6        8    8         OpenCL address of AqlWrap struct used by
                             enqueue_kernel.
     7        8    8         Pointer argument used for Multi-gird
                             synchronization.
     ======== ==== ========= ===========================================

.. _amdgpu-hcc:

HCC
---

When the language is HCC the following differences occur:

1. The HSA memory model is used (see :ref:`amdgpu-amdhsa-memory-model`).

.. _amdgpu-assembler:

Assembler
---------

AMDGPU backend has LLVM-MC based assembler which is currently in development.
It supports AMDGCN GFX6-GFX10.

This section describes general syntax for instructions and operands.

Instructions
~~~~~~~~~~~~

.. toctree::
   :hidden:

   AMDGPU/AMDGPUAsmGFX7
   AMDGPU/AMDGPUAsmGFX8
   AMDGPU/AMDGPUAsmGFX9
   AMDGPU/AMDGPUAsmGFX900
   AMDGPU/AMDGPUAsmGFX904
   AMDGPU/AMDGPUAsmGFX906
   AMDGPU/AMDGPUAsmGFX908
   AMDGPU/AMDGPUAsmGFX10
   AMDGPU/AMDGPUAsmGFX1011
   AMDGPUModifierSyntax
   AMDGPUOperandSyntax
   AMDGPUInstructionSyntax
   AMDGPUInstructionNotation

An instruction has the following :doc:`syntax<AMDGPUInstructionSyntax>`:

  | ``<``\ *opcode*\ ``> <``\ *operand0*\ ``>, <``\ *operand1*\ ``>,...
    <``\ *modifier0*\ ``> <``\ *modifier1*\ ``>...``

:doc:`Operands<AMDGPUOperandSyntax>` are comma-separated while
:doc:`modifiers<AMDGPUModifierSyntax>` are space-separated.

The order of operands and modifiers is fixed.
Most modifiers are optional and may be omitted.

Links to detailed instruction syntax description may be found in the following
table. Note that features under development are not included
in this description.

    =================================== =======================================
    Core ISA                            ISA Extensions
    =================================== =======================================
    :doc:`GFX7<AMDGPU/AMDGPUAsmGFX7>`   \-
    :doc:`GFX8<AMDGPU/AMDGPUAsmGFX8>`   \-
    :doc:`GFX9<AMDGPU/AMDGPUAsmGFX9>`   :doc:`gfx900<AMDGPU/AMDGPUAsmGFX900>`

                                        :doc:`gfx902<AMDGPU/AMDGPUAsmGFX900>`

                                        :doc:`gfx904<AMDGPU/AMDGPUAsmGFX904>`

                                        :doc:`gfx906<AMDGPU/AMDGPUAsmGFX906>`

                                        :doc:`gfx908<AMDGPU/AMDGPUAsmGFX908>`

                                        :doc:`gfx909<AMDGPU/AMDGPUAsmGFX900>`

    :doc:`GFX10<AMDGPU/AMDGPUAsmGFX10>` :doc:`gfx1011<AMDGPU/AMDGPUAsmGFX1011>`

                                        :doc:`gfx1012<AMDGPU/AMDGPUAsmGFX1011>`
    =================================== =======================================

For more information about instructions, their semantics and supported
combinations of operands, refer to one of instruction set architecture manuals
[AMD-GCN-GFX6]_, [AMD-GCN-GFX7]_, [AMD-GCN-GFX8]_, [AMD-GCN-GFX9]_ and
[AMD-GCN-GFX10]_.

Operands
~~~~~~~~

Detailed description of operands may be found :doc:`here<AMDGPUOperandSyntax>`.

Modifiers
~~~~~~~~~

Detailed description of modifiers may be found
:doc:`here<AMDGPUModifierSyntax>`.

Instruction Examples
~~~~~~~~~~~~~~~~~~~~

DS
++

.. code-block:: nasm

  ds_add_u32 v2, v4 offset:16
  ds_write_src2_b64 v2 offset0:4 offset1:8
  ds_cmpst_f32 v2, v4, v6
  ds_min_rtn_f64 v[8:9], v2, v[4:5]

For full list of supported instructions, refer to "LDS/GDS instructions" in ISA
Manual.

FLAT
++++

.. code-block:: nasm

  flat_load_dword v1, v[3:4]
  flat_store_dwordx3 v[3:4], v[5:7]
  flat_atomic_swap v1, v[3:4], v5 glc
  flat_atomic_cmpswap v1, v[3:4], v[5:6] glc slc
  flat_atomic_fmax_x2 v[1:2], v[3:4], v[5:6] glc

For full list of supported instructions, refer to "FLAT instructions" in ISA
Manual.

MUBUF
+++++

.. code-block:: nasm

  buffer_load_dword v1, off, s[4:7], s1
  buffer_store_dwordx4 v[1:4], v2, ttmp[4:7], s1 offen offset:4 glc tfe
  buffer_store_format_xy v[1:2], off, s[4:7], s1
  buffer_wbinvl1
  buffer_atomic_inc v1, v2, s[8:11], s4 idxen offset:4 slc

For full list of supported instructions, refer to "MUBUF Instructions" in ISA
Manual.

SMRD/SMEM
+++++++++

.. code-block:: nasm

  s_load_dword s1, s[2:3], 0xfc
  s_load_dwordx8 s[8:15], s[2:3], s4
  s_load_dwordx16 s[88:103], s[2:3], s4
  s_dcache_inv_vol
  s_memtime s[4:5]

For full list of supported instructions, refer to "Scalar Memory Operations" in
ISA Manual.

SOP1
++++

.. code-block:: nasm

  s_mov_b32 s1, s2
  s_mov_b64 s[0:1], 0x80000000
  s_cmov_b32 s1, 200
  s_wqm_b64 s[2:3], s[4:5]
  s_bcnt0_i32_b64 s1, s[2:3]
  s_swappc_b64 s[2:3], s[4:5]
  s_cbranch_join s[4:5]

For full list of supported instructions, refer to "SOP1 Instructions" in ISA
Manual.

SOP2
++++

.. code-block:: nasm

  s_add_u32 s1, s2, s3
  s_and_b64 s[2:3], s[4:5], s[6:7]
  s_cselect_b32 s1, s2, s3
  s_andn2_b32 s2, s4, s6
  s_lshr_b64 s[2:3], s[4:5], s6
  s_ashr_i32 s2, s4, s6
  s_bfm_b64 s[2:3], s4, s6
  s_bfe_i64 s[2:3], s[4:5], s6
  s_cbranch_g_fork s[4:5], s[6:7]

For full list of supported instructions, refer to "SOP2 Instructions" in ISA
Manual.

SOPC
++++

.. code-block:: nasm

  s_cmp_eq_i32 s1, s2
  s_bitcmp1_b32 s1, s2
  s_bitcmp0_b64 s[2:3], s4
  s_setvskip s3, s5

For full list of supported instructions, refer to "SOPC Instructions" in ISA
Manual.

SOPP
++++

.. code-block:: nasm

  s_barrier
  s_nop 2
  s_endpgm
  s_waitcnt 0 ; Wait for all counters to be 0
  s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0) ; Equivalent to above
  s_waitcnt vmcnt(1) ; Wait for vmcnt counter to be 1.
  s_sethalt 9
  s_sleep 10
  s_sendmsg 0x1
  s_sendmsg sendmsg(MSG_INTERRUPT)
  s_trap 1

For full list of supported instructions, refer to "SOPP Instructions" in ISA
Manual.

Unless otherwise mentioned, little verification is performed on the operands
of SOPP Instructions, so it is up to the programmer to be familiar with the
range or acceptable values.

VALU
++++

For vector ALU instruction opcodes (VOP1, VOP2, VOP3, VOPC, VOP_DPP, VOP_SDWA),
the assembler will automatically use optimal encoding based on its operands. To
force specific encoding, one can add a suffix to the opcode of the instruction:

* _e32 for 32-bit VOP1/VOP2/VOPC
* _e64 for 64-bit VOP3
* _dpp for VOP_DPP
* _sdwa for VOP_SDWA

VOP1/VOP2/VOP3/VOPC examples:

.. code-block:: nasm

  v_mov_b32 v1, v2
  v_mov_b32_e32 v1, v2
  v_nop
  v_cvt_f64_i32_e32 v[1:2], v2
  v_floor_f32_e32 v1, v2
  v_bfrev_b32_e32 v1, v2
  v_add_f32_e32 v1, v2, v3
  v_mul_i32_i24_e64 v1, v2, 3
  v_mul_i32_i24_e32 v1, -3, v3
  v_mul_i32_i24_e32 v1, -100, v3
  v_addc_u32 v1, s[0:1], v2, v3, s[2:3]
  v_max_f16_e32 v1, v2, v3

VOP_DPP examples:

.. code-block:: nasm

  v_mov_b32 v0, v0 quad_perm:[0,2,1,1]
  v_sin_f32 v0, v0 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
  v_mov_b32 v0, v0 wave_shl:1
  v_mov_b32 v0, v0 row_mirror
  v_mov_b32 v0, v0 row_bcast:31
  v_mov_b32 v0, v0 quad_perm:[1,3,0,1] row_mask:0xa bank_mask:0x1 bound_ctrl:0
  v_add_f32 v0, v0, |v0| row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
  v_max_f16 v1, v2, v3 row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0

VOP_SDWA examples:

.. code-block:: nasm

  v_mov_b32 v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
  v_min_u32 v200, v200, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
  v_sin_f32 v0, v0 dst_unused:UNUSED_PAD src0_sel:WORD_1
  v_fract_f32 v0, |v0| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
  v_cmpx_le_u32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

For full list of supported instructions, refer to "Vector ALU instructions".

.. TODO::

  Remove once we switch to code object v3 by default.

.. _amdgpu-amdhsa-assembler-predefined-symbols-v2:

Code Object V2 Predefined Symbols (-mattr=-code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Code Object V2 is not the default code object version emitted by
  this version of LLVM. For a description of the predefined symbols available
  with the default configuration (Code Object V3) see
  :ref:`amdgpu-amdhsa-assembler-predefined-symbols-v3`.

The AMDGPU assembler defines and updates some symbols automatically. These
symbols do not affect code generation.

.option.machine_version_major
+++++++++++++++++++++++++++++

Set to the GFX major generation number of the target being assembled for. For
example, when assembling for a "GFX9" target this will be set to the integer
value "9". The possible GFX major generation numbers are presented in
:ref:`amdgpu-processors`.

.option.machine_version_minor
+++++++++++++++++++++++++++++

Set to the GFX minor generation number of the target being assembled for. For
example, when assembling for a "GFX810" target this will be set to the integer
value "1". The possible GFX minor generation numbers are presented in
:ref:`amdgpu-processors`.

.option.machine_version_stepping
++++++++++++++++++++++++++++++++

Set to the GFX stepping generation number of the target being assembled for.
For example, when assembling for a "GFX704" target this will be set to the
integer value "4". The possible GFX stepping generation numbers are presented
in :ref:`amdgpu-processors`.

.kernel.vgpr_count
++++++++++++++++++

Set to zero each time a
:ref:`amdgpu-amdhsa-assembler-directive-amdgpu_hsa_kernel` directive is
encountered. At each instruction, if the current value of this symbol is less
than or equal to the maximum VPGR number explicitly referenced within that
instruction then the symbol value is updated to equal that VGPR number plus
one.

.kernel.sgpr_count
++++++++++++++++++

Set to zero each time a
:ref:`amdgpu-amdhsa-assembler-directive-amdgpu_hsa_kernel` directive is
encountered. At each instruction, if the current value of this symbol is less
than or equal to the maximum VPGR number explicitly referenced within that
instruction then the symbol value is updated to equal that SGPR number plus
one.

.. _amdgpu-amdhsa-assembler-directives-v2:

Code Object V2 Directives (-mattr=-code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Code Object V2 is not the default code object version emitted by
  this version of LLVM. For a description of the directives supported with
  the default configuration (Code Object V3) see
  :ref:`amdgpu-amdhsa-assembler-directives-v3`.

AMDGPU ABI defines auxiliary data in output code object. In assembly source,
one can specify them with assembler directives.

.hsa_code_object_version major, minor
+++++++++++++++++++++++++++++++++++++

*major* and *minor* are integers that specify the version of the HSA code
object that will be generated by the assembler.

.hsa_code_object_isa [major, minor, stepping, vendor, arch]
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


*major*, *minor*, and *stepping* are all integers that describe the instruction
set architecture (ISA) version of the assembly program.

*vendor* and *arch* are quoted strings. *vendor* should always be equal to
"AMD" and *arch* should always be equal to "AMDGPU".

By default, the assembler will derive the ISA version, *vendor*, and *arch*
from the value of the -mcpu option that is passed to the assembler.

.. _amdgpu-amdhsa-assembler-directive-amdgpu_hsa_kernel:

.amdgpu_hsa_kernel (name)
+++++++++++++++++++++++++

This directives specifies that the symbol with given name is a kernel entry
point (label) and the object should contain corresponding symbol of type
STT_AMDGPU_HSA_KERNEL.

.amd_kernel_code_t
++++++++++++++++++

This directive marks the beginning of a list of key / value pairs that are used
to specify the amd_kernel_code_t object that will be emitted by the assembler.
The list must be terminated by the *.end_amd_kernel_code_t* directive. For any
amd_kernel_code_t values that are unspecified a default value will be used. The
default value for all keys is 0, with the following exceptions:

- *amd_code_version_major* defaults to 1.
- *amd_kernel_code_version_minor* defaults to 2.
- *amd_machine_kind* defaults to 1.
- *amd_machine_version_major*, *machine_version_minor*, and
  *amd_machine_version_stepping* are derived from the value of the -mcpu option
  that is passed to the assembler.
- *kernel_code_entry_byte_offset* defaults to 256.
- *wavefront_size* defaults 6 for all targets before GFX10. For GFX10 onwards
  defaults to 6 if target feature ``wavefrontsize64`` is enabled, otherwise 5.
  Note that wavefront size is specified as a power of two, so a value of **n**
  means a size of 2^ **n**.
- *call_convention* defaults to -1.
- *kernarg_segment_alignment*, *group_segment_alignment*, and
  *private_segment_alignment* default to 4. Note that alignments are specified
  as a power of 2, so a value of **n** means an alignment of 2^ **n**.
- *enable_wgp_mode* defaults to 1 if target feature ``cumode`` is disabled for
  GFX10 onwards.
- *enable_mem_ordered* defaults to 1 for GFX10 onwards.

The *.amd_kernel_code_t* directive must be placed immediately after the
function label and before any instructions.

For a full list of amd_kernel_code_t keys, refer to AMDGPU ABI document,
comments in lib/Target/AMDGPU/AmdKernelCodeT.h and test/CodeGen/AMDGPU/hsa.s.

.. _amdgpu-amdhsa-assembler-example-v2:

Code Object V2 Example Source Code (-mattr=-code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: Code Object V2 is not the default code object version emitted by
  this version of LLVM. For a description of the directives supported with
  the default configuration (Code Object V3) see
  :ref:`amdgpu-amdhsa-assembler-example-v3`.

Here is an example of a minimal assembly source file, defining one HSA kernel:

.. code::
   :number-lines:

   .hsa_code_object_version 1,0
   .hsa_code_object_isa

   .hsatext
   .globl  hello_world
   .p2align 8
   .amdgpu_hsa_kernel hello_world

   hello_world:

      .amd_kernel_code_t
         enable_sgpr_kernarg_segment_ptr = 1
         is_ptr64 = 1
         compute_pgm_rsrc1_vgprs = 0
         compute_pgm_rsrc1_sgprs = 0
         compute_pgm_rsrc2_user_sgpr = 2
         compute_pgm_rsrc1_wgp_mode = 0
         compute_pgm_rsrc1_mem_ordered = 0
         compute_pgm_rsrc1_fwd_progress = 1
     .end_amd_kernel_code_t

     s_load_dwordx2 s[0:1], s[0:1] 0x0
     v_mov_b32 v0, 3.14159
     s_waitcnt lgkmcnt(0)
     v_mov_b32 v1, s0
     v_mov_b32 v2, s1
     flat_store_dword v[1:2], v0
     s_endpgm
   .Lfunc_end0:
        .size   hello_world, .Lfunc_end0-hello_world

.. _amdgpu-amdhsa-assembler-predefined-symbols-v3:

Code Object V3 Predefined Symbols (-mattr=+code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AMDGPU assembler defines and updates some symbols automatically. These
symbols do not affect code generation.

.amdgcn.gfx_generation_number
+++++++++++++++++++++++++++++

Set to the GFX major generation number of the target being assembled for. For
example, when assembling for a "GFX9" target this will be set to the integer
value "9". The possible GFX major generation numbers are presented in
:ref:`amdgpu-processors`.

.amdgcn.gfx_generation_minor
++++++++++++++++++++++++++++

Set to the GFX minor generation number of the target being assembled for. For
example, when assembling for a "GFX810" target this will be set to the integer
value "1". The possible GFX minor generation numbers are presented in
:ref:`amdgpu-processors`.

.amdgcn.gfx_generation_stepping
+++++++++++++++++++++++++++++++

Set to the GFX stepping generation number of the target being assembled for.
For example, when assembling for a "GFX704" target this will be set to the
integer value "4". The possible GFX stepping generation numbers are presented
in :ref:`amdgpu-processors`.

.. _amdgpu-amdhsa-assembler-symbol-next_free_vgpr:

.amdgcn.next_free_vgpr
++++++++++++++++++++++

Set to zero before assembly begins. At each instruction, if the current value
of this symbol is less than or equal to the maximum VGPR number explicitly
referenced within that instruction then the symbol value is updated to equal
that VGPR number plus one.

May be used to set the `.amdhsa_next_free_vpgr` directive in
:ref:`amdhsa-kernel-directives-table`.

May be set at any time, e.g. manually set to zero at the start of each kernel.

.. _amdgpu-amdhsa-assembler-symbol-next_free_sgpr:

.amdgcn.next_free_sgpr
++++++++++++++++++++++

Set to zero before assembly begins. At each instruction, if the current value
of this symbol is less than or equal the maximum SGPR number explicitly
referenced within that instruction then the symbol value is updated to equal
that SGPR number plus one.

May be used to set the `.amdhsa_next_free_spgr` directive in
:ref:`amdhsa-kernel-directives-table`.

May be set at any time, e.g. manually set to zero at the start of each kernel.

.. _amdgpu-amdhsa-assembler-directives-v3:

Code Object V3 Directives (-mattr=+code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directives which begin with ``.amdgcn`` are valid for all ``amdgcn``
architecture processors, and are not OS-specific. Directives which begin with
``.amdhsa`` are specific to ``amdgcn`` architecture processors when the
``amdhsa`` OS is specified. See :ref:`amdgpu-target-triples` and
:ref:`amdgpu-processors`.

.amdgcn_target <target>
+++++++++++++++++++++++

Optional directive which declares the target supported by the containing
assembler source file. Valid values are described in
:ref:`amdgpu-amdhsa-code-object-target-identification`. Used by the assembler
to validate command-line options such as ``-triple``, ``-mcpu``, and those
which specify target features.

.amdhsa_kernel <name>
+++++++++++++++++++++

Creates a correctly aligned AMDHSA kernel descriptor and a symbol,
``<name>.kd``, in the current location of the current section. Only valid when
the OS is ``amdhsa``. ``<name>`` must be a symbol that labels the first
instruction to execute, and does not need to be previously defined.

Marks the beginning of a list of directives used to generate the bytes of a
kernel descriptor, as described in :ref:`amdgpu-amdhsa-kernel-descriptor`.
Directives which may appear in this list are described in
:ref:`amdhsa-kernel-directives-table`. Directives may appear in any order, must
be valid for the target being assembled for, and cannot be repeated. Directives
support the range of values specified by the field they reference in
:ref:`amdgpu-amdhsa-kernel-descriptor`. If a directive is not specified, it is
assumed to have its default value, unless it is marked as "Required", in which
case it is an error to omit the directive. This list of directives is
terminated by an ``.end_amdhsa_kernel`` directive.

  .. table:: AMDHSA Kernel Assembler Directives
     :name: amdhsa-kernel-directives-table

     ======================================================== =================== ============ ===================
     Directive                                                Default             Supported On Description
     ======================================================== =================== ============ ===================
     ``.amdhsa_group_segment_fixed_size``                     0                   GFX6-GFX10   Controls GROUP_SEGMENT_FIXED_SIZE in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_private_segment_fixed_size``                   0                   GFX6-GFX10   Controls PRIVATE_SEGMENT_FIXED_SIZE in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_private_segment_buffer``             0                   GFX6-GFX10   Controls ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_dispatch_ptr``                       0                   GFX6-GFX10   Controls ENABLE_SGPR_DISPATCH_PTR in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_queue_ptr``                          0                   GFX6-GFX10   Controls ENABLE_SGPR_QUEUE_PTR in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_kernarg_segment_ptr``                0                   GFX6-GFX10   Controls ENABLE_SGPR_KERNARG_SEGMENT_PTR in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_dispatch_id``                        0                   GFX6-GFX10   Controls ENABLE_SGPR_DISPATCH_ID in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_flat_scratch_init``                  0                   GFX6-GFX10   Controls ENABLE_SGPR_FLAT_SCRATCH_INIT in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_user_sgpr_private_segment_size``               0                   GFX6-GFX10   Controls ENABLE_SGPR_PRIVATE_SEGMENT_SIZE in
                                                                                               :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
     ``.amdhsa_wavefront_size32``                             Target              GFX10        Controls ENABLE_WAVEFRONT_SIZE32 in
                                                              Feature                          :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
                                                              Specific
                                                              (-wavefrontsize64)
     ``.amdhsa_system_sgpr_private_segment_wavefront_offset`` 0                   GFX6-GFX10   Controls ENABLE_SGPR_PRIVATE_SEGMENT_WAVEFRONT_OFFSET in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_system_sgpr_workgroup_id_x``                   1                   GFX6-GFX10   Controls ENABLE_SGPR_WORKGROUP_ID_X in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_system_sgpr_workgroup_id_y``                   0                   GFX6-GFX10   Controls ENABLE_SGPR_WORKGROUP_ID_Y in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_system_sgpr_workgroup_id_z``                   0                   GFX6-GFX10   Controls ENABLE_SGPR_WORKGROUP_ID_Z in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_system_sgpr_workgroup_info``                   0                   GFX6-GFX10   Controls ENABLE_SGPR_WORKGROUP_INFO in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_system_vgpr_workitem_id``                      0                   GFX6-GFX10   Controls ENABLE_VGPR_WORKITEM_ID in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
                                                                                               Possible values are defined in
                                                                                               :ref:`amdgpu-amdhsa-system-vgpr-work-item-id-enumeration-values-table`.
     ``.amdhsa_next_free_vgpr``                               Required            GFX6-GFX10   Maximum VGPR number explicitly referenced, plus one.
                                                                                               Used to calculate GRANULATED_WORKITEM_VGPR_COUNT in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_next_free_sgpr``                               Required            GFX6-GFX10   Maximum SGPR number explicitly referenced, plus one.
                                                                                               Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_reserve_vcc``                                  1                   GFX6-GFX10   Whether the kernel may use the special VCC SGPR.
                                                                                               Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_reserve_flat_scratch``                         1                   GFX7-GFX10   Whether the kernel may use flat instructions to access
                                                                                               scratch memory. Used to calculate
                                                                                               GRANULATED_WAVEFRONT_SGPR_COUNT in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_reserve_xnack_mask``                           Target              GFX8-GFX10   Whether the kernel may trigger XNACK replay.
                                                              Feature                          Used to calculate GRANULATED_WAVEFRONT_SGPR_COUNT in
                                                              Specific                         :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
                                                              (+xnack)
     ``.amdhsa_float_round_mode_32``                          0                   GFX6-GFX10   Controls FLOAT_ROUND_MODE_32 in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
                                                                                               Possible values are defined in
                                                                                               :ref:`amdgpu-amdhsa-floating-point-rounding-mode-enumeration-values-table`.
     ``.amdhsa_float_round_mode_16_64``                       0                   GFX6-GFX10   Controls FLOAT_ROUND_MODE_16_64 in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
                                                                                               Possible values are defined in
                                                                                               :ref:`amdgpu-amdhsa-floating-point-rounding-mode-enumeration-values-table`.
     ``.amdhsa_float_denorm_mode_32``                         0                   GFX6-GFX10   Controls FLOAT_DENORM_MODE_32 in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
                                                                                               Possible values are defined in
                                                                                               :ref:`amdgpu-amdhsa-floating-point-denorm-mode-enumeration-values-table`.
     ``.amdhsa_float_denorm_mode_16_64``                      3                   GFX6-GFX10   Controls FLOAT_DENORM_MODE_16_64 in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
                                                                                               Possible values are defined in
                                                                                               :ref:`amdgpu-amdhsa-floating-point-denorm-mode-enumeration-values-table`.
     ``.amdhsa_dx10_clamp``                                   1                   GFX6-GFX10   Controls ENABLE_DX10_CLAMP in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_ieee_mode``                                    1                   GFX6-GFX10   Controls ENABLE_IEEE_MODE in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_fp16_overflow``                                0                   GFX9-GFX10   Controls FP16_OVFL in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_workgroup_processor_mode``                     Target              GFX10        Controls ENABLE_WGP_MODE in
                                                              Feature                          :ref:`amdgpu-amdhsa-kernel-descriptor-gfx6-gfx10-table`.
                                                              Specific
                                                              (-cumode)
     ``.amdhsa_memory_ordered``                               1                   GFX10        Controls MEM_ORDERED in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_forward_progress``                             0                   GFX10        Controls FWD_PROGRESS in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc1-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_ieee_invalid_op``                 0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_denorm_src``                      0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_FP_DENORMAL_SOURCE in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_ieee_div_zero``                   0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_ieee_overflow``                   0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_ieee_underflow``                  0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_fp_ieee_inexact``                    0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_IEEE_754_FP_INEXACT in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ``.amdhsa_exception_int_div_zero``                       0                   GFX6-GFX10   Controls ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO in
                                                                                               :ref:`amdgpu-amdhsa-compute_pgm_rsrc2-gfx6-gfx10-table`.
     ======================================================== =================== ============ ===================

.amdgpu_metadata
++++++++++++++++

Optional directive which declares the contents of the ``NT_AMDGPU_METADATA``
note record (see :ref:`amdgpu-elf-note-records-table-v3`).

The contents must be in the [YAML]_ markup format, with the same structure and
semantics described in :ref:`amdgpu-amdhsa-code-object-metadata-v3`.

This directive is terminated by an ``.end_amdgpu_metadata`` directive.

.. _amdgpu-amdhsa-assembler-example-v3:

Code Object V3 Example Source Code (-mattr=+code-object-v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of a minimal assembly source file, defining one HSA kernel:

.. code::
   :number-lines:

   .amdgcn_target "amdgcn-amd-amdhsa--gfx900+xnack" // optional

   .text
   .globl hello_world
   .p2align 8
   .type hello_world,@function
   hello_world:
     s_load_dwordx2 s[0:1], s[0:1] 0x0
     v_mov_b32 v0, 3.14159
     s_waitcnt lgkmcnt(0)
     v_mov_b32 v1, s0
     v_mov_b32 v2, s1
     flat_store_dword v[1:2], v0
     s_endpgm
   .Lfunc_end0:
     .size   hello_world, .Lfunc_end0-hello_world

   .rodata
   .p2align 6
   .amdhsa_kernel hello_world
     .amdhsa_user_sgpr_kernarg_segment_ptr 1
     .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
     .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
   .end_amdhsa_kernel

   .amdgpu_metadata
   ---
   amdhsa.version:
     - 1
     - 0
   amdhsa.kernels:
     - .name: hello_world
       .symbol: hello_world.kd
       .kernarg_segment_size: 48
       .group_segment_fixed_size: 0
       .private_segment_fixed_size: 0
       .kernarg_segment_align: 4
       .wavefront_size: 64
       .sgpr_count: 2
       .vgpr_count: 3
       .max_flat_workgroup_size: 256
   ...
   .end_amdgpu_metadata

If an assembly source file contains multiple kernels and/or functions, the
:ref:`amdgpu-amdhsa-assembler-symbol-next_free_vgpr` and
:ref:`amdgpu-amdhsa-assembler-symbol-next_free_sgpr` symbols may be reset using
the ``.set <symbol>, <expression>`` directive. For example, in the case of two
kernels, where ``function1`` is only called from ``kernel1`` it is sufficient
to group the function with the kernel that calls it and reset the symbols
between the two connected components:

.. code::
   :number-lines:

   .amdgcn_target "amdgcn-amd-amdhsa--gfx900+xnack" // optional

   // gpr tracking symbols are implicitly set to zero

   .text
   .globl kern0
   .p2align 8
   .type kern0,@function
   kern0:
     // ...
     s_endpgm
   .Lkern0_end:
     .size   kern0, .Lkern0_end-kern0

   .rodata
   .p2align 6
   .amdhsa_kernel kern0
     // ...
     .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
     .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
   .end_amdhsa_kernel

   // reset symbols to begin tracking usage in func1 and kern1
   .set .amdgcn.next_free_vgpr, 0
   .set .amdgcn.next_free_sgpr, 0

   .text
   .hidden func1
   .global func1
   .p2align 2
   .type func1,@function
   func1:
     // ...
     s_setpc_b64 s[30:31]
   .Lfunc1_end:
   .size func1, .Lfunc1_end-func1

   .globl kern1
   .p2align 8
   .type kern1,@function
   kern1:
     // ...
     s_getpc_b64 s[4:5]
     s_add_u32 s4, s4, func1@rel32@lo+4
     s_addc_u32 s5, s5, func1@rel32@lo+4
     s_swappc_b64 s[30:31], s[4:5]
     // ...
     s_endpgm
   .Lkern1_end:
     .size   kern1, .Lkern1_end-kern1

   .rodata
   .p2align 6
   .amdhsa_kernel kern1
     // ...
     .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
     .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
   .end_amdhsa_kernel

These symbols cannot identify connected components in order to automatically
track the usage for each kernel. However, in some cases careful organization of
the kernels and functions in the source file means there is minimal additional
effort required to accurately calculate GPR usage.

Additional Documentation
========================

.. [AMD-RADEON-HD-2000-3000] `AMD R6xx shader ISA <http://developer.amd.com/wordpress/media/2012/10/R600_Instruction_Set_Architecture.pdf>`__
.. [AMD-RADEON-HD-4000] `AMD R7xx shader ISA <http://developer.amd.com/wordpress/media/2012/10/R700-Family_Instruction_Set_Architecture.pdf>`__
.. [AMD-RADEON-HD-5000] `AMD Evergreen shader ISA <http://developer.amd.com/wordpress/media/2012/10/AMD_Evergreen-Family_Instruction_Set_Architecture.pdf>`__
.. [AMD-RADEON-HD-6000] `AMD Cayman/Trinity shader ISA <http://developer.amd.com/wordpress/media/2012/10/AMD_HD_6900_Series_Instruction_Set_Architecture.pdf>`__
.. [AMD-GCN-GFX6] `AMD Southern Islands Series ISA <http://developer.amd.com/wordpress/media/2012/12/AMD_Southern_Islands_Instruction_Set_Architecture.pdf>`__
.. [AMD-GCN-GFX7] `AMD Sea Islands Series ISA <http://developer.amd.com/wordpress/media/2013/07/AMD_Sea_Islands_Instruction_Set_Architecture.pdf>`_
.. [AMD-GCN-GFX8] `AMD GCN3 Instruction Set Architecture <http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf>`__
.. [AMD-GCN-GFX9] `AMD "Vega" Instruction Set Architecture <http://developer.amd.com/wordpress/media/2013/12/Vega_Shader_ISA_28July2017.pdf>`__
.. [AMD-GCN-GFX10] `AMD "RDNA 1.0" Instruction Set Architecture <https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Shader_ISA_5August2019.pdf>`__
.. [AMD-ROCm] `ROCm: Open Platform for Development, Discovery and Education Around GPU Computing <http://gpuopen.com/compute-product/rocm/>`__
.. [AMD-ROCm-github] `ROCm github <http://github.com/RadeonOpenCompute>`__
.. [HSA] `Heterogeneous System Architecture (HSA) Foundation <http://www.hsafoundation.com/>`__
.. [HIP] `HIP Programming Guide <https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-programing-guide>`__
.. [ELF] `Executable and Linkable Format (ELF) <http://www.sco.com/developers/gabi/>`__
.. [DWARF] `DWARF Debugging Information Format <http://dwarfstd.org/>`__
.. [YAML] `YAML Ain't Markup Language (YAML™) Version 1.2 <http://www.yaml.org/spec/1.2/spec.html>`__
.. [MsgPack] `Message Pack <http://www.msgpack.org/>`__
.. [SEMVER] `Semantic Versioning <https://semver.org/>`__
.. [OpenCL] `The OpenCL Specification Version 2.0 <http://www.khronos.org/registry/cl/specs/opencl-2.0.pdf>`__
.. [HRF] `Heterogeneous-race-free Memory Models <http://benedictgaster.org/wp-content/uploads/2014/01/asplos269-FINAL.pdf>`__
.. [CLANG-ATTR] `Attributes in Clang <https://clang.llvm.org/docs/AttributeReference.html>`__
