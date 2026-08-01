llvm-calc-occupancy - AMDGPU occupancy calculator
==================================================

.. program:: llvm-calc-occupancy

SYNOPSIS
--------

:program:`llvm-calc-occupancy` -mcpu=<gfxNNN> [*options*]

DESCRIPTION
-----------

:program:`llvm-calc-occupancy` reports the occupancy (waves per execution unit)
that an AMDGPU kernel would achieve for a given combination of workgroup size,
VGPR usage, SGPR usage and LDS allocation. It is a thin front-end over the same
occupancy math the AMDGPU backend uses (``GCNSubtarget``), so the numbers match
what the compiler computes for an equivalent kernel.

Any resource that is not specified is treated as unconstrained. When the
workgroup size is left unspecified (or given as a range), the occupancy is
reported as a range as well.

The tool only supports the AMDGPU target. New callers should use the
``amdgpu`` triple (for example ``amdgpu-amd-amdhsa``); the legacy ``amdgcn``
spelling is still accepted.

EXAMPLE
-------

.. code-block:: console

  $ llvm-calc-occupancy -mcpu=gfx90a --wg-size=512 --vgprs=50 --sgprs=30
  llvm-calc-occupancy - AMDGPU occupancy calculator

  Target
    Triple:              amdgpu-amd-amdhsa
    GPU (-mcpu):         gfx90a
    Wavefront size:      64
    Max waves/EU:        8 (waves per SIMD, hardware limit)
    EUs (SIMDs) per CU:  4
    ...

  Per-constraint occupancy (waves/EU)
    Workgroup + LDS:     8
    VGPRs:               8
    SGPRs:               8

  Result
    Occupancy:           8 waves/EU (32 waves/CU)
    Limited by:          workgroup size / LDS, VGPRs, SGPRs
    Next step:           already at the hardware maximum

OPTIONS
-------

.. option:: -mcpu=<gfxNNN>

  Target GPU, for example ``gfx90a``. Required.

.. option:: -mtriple=<triple>

  Target triple. Defaults to ``amdgpu-amd-amdhsa``. Must be an AMDGPU triple
  (the ``amdgpu`` or legacy ``amdgcn`` arch).

.. option:: -mattr=<features>

  Comma-separated list of subtarget features, for example
  ``+wavefrontsize32``.

.. option:: --wg-size=<N>, --flat-workgroup-size=<N>

  Flat workgroup size. Accepts a single value ``N`` or a range ``MIN:MAX``.
  When omitted, the full legal range is assumed and the occupancy is reported
  as a range.

.. option:: --vgprs=<N>

  Number of VGPRs used per lane. When omitted, VGPRs do not constrain the
  occupancy.

.. option:: --sgprs=<N>

  Number of SGPRs used per wave. When omitted, SGPRs do not constrain the
  occupancy.

.. option:: --lds=<size>

  LDS bytes allocated per workgroup. Accepts an optional ``k``/``kb`` or
  ``m``/``mb`` suffix (base 1024).

.. option:: --dynamic-vgpr-block-size=<N>

  Dynamic VGPR block size. ``0`` (the default) disables dynamic VGPR mode.

.. option:: --limits

  Print, for each occupancy level supported by the GPU, the maximum number of
  VGPRs and SGPRs that still reaches that level.

.. option:: --help

  Print a summary of command line options and exit.

EXIT STATUS
-----------

:program:`llvm-calc-occupancy` returns 0 on success and a non-zero exit code if
the arguments are invalid (for example a missing or non-AMDGPU target).
