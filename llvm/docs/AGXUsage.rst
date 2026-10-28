=============================================
User Guide for the AGX Backend (Experimental)
=============================================

.. contents::
   :local:
   :depth: 2

Introduction
============

The AGX backend generates machine code for the GPUs of Apple silicon (the
"AGX" architecture of the M1 and later SoCs). It is an experimental target:
build it with ``-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=AGX``.

Apple does not document the instruction set. Everything in this backend comes
from public reverse engineering:

* `applegpu <https://github.com/dougallj/applegpu>`_, the reference for the
  G13 instruction encodings. The MC layer is tested against it: every
  encoding produced by the assembler and the code generator decodes to the
  same instruction in applegpu's disassembler.
* The Mesa ``asahi`` driver, the reference for the code that the hardware
  expects in practice (control flow on the nesting counter, scoreboarding,
  memory addressing). Only facts about the hardware were taken from it.
* Measurements of instruction latencies on G14 (M2 Max), used for the
  scheduling models.
* Reverse engineering notes on the G16/G17 ("Apple9") generation. These are
  incomplete, which is why code generation for those GPUs is not supported
  yet (see :ref:`agx-apple9`).

The backend is modelled on the AMDGPU backend: the same split between code
generation pseudo instructions and real, per encoding family instructions,
the same structurization of control flow and the same layout of the
sources.

Target triples
==============

Use ``agx`` as the architecture, for example ``agx-apple-macosx`` or simply
``-mtriple=agx``. The triple has no effect on code generation beyond selecting
the target. Object files are ELF (``EM_AGX``, ``0x4147``, a value that is not
registered); they contain the raw instructions of each kernel in ``.text``.

Processors
==========

Use ``-mcpu`` (or the ``target-cpu`` function attribute) to select the GPU.

.. table:: AGX processors
   :name: agx-processors-table

   =============== ========== ================= =====================================
   Processor       Generation Encoding family   Products
   =============== ========== ================= =====================================
   ``generic``     G13        AGX2              Any G13-G15 GPU
   ``agx-g13g``    G13        AGX2              M1
   ``agx-g13s``    G13        AGX2              M1 Pro (``multi-die-coherency``)
   ``agx-g13c``    G13        AGX2              M1 Max (``multi-die-coherency``)
   ``agx-g13d``    G13        AGX2              M1 Ultra (``multi-die-coherency``)
   ``agx-g14g``    G14        AGX2              M2
   ``agx-g14s``    G14        AGX2              M2 Pro
   ``agx-g14c``    G14        AGX2              M2 Max
   ``agx-g14d``    G14        AGX2              M2 Ultra (``multi-die-coherency``)
   ``agx-g15g``    G15        AGX2              M3
   ``agx-g15s``    G15        AGX2              M3 Pro
   ``agx-g15c``    G15        AGX2              M3 Max
   ``agx-g15d``    G15        AGX2              M3 Ultra (``multi-die-coherency``)
   ``agx-g16g``    G16        Apple9            M4
   ``agx-g16s``    G16        Apple9            M4 Pro
   ``agx-g16c``    G16        Apple9            M4 Max
   ``agx-g17p``    G17        Apple9            A18 Pro
   ``agx-g17g``    G17        Apple9            M5
   =============== ========== ================= =====================================

G15 support assumes the G14 encoding; it has not been validated on hardware.

M2 Max (``agx-g14c``)
---------------------

The M2 Max is the reference G14 target. The values below were measured on an
M2 Max (driver-reported G14C B1) under Asahi Linux by the AGX Performance
Simulator project, and are the ones its cost and dispatch models use. A
runtime should query the core count and the clock rather than assume them:
binned parts have fewer cores.

.. table:: M2 Max (G14C) execution resources
   :name: agx-g14c-table

   ============================== ==============================================
   Resource                       Value
   ============================== ==============================================
   GPU cores                      38 on the full part, in 4 clusters of
                                  10, 9, 9 and 10 cores
   SIMD-group width               32 threads
   Resident SIMD-groups per core  ``min(96, 2 * floor(3328 / (2 * H)))``, where
                                  ``H`` is the number of 16-bit register halves
                                  the kernel declares (minimum 8); whole
                                  workgroups only
   Threadgroup memory per core    64 KiB; ``floor(65536 / bytes)`` workgroups
                                  fit, by threadgroup memory alone
   Threadgroup memory banks       32 banks of 4 bytes
   Threads per workgroup          1 to 1024
   Cache line                     64 bytes (L1 and L2)
   L1 data cache                  8 KiB per core
   L2 cache                       256 KiB per core
   DRAM bandwidth                 about 410 GB/s peak, 360 GB/s for a whole
                                  dispatch
   Top GPU clock                  1398 MHz (the clock ramps up over about
                                  15 ms after the GPU wakes)
   Concurrent compute commands    at most 4 GPU-wide
   Kernel launch                  about 55 us warm, 0.4-0.7 ms after wake-up
   ============================== ==============================================

Occupancy is limited by whichever of the register and threadgroup memory
limits is lower, in whole workgroups: for example, with 32-thread workgroups
the limit is near 55 SIMD-groups per core, and with 64-thread workgroups near
84. The ``max-gprs-128`` and ``max-gprs-64`` features trade registers for
occupancy along this curve.

Target features
===============

Each generation implies a set of features. Features can also be enabled or
disabled individually with ``-mattr`` (or the ``target-features`` attribute),
which allows code for one generation to be restricted to the capabilities of
another.

.. table:: AGX target features
   :name: agx-features-table

   ============================== ==============================================
   Feature                        Description
   ============================== ==============================================
   ``agx2-encoding``              G13/G14 instruction encoding.
   ``apple9-encoding``            G16/G17 instruction encoding.
   ``fp16-arith``                 Native half precision arithmetic.
   ``int64-add``                  64-bit ``iadd``/``imadd`` operands.
   ``simd-shuffle``               SIMD-group and quad shuffles.
   ``simd-reduce``                SIMD-group and quad reductions and prefixes.
   ``simd-ballot``                Ballots.
   ``simd-matrix``                8x8 SIMD-group matrix multiply-accumulate.
   ``quad-shuffle-index-merge``   Shuffles merge the low index bits of each
                                  quad. Divergent shuffle indices are expanded
                                  into four shuffles (see `SIMD-group
                                  shuffles`_).
   ``multi-die-coherency``        Device atomics set the coherency bit
                                  (multi-cluster G13 parts and two-die parts).
   ``threadgroup-atomics``        Threadgroup memory atomics.
   ``device-atomics``             Device memory atomics.
   ``device-atomic-fadd``         A float32 device atomic add (G14; rounds to
                                  nearest even and flushes denormals, like
                                  PTX ``atom.add.f32``).
   ``device-atomic-u64-minmax``   A 64-bit unsigned device atomic min and max
                                  without a result (G14), used for
                                  ``atomicrmw umin/umax i64`` whose old value
                                  is unused.
   ``stack-access``               Per-thread stack loads and stores.
   ``block-barrier``              ``block_barrier`` (``78 09``), a barrier over
                                  the caller's aligned block of 4 SIMD-groups
                                  that orders their memory accesses (G14,
                                  measured on G14C; AGX Performance Simulator
                                  ``tasks/ptx-re-blockers.md``, B16).
                                  ``llvm.agx.block.barrier`` is a
                                  ``threadgroup_barrier`` without it.
   ``dynamic-caching``            Registers and threadgroup memory are
                                  allocated dynamically (G15 and later).
   ``ray-tracing``                Hardware ray tracing (not used by code
                                  generation).
   ``mesh-shading``               Hardware mesh shading (not used by code
                                  generation).
   ``source-lifetime``            ALU sources carry register release flags
                                  (Apple9).
   ``max-gprs-128``, ``-64``      Limit the general purpose registers used.
   ``ieee-fp32``                  IEEE 754 ``float`` arithmetic in software
                                  where the hardware differs (off by default,
                                  see `Strict floating point`_).
   ``canonical-nan``              NaN results are the canonical NaN of PTX
                                  (off by default).
   ``agx-strict``                 Only the instruction forms Mesa's AGX
                                  compiler emits (G13 and G14; off by default,
                                  see `Strict mode`_).
   ============================== ==============================================

Strict mode
-----------

The ``agx-strict`` feature restricts code generation for G13 (M1) and G14 (M2)
processors to the instructions Mesa's asahi compiler emits for the same GPU,
so that the code relies only on encodings that Mesa's driver uses in
production. It is selected in IR by either

* the function attribute ``"target-features"="+agx-strict"`` (or ``-mattr=+agx-strict``), or
* the module flag ``!{i32 1, !"agx-strict", i32 1}``, which applies to every
  function of the module.

On a processor other than G13 and G14 the feature is a fatal error. It turns
off the capabilities Mesa does not use: ``device-atomic-fadd`` (float atomics
become compare-and-swap loops), ``device-atomic-u64-minmax`` (64-bit min and
max take the lock table), ``simd-matrix`` and ``block-barrier``. Device and threadgroup loads,
stores and atomics are encoded in their long form, as Mesa packs them.

Every instruction is then checked against Mesa's instruction forms after
encoding, for assembly output as well as for objects and for ``llvm-mc``: an
instruction Mesa would not emit is an error that names the closest Mesa form
and the field that differs, for example::

  error: ATOMIC_g13 is not an instruction Mesa emits (agx-strict): atomic: field value

The forms are generated by ``llvm/utils/agx-mesa-isa/mesa_isa.py`` from Mesa's
packer (``src/asahi/compiler/agx_pack.c``) and the operand values its compiler
creates: operand sizes per opcode, contiguous load and store masks, the
memory formats, the atomic operations (``translate_atomic_opcode``), the
special registers, the conversion modes and the long-form choices. The only
difference between G13 and G14 in Mesa's output is bit 45 of device atomics,
which Mesa sets for multi-cluster G13 parts and two-die parts
(``needs_g13x_coherency``); the ``multi-die-coherency`` feature follows the
same rule. ``tools/fuzz.sh`` checks the forms against Mesa's own packer: every
instruction Mesa packs from randomly built instructions is accepted.

The AGX Performance Simulator's ``--agx-strict GPU`` option applies the same
check to the code it runs.

What the mode costs (``llvm/utils/agx-bench/agx-bench.py --agx-strict``, the 8
benchmark kernels on ``agx-g13g`` and ``agx-g14c``): 6.0% more code (3142 to
3332 bytes, the long memory forms) and 0.4% more estimated cycles (16655 to
16719). All of the cycle difference is the ``reduce`` kernel on G14, whose
float atomic add becomes a compare-and-swap loop (1449.7 to 1514.4 cycles);
on G13, which has no native float atomic, the cycles are the same.

Debug information
-----------------

With debug locations, objects carry DWARF (``.debug_line`` and, for full
debug info, ``.debug_info``); addresses are offsets in ``.text``. Debug
sections use the two data relocations ``R_AGX_ABS32`` and ``R_AGX_ABS64``,
which ``llvm-dwarfdump``, ``llvm-objdump -l``, ``llvm-addr2line`` and
``llvm-symbolizer`` resolve, so these map an AGX PC to its source line. The
ZLUDA runtime ignores the debug sections when loading; the AGX Performance
Simulator's ``agx_run --lines`` names the source line of a faulting
instruction.

Checked mode
------------

The ``agx-checked`` feature (``-mattr=+agx-checked``, or the module flag
``!{i32 1, !"agx-checked", i32 1}``) checks memory accesses as the kernel
runs and reports the first bad one to the runtime, which turns it into an
error such as ``CUDA_ERROR_ILLEGAL_ADDRESS``. It is meant for debugging: the
hardware reports nothing about a bad access (shader faults on Asahi are soft:
loads from unmapped memory read 0 and stores are dropped), and the checks are
ordinary instructions, so the mode can be combined with ``agx-strict``.

Each check runs before its access:

============================  ================================================
Access                        Check
============================  ================================================
device load, store, atomic    the address is at least 4096 (null); it is a
                              multiple of the access size, up to 16 bytes
                              (alignment); and, when the pointer is based on a
                              kernel argument or a module global, the whole
                              access lies in the allocation that pointer was
                              in at launch (bounds)
threadgroup                   the access lies in its threadgroup array, or for
                              dynamic shared memory in the threadgroup memory
                              of the launch
stack (``alloca``)            a variable offset stays within the object
============================  ================================================

A pointer whose base cannot be traced to an argument or a global (one loaded
from memory, for example, or beyond the 16 bounds slots) is checked against
the runtime's *allocation table*: a binary search finds the allocation it
points into, which must hold the whole access. With no table (a runtime of
version 1, or ``-agx-checked-table=false``) it gets the null and alignment
checks only. The search costs about 80 instructions plus 20 per doubling of
the number of allocations, per check. Bases are
traced through address arithmetic, including the integer form front ends
like ZLUDA use (``inttoptr`` of a 64-bit parameter plus an offset).

Accesses in one basic block at constant offsets from the same base are
checked together, once, before the first of them, and an access whose bytes
an earlier check of its block covers is not checked again. A failure of such
a merged check is reported with the address and size of the whole range, and
before the earlier accesses of the range have run.

Accesses that cannot fail are not checked: constant offsets inside a device
global or a static threadgroup array of known size, aligned as the global is,
and stack accesses at constant offsets. A kernel with no other access is left
unchanged: no hidden arguments and no ``.agx.kernels`` extension 2, so the
runtime launches it as an unchecked kernel. MLIR memref descriptors are seen
through: a pointer read back from an ``insertvalue`` chain has the bounds of
the argument stored in it.

On the agx-bench kernels the mode costs about 1.5 times the cycles (from 1.1
times for compute-bound kernels to 5 times for very short ones) and 5 times
the code, mostly a fixed part per kernel.

A thread whose check fails writes the error record if no other thread has
(see below) and returns from the kernel; the rest of the launch continues.
The threadgroup barrier does not wait for threads that have returned
(measured on G14C). Every checked kernel first reads the error record and
returns at once if an error is already recorded, so that later launches do
not run on bad data.

**Runtime interface (version 1).** The backend adds hidden arguments to each
checked kernel, after the source arguments, passed in uniform registers like
the others: the 64-bit device address of the error record, the threadgroup
memory bytes of the launch (static and dynamic, 32-bit), and for each *bounds
slot* of the kernel two 64-bit addresses, the first and one past the last
byte of the allocation that the slot's pointer was in at launch. A pointer
that is not in any allocation gets its own value twice, so that any access
through it fails. At most 16 slots are used; a kernel with more pointer bases
checks the others for null and alignment only.

The kernel's ``.agx.kernels`` record gets the extension ``2, version, N,
record half, threadgroup bytes half``, then for each of the N slots ``argument,
offset, first half, last half``: the source argument the slot's pointer comes
from, the byte offset of the 64-bit pointer in that argument's bytes (or
``0xffffffff`` when the argument itself is the pointer), and the uniform halves
of its two bounds. The hidden arguments are not in the record's argument
table.

**Allocation table (version 2).** A kernel with checks against the table has
two more hidden arguments after the bounds: the 64-bit address of the table
and its 32-bit entry count. The table is ``count`` pairs of 64-bit addresses
``(first byte, one past the last byte)`` of every live allocation (device
memory and global data buffers), sorted by first byte, not overlapping, and
8-byte aligned. The record gets a last extension, ``3, table half, count
half``; a runtime of version 1, which stops at an extension it does not know,
leaves the count 0. The table must not change while a launch that uses it
runs (ZLUDA copies it into the launch's staging memory).

The *error record* is 64 bytes of device memory, 8-byte aligned, zeroed by
the runtime:

==========  =====  ===========================================================
Offset      Size   Field
==========  =====  ===========================================================
0           4      claimed: 0 while there is no error; the first failing
                   thread sets it to 1 with a compare-and-swap
4           4      kind (bits 0-7): 1 out of bounds, 2 null, 3 misaligned,
                   4 threadgroup out of bounds, 5 stack out of bounds; bit 8
                   set for a store, bit 9 for an atomic
8           8      the address (for threadgroup and stack accesses, the offset)
16          4      access size in bytes
20          4      check site (0 if unknown)
24          12     the thread's position in its threadgroup (x, y, z)
36          12     the threadgroup's position in the grid (x, y, z)
48          16     zero
==========  =====  ===========================================================

The fields are written after ``claimed``, so the runtime reads the record
after the launch has finished. It keeps the error until the context is reset
(and must then zero ``claimed``).

*Check sites* are described by the constant global ``__agx_checked_sites``
(in the module's global data), a text with one line per site:
``<site>\t<access> <size>\t<frame>[\t<frame>...]\n``, where the access is
``load``, ``store`` or ``atomic`` and each frame is ``<function>
<file>:<line>:<column>``, the innermost (inlined) one first, from the debug
locations. Without debug information the frame is the kernel's name.

ZLUDA's ``zluda_agx`` implements the runtime side (``zoc --agx-checked``,
``ZLUDA_AGX_CHECKED=1``; its ``docs/src/agx.md``, "Checked mode").
``llvm/utils/agx-checked/run_checked.py`` does the same for the AGX
Performance Simulator's ``agx_run``; ``checked_tests.py`` (IR kernels) and
``zluda_tests.py`` (PTX through ``zoc``) check on it that each bug is reported
with the right fields and that correct kernels are unchanged.

Occupancy attributes
--------------------

A core of G13 and G14 runs 2 * floor(3328 / (2 * H)) SIMD-groups of 32
threads, at most 1024 threads, when each thread uses H 16-bit registers,
rounded up to a multiple of 8 (registers are allocated in groups of 8 halves;
this is the occupancy table of Mesa's AGX compiler). All the threads of a threadgroup run
on one core, so the registers a kernel uses limit its threadgroup size: 52
32-bit registers allow 1024 threads, 128 allow 384. The resource comments
print the limit (``up to N threads per threadgroup``). Two function
attributes lower the register use, spilling the rest to the stack:

* ``"agx-max-threads-per-threadgroup"="N"`` (PTX ``.maxntid`` and
  ``.reqntid``, MLIR's ``known_block_size``): at most as many registers as let
  N threads fit. With N at most 128 the threadgroup is one aligned block of 4
  SIMD-groups, and on chips with ``block_barrier`` (G14, not ``agx-strict``)
  its threadgroup barriers become block barriers, without the workgroup
  fences around them (the block barrier orders the block's memory).
* ``"agx-max-gprs"="N"`` (PTX ``.maxnreg``): at most N 32-bit registers, and
  no fewer than 16.

Later generations are assumed to have the same register budget; this is not
measured.

Address spaces
==============

.. table:: AGX address spaces
   :name: agx-address-spaces-table

   ============= ============ ============= =================================
   Address space Name         Pointer size  Memory
   ============= ============ ============= =================================
   0             generic      64            Device memory
   1             device       64            Device memory
   3             threadgroup  32            Threadgroup (shared) memory
   4             constant     64            Device memory, read only
   5             private      32            Per-thread stack
   ============= ============ ============= =================================

Generic, device and constant pointers are interchangeable: casts between them
are no-ops. There is no generic addressing of threadgroup or private memory.

The backend's alias analysis (``agx-aa``, ahead of BasicAA in both pass
managers, and used by the DAG combiner and the machine scheduler) takes two
accesses in different non-generic address spaces never to alias, and the
constant address space never to be written: kernels do not store to kernel
arguments or to constant data (PTX ``.const``), as for NVPTX. It lets the
scheduler issue device loads ahead of threadgroup stores, so their latencies
overlap.

Kernel ABI
==========

Kernels use the ``agx_kernel`` calling convention and return ``void``. By
default, every other function is inlined (recursive ones after the lowering
in `Recursion`_), and the backend reports an error for calls that remain.
The experimental ``-agx-experimental-leaf-calls`` codegen option outlines
eligible leaf functions with up to 28 legalized 32-bit argument slots,
returning through the same register window. Calls are direct and non-nested;
the kernel and one leaf have separate bounded 4 KiB scratch regions. This
path is only verified by offline simulator fixtures and is not a general
device calling convention or a benchmark option. The option removes dead
device functions after inlining, before per-function machine-code generation.
A module without
``agx_kernel`` functions
compiles each external function as an entry point with the kernel ABI.

Kernel arguments are passed in uniform registers. They are packed in order in
16-bit units (``u0l``, ``u0h``, ``u1l`` ...), each argument aligned to its
size and 64-bit arguments to 32 bits; arguments are split into their legal
parts first (``i8`` and ``i1`` take 16 bits, vectors one unit per element).
The 32-bit values ``threadgroups_per_grid.x``, ``.y`` and ``.z`` follow the
last argument, aligned to 32 bits; ``llvm.agx.threadgroups.per.grid.*``
read them.

Before each kernel, the assembly printer emits comments with the resources
the driver must provide::

  // uniform registers: 18 x 16-bit (threadgroups_per_grid at u6)
  // general purpose registers: 4 x 32-bit
  // threadgroup memory: 0 bytes
  // stack size: 0 bytes

Arguments passed by reference (``ptr addrspace(4) byref(T)``, as ZLUDA emits
PTX ``.param`` arguments) whose every use is a load at a constant offset are
passed by value: their bytes occupy consecutive uniform registers (in 64-bit
parts and a last 32-bit one) and the loads read the registers. Other
by-reference arguments, and those that do not fit in what is left of the 1 KiB
uniform file (256 32-bit registers, all preloaded at no measured cost), stay a 64-bit
pointer to the bytes, which the driver places in memory.

The object also describes each kernel in the ``.agx.kernels`` section, for
drivers, as a sequence of records of little-endian 32-bit words:

==========================  ==============================================
Word                        Meaning
==========================  ==============================================
0                           size of the record in bytes
1                           version (1)
2                           length of the kernel's name
3 ...                       the name, padded with zeros to 4 bytes
then                        uniform halves; first half of
                            ``threadgroups_per_grid``; first half of the
                            global data buffer address (0xffffffff if
                            none); general purpose registers; threadgroup
                            memory bytes; stack bytes per thread
then                        the number of arguments of the source kernel,
                            and for each: kind (0 = its bytes are in the
                            uniforms, 1 = a pointer to them is), size in
                            bytes, first half (0xffffffff if unused)
then                        optional tagged extensions, up to the size of
                            the record; a reader stops at a tag it does not
                            know
==========================  ==============================================

The extensions are:

* ``1, x, y, z``: the kernel runs clusters of x * y * z CTAs as one
  threadgroup (function attribute ``agx-cluster-dims``).
* ``2, ...`` and ``3, ...``: checked mode (see "Checked mode").
* ``4, half``: the first half of the kernel argument with the parameter
  attribute ``"agx-hidden"="threadgroup-bytes"``, which is not a source
  argument: the driver sets it to the threadgroup memory of the dispatch, the
  kernel's threadgroup memory bytes plus the dynamic ones of the launch.
* ``5, half``: the driver sets the 64-bit uniform from ``half`` (always 0)
  to the address of the texture heap (see "Textures").
* ``6, n``, then ``n`` pairs ``half, value``: with ``-agx-uniform-constants``,
  the kernel reads its 32-bit constants too wide for an immediate from
  uniforms instead of materializing them with ``mov_imm``; the driver sets
  the 32-bit uniform from each ``half`` to its ``value``. They follow the
  arguments and ``threadgroups_per_grid``, within the uniform halves of the
  record. A driver that does not set them must not use the option, since a
  reader that skips the extension leaves the constants zero.
* ``7, n``, then ``n`` triples ``half, source half, kind``: with
  ``-agx-launch-divisors``, a 32-bit division or remainder by a kernel
  argument (or a 32-bit word of one) reads the argument's multiplier and
  shifts from uniforms, which the driver computes at launch from the 32-bit
  value ``d`` at ``source half`` (an argument it has just set) and puts in
  the 32-bit uniform from ``half``. With ``l = ceil(log2(d))``, kind 0 is
  ``2^32 * (2^l - d) / d + 1`` (truncated to 32 bits), kind 1 is
  ``min(l, 1)`` and kind 2 is ``max(l, 1) - 1``; kinds 3, 4 and 5 are the
  same of ``|d|`` (a signed divisor), and kinds 6, 7 and 8 the same of
  ``max(d, 1)`` (ZLUDA's guard of a PTX division's divisor). All are 0 for
  ``d = 0``. The kernel
  divides as ``q = (t + ((x - t) >> s1)) >> s2``, ``t = mulhi(x, m)``,
  exact for every 32-bit ``x`` (Granlund and Montgomery, "Division by
  Invariant Integers using Multiplication", PLDI 1994), in place of a
  reciprocal estimate and two corrections. As with extension 6, a driver
  that does not set them must not use the option.

Each kernel ends with ``stop`` followed by eight ``trap`` instructions, since
the instruction prefetcher reads past the end of the program.

Threadgroup variables are globals in address space 3 without an initializer.
They are allocated per kernel in threadgroup memory, those the kernel uses in
module order, each aligned to its alignment, then those of size zero (whose
memory is the dynamic threadgroup memory of the launch), in module order.

Device and constant variables (globals in address spaces 0, 1 and 4) live in
the *global data buffer*. When a module defines any, each kernel receives
the 64-bit address of that buffer in the uniform registers following
``threadgroups_per_grid`` (reported in the resource comments as ``global
data buffer at uN_uN+1``). The variables are laid out in module order, each
aligned to its alignment, and the initial contents of the buffer, with that
layout, are the ``.agx.globals`` section of the object. Initializers that
refer to the addresses of other globals have relocations in
``.rela.agx.globals``, against symbols defined in ``.agx.globals``; the loader
computes S + A with S the variable's address in the placed buffer:

=================  =====  ================================
Type               Value  Field
=================  =====  ================================
``R_AGX_ABS64``    1      64-bit word, S + A
``R_AGX_ABS32``    2      32-bit word, S + A (truncated)
=================  =====  ================================

There are no relocations in ``.text``, and references to globals of other
objects are not supported.

Textures
--------

``llvm.agx.texture.sample(i32 %offset, i16 %sampler, float %x, float %y,
i32 immarg %dim)`` samples a texture with the texture unit and returns its
four channels as ``<4 x i32>``: the bits of floats for float, unorm and snorm
formats, integers for integer formats. ``%dim`` is 0 (1D, ``%y`` unused) or 2
(2D). It is ``texture_sample`` in its bindless form: the 24-byte texture
descriptor is at ``%offset`` bytes from the *texture heap* address, and
``%sampler`` is an index into the sampler heap of the compute command
(``drm_asahi_cmd_compute.sampler_heap`` and ``sampler_count``: 8-byte sampler
descriptors). The level is chosen automatically (level 0 without
derivatives), so mipmapped textures are not supported. Both descriptors use
the layouts of Mesa's ``cmdbuf.xml`` (``Texture``, ``Sampler``). The result
is asynchronous like a load, followed by a wait before its first use, and
does not share registers with the sources.

The descriptor base of a bindless ``texture_sample`` must be a 64-bit uniform
register below the 64th at a multiple of four halves. If any function of a
module samples, every kernel of the module therefore receives the texture
heap address in ``u0_u1``, before its arguments (record extension 5, and
``texture heap at u0_u1`` in the resource comments): non-kernel functions
read the same uniforms.

Execution model
===============

A SIMD-group has 32 threads. There is no scalar unit and no scalar branch:
every branch is executed by the SIMD-group as a whole, under an execution
mask kept in the 16-bit nesting counter ``r0l`` of each thread (a thread is
active when its counter is zero). The code generator therefore structurizes
every function, uniform branches included, and annotates it with:

* ``llvm.agx.if``: ``if_icmp r0l, nueq, cond, 0, 1`` deactivates the threads
  where the condition is false, followed by ``jmp_exec_none`` to skip the
  region when no thread remains. The jump (here and after ``else``) is left
  out when the region is a straight run of at most 4 ALU instructions
  (``-agx-max-unskipped-instrs``): on G14C an ALU instruction issued with no
  active thread costs about as little as the taken jump over it, while
  masked loads, stores and their waits still cost much of their latency
  (B31).
* ``llvm.agx.else``: ``else_fcmp r0l, eq, 0.0, 0.0, 1`` swaps the threads of
  the innermost level.
* ``llvm.agx.end.cf``: ``pop_exec r0l, 1`` reactivates the threads of the
  innermost construct. It is emitted at the start of the reconvergence
  block.
* ``llvm.agx.loop.begin``: ``push_exec r0l, 1`` in the loop preheader.
* ``llvm.agx.loop``: ``while_icmp r0l, ueq, break, 0, 1`` at the latch
  deactivates the threads leaving the loop, and ``jmp_exec_any`` branches back
  while any thread remains.

The nesting counter is zeroed at the start of each kernel that uses control
flow. Blocks are laid out in the reverse post-order of the structured graph,
so constructs fall through into each other as in Mesa's output.

Loads and stores of 2 to 4 consecutive 32-bit values (vectors, and
neighbouring scalar accesses the load/store vectorizer merges) in device or
threadgroup memory are one ``xy`` / ``xyz`` / ``xyzw`` access at any 4-byte
alignment, which the M2 Max handles exactly at every offset, across 64-byte
lines and pages (ptx-re-blockers B33). The vectorizer runs after checked mode,
so the checks see the accesses the source wrote. Vector arithmetic is still
scalar.

Threadgroup and private addresses that PTX computes in integers (``inttoptr``
of an integer expression) are first written as a constant offset from the
address without its constant terms, rebuilt from the instructions already
there (``AGXThreadgroupAddresses``, ``-agx-threadgroup-address-offsets``), so
that the vectorizer sees ``a[k]`` and ``a[k + 1]`` as one base at offsets 0
and 4. Loops with a known trip count whose threadgroup loads read the 4 or 8
bytes by which their address advances each iteration are unrolled until those
reads cover 16 bytes (``-agx-threadgroup-unroll``: 0, the default, for that;
1 for no unrolling; N for up to N copies of every single-block loop that
loads threadgroup memory): the inner loop of a matrix product over a tile
then loads ``xyzw``.

In a loop that writes no memory and exits only at its latch, a device load in
a block that runs exactly when ``j != i`` (``i`` loop-invariant, the test run
every iteration) moves to the loop header when the same address with ``i``
for ``j`` was loaded before the loop (``AGXHoistGuardedLoads``,
``-agx-hoist-skipped-loads``): an all-pairs loop that skips its own element
then issues the loads of its unrolled copies together instead of one after
another.

Memory loads, stack loads and atomics complete asynchronously. A ``wait`` is
inserted before the first instruction that reads or overwrites a pending
result, before barriers and at the end of each block.

Registers
=========

``r0`` to ``r127`` are the 32-bit general purpose registers, with 16-bit
halves ``rNl`` and ``rNh``. ``r0`` is reserved for the nesting counter. 64-bit
ALU values use aligned register pairs; memory operations accept any register
tuple. ``u0`` to ``u255`` are the uniform registers. They are never allocated;
copies from them are folded into the ALU sources and memory bases that can
read them (64-bit ALU sources cannot be uniform).

Code generation
===============

Supported:

* 16, 32 and 64-bit integer arithmetic, including 64-bit multiplication,
  shifts, comparisons and bit counting. Division is expanded in IR, except
  by constants. 64-bit additions are single instructions; 64-bit logic,
  selects, shifts, comparisons and products are lowered on the 32-bit halves
  (split with ``truncate`` and ``AGXISD::HI32``, joined with ``BUILD_PAIR``),
  so known or constant halves fold. Shifts by a variable amount use the 7-bit
  shift amounts of ``bfi``, ``extr`` and ``asr`` (a logical right shift is
  two ``extr``), as documented by dougallj's applegpu for G13; G14 is to be
  confirmed (ptx-re-blockers B47).
* Half and single precision arithmetic, with source modifiers and 8-bit
  floating point immediates folded. Division uses ``rcp`` and a multiply
  with a Newton-Raphson step (omitted with ``arcp``/``afn`` and for
  ``half``); ``sin``/``cos`` reduce to quadrants for ``sin_pt_1`` and
  ``sin_pt_2``; ``exp``, ``log`` and ``pow`` go through ``exp2``/``log2``.
  ``llvm.agx.sin``, ``llvm.agx.cos``, ``llvm.agx.exp2`` and ``llvm.agx.log2``
  are the code of ``llvm.sin`` and the others (with the same IEEE lowering),
  which LLVM does not evaluate on the host for a constant argument: a
  constant gives the bits of the hardware approximation, as a variable does
  (``llvm.agx.rcp`` and ``llvm.agx.rsqrt.special`` likewise).
  Conversions between 64-bit integers and ``float`` are expanded.
* Device memory loads and stores of 8 to 64 bits with folded base, scaled
  index and constant offsets; threadgroup and stack memory; memory
  intrinsics (inline or as loops).
* Device and threadgroup 32-bit atomics. Other atomic operations become
  compare-and-swap loops; atomic loads and stores become atomic operations.
  The hardware has no 64-bit or 128-bit atomics (beyond 64-bit ``umin`` and
  ``umax`` without a result): those take a lock from a table of 1024 device
  memory locks (the ``__agx_atomic_locks`` global, in the global data
  buffer), chosen by the address of the 16-byte block they access, and read
  and write the value with 32-bit atomics between fences while they hold it.
  The critical section is in the loop that takes the lock, so the threads of
  a SIMD-group that wait for the same lock do not deadlock. They are atomic
  with respect to each other, not to 32-bit atomics on the same memory.
* SIMD-group and quad shuffles, ballots, reductions and prefixes, derivatives,
  barriers and special registers through the ``llvm.agx.*`` intrinsics.

There is no floating point environment: constrained floating point
intrinsics are compiled as the ordinary operations, with round to nearest
even. 32-bit operations always flush denormals. 16-bit operations keep them;
in functions whose ``denormal_fpenv`` flushes ``half`` denormals, the
operands and results of ``half`` arithmetic, comparisons and conversions are
flushed explicitly.

Fences and ordered atomics follow the barriers measured with cross-core
litmus tests on G14C (M2 Max). ``memory_barrier`` with operands ``a, b, c``
is the 16-bit word ``0xf5 | b << 8 | a << 10 | c << 12``:

.. table:: AGX fences
   :name: agx-fences-table

   ===================================== ====================================
   Fence                                 Barriers
   ===================================== ====================================
   ``singlethread``                      none
   ``workgroup``, any ordering           ``f596`` (threads of a threadgroup
                                         share a core; no reordering was
                                         observed there)
   device or system, ``release``         ``f533`` (the store release)
   device or system, ``acquire``         ``wait 0``, ``f59d``
   device or system, ``acq_rel``         ``f533``, ``wait 0``, ``f59d``
   device or system, ``seq_cst``         Mesa's sequence ``f596 f533 f59a
                                         f59e f503 f599``, ``wait 0``,
                                         ``f59d`` (the only one that orders
                                         store buffering in every case)
   ===================================== ====================================

``f596`` alone, which Mesa uses below queue-family scope, does not order
stores between cores, and a device atomic is no release by itself. Ordered
atomic operations are monotonic accesses between the fences of their scope
(a release fence before, an acquire fence after). The release costs about
25 ns before a store and 250 ns after a load; the sequentially consistent
sequence costs 0.4 to 0.65 us.

Not supported: ``frem`` and vector arithmetic on ``double`` (see `Double
precision`_; diagnosed as unsupported), textures and graphics shaders, calls
to functions that are not defined in the module, recursion through variadic
functions or ``musttail`` calls, and dynamic ``alloca`` in recursive functions.

The default backend does not use hardware call and return instructions: all
functions are inlined into the kernels. The opt-in direct leaf-call path is
described in `Kernel ABI`_; its emitted code has offline simulator coverage
but has not been validated as a compiler-generated ABI on the GPU. G14C
direct, register-indirect, and divergent call/return behavior has been
measured separately (see the Apple M2 Playground call/return handoff). An
indirect call becomes a test of the callee against each address-taken
function of the module with the same type, with a direct call (inlined) for
each, and a trap if none matches. The address of such a function, as a
value (stored, compared or passed), is its number in the module (1, 2, ...).

Recursion
---------

Each cycle of the call graph (after indirect calls became direct ones)
becomes, for each of its functions called from outside the cycle, one
function that runs the whole cycle over an explicit stack of frames in
private memory, and is then inlined like any other:

* A function's arguments, its ``alloca``\ s and the values it uses outside
  their basic block (its blocks are split after each call into the cycle)
  are fields of its frame, after a header of two 32-bit words: the number of
  the call site to return to, and the caller's stack pointer. Frames are
  16-byte aligned.
* A call stores the arguments in the callee's frame, which follows the
  caller's, and branches to the callee's entry; a return stores the return
  value in a slot of the function, pops its frame and branches on the call
  site number to the code after the call (or out, for 0).

The control flow passes make the result structured, so the threads of a
SIMD-group may recurse to different depths. The stack is
``-agx-recursion-stack-size`` bytes per thread (by default four times the
module flag ``!{i32 2, !"agx-stack-size", i32 N}``, else 4096), part of the
kernel's private memory. A thread whose call would pass its end stores 1 to
``__agx_stack_overflow``, a 32-bit global of device memory that is 0 at load,
and ends; the runtime reports the launch as failed (ZLUDA's ``zluda_agx``
gives ``CUDA_ERROR_LAUNCH_FAILED``) and clears the word.

.. _agx-apple9:

Apple9 (G16/G17)
================

The G16 and G17 GPUs use a different instruction encoding, of which too
little is known to generate code reliably. The processors are defined so that
features and scheduling can be selected, but instruction emission reports
``no AGX instruction encoding for ... (the Apple9 encoding is not yet
supported)``. Pseudo instructions carry an ``Apple9Real`` field for the
real instruction of that family, to be filled in as encodings are confirmed.

Intrinsics
==========

See ``llvm/include/llvm/IR/IntrinsicsAGX.td``. The ``float`` and ``half``
to 32- and 16-bit integer ``convert`` saturates and converts NaN to 0
(measured on G14C, B27), so ``llvm.fptosi.sat`` and ``llvm.fptoui.sat`` to
``i32`` and ``i16`` are a single ``convert``; narrower saturation widths are
clamped first. The special register
intrinsics (``llvm.agx.thread.position.in.grid.x`` and so on) read
``get_sr`` registers, ``llvm.agx.read.sr`` reads any special register by
number, and ``llvm.agx.simd.reduce``/``prefix`` and
``llvm.agx.quad.reduce``/``prefix`` take the operation as an immediate:

.. table:: SIMD-group reduction operations
   :name: agx-simd-ops-table

   ===== ========= ===== =========
   Value Operation Value Operation
   ===== ========= ===== =========
   0     and       16    iadd
   1     fadd      20    smin
   2     or        22    smax
   3     fmul      28    umin
   4     xor       30    umax
   5     fmin
   7     fmax
   ===== ========= ===== =========

Strict floating point
---------------------

The 32-bit floating point unit flushes denormal operands and results to zero,
its reciprocal and reciprocal square root are approximations (within 1 ulp),
and NaN results have the hardware's bit patterns.

Without the features below, ``fdiv`` on ``float`` (without ``arcp`` or
``afn``) and ``llvm.sqrt`` (without ``afn``) are still correctly rounded, with
denormal operands and results flushed (``AGXISelLowering.cpp``). Division
divides the significands, a in [1, 2) and d in [2, 4) (one ``bfi`` each sets
the exponent field), with the refined reciprocal y = y0 + y0 (1 - d y0), y0 =
``rcp(d)``, which is the correctly rounded 1 / d for every significand of the
recorded ``rcp``, and one correction q = q0 + (a - d q0) y of q0 = a y,
correctly rounded by Markstein's theorem; the exponents come back as two
power-of-two factors that also give zeros and infinities. ``sqrt`` corrects s0
= x ``rsqrt_special(x)`` once, s = s0 + (x - s0^2) ``rsqrt_special(x)`` / 2,
which is correctly rounded for every significand (x below 2^-80 is scaled by
2^64 so that the remainder is not flushed). The hardware's ``rcp`` and
``rsqrt_special`` are those recorded for every significand on an M2 Max (AGX
Performance Simulator, ``testsuite/transcendental``). A division takes 19
instructions (1 / x 9) and a square root 13, constants included.

``frem`` on ``float`` and ``half`` is an exact ``fmod``
(``AGXCodeGenPrepare.cpp``): a loop reduces |a| by |b| 2^j while the quotient
could reach 2^24, then a last step by |b| truncates the correctly rounded
quotient and keeps whichever of a - q |b| and a - (q - 1) |b| lies in
[0, |b|).

Two features, off by default, implement IEEE 754 and PTX behavior in software
(``AGXStrictFP.cpp``):

``ieee-fp32``
  In functions whose ``denormal_fpenv`` for ``float`` is IEEE (the default):

  * ``fadd``, ``fsub``, ``fmul`` and ``llvm.fma``/``llvm.fmuladd`` keep the
    hardware instruction, and recompute the result exactly with integer
    arithmetic in a conditional block where it can be wrong: an addition
    whose operands are both below 2\ :sup:`-101`, a multiplication or fused
    multiply-add with a denormal operand or a result of at most
    2\ :sup:`-125`.
  * ``fdiv`` (without ``arcp`` or ``afn``) and ``llvm.sqrt`` (without ``afn``)
    are correctly rounded: the mantissas are divided (or the root taken) with
    refined hardware estimates, and corrected until the exact FMA remainder
    shows that the result is correctly rounded. Operands or results outside
    the normal range use integer arithmetic.
  * Comparisons, ``minnum``/``maxnum``/``minimum``/``maximum`` (and the
    ``num`` variants) compare zeros and denormals as integers, and rounding
    functions, ``llvm.agx.rcp``, ``llvm.agx.rsqrt`` and ``llvm.log2`` handle
    denormal operands or results explicitly. ``llvm.exp2`` computes
    2\ :sup:`f` in hardware for the fraction f of x and scales it by
    2\ :sup:`floor(x)` exactly, so that integral operands give exact powers of
    2 and denormal results are rounded once; ``llvm.sin`` and ``llvm.cos`` of
    operands below 2\ :sup:`-12` are x and 1 (keeping the sign of zero and
    denormal operands). They are otherwise the hardware approximations.

  In functions that flush ``float`` denormals, ``fdiv`` and ``llvm.sqrt`` are
  correctly rounded in the same way, on flushed operands and with a flushed
  result.

  ``bfloat`` arithmetic (which keeps denormals) is computed as ``float``
  arithmetic on the extended operands with the IEEE lowering, rounded back
  once; ``float`` has more than twice the precision, so the result is
  correctly rounded. ``llvm.fma`` on ``bfloat`` is computed exactly with
  integer arithmetic (see `Double precision`_), since a ``float`` fma would
  round twice.

``!agx.keep-denormals``
  Independently of the features, a ``float`` operation carrying this metadata
  (an empty node) gets the ``ieee-fp32`` lowering above in any function, also
  one that flushes denormals: a front end's IEEE operation in flushing code,
  such as the addition of an atomic add that keeps denormals in a
  compare-and-swap loop.

``canonical-nan``
  NaN results of floating point arithmetic and conversions are ``0x7fffffff``
  (``0x7fff`` for ``half`` and ``bfloat``); minimum and maximum keep the bits
  of a non-NaN operand. This costs one instruction per operation.

Measured on the functional simulator, per SIMD-group and operation with
normal operands, ``ieee-fp32`` adds about 8 instructions to an addition, 12
to a multiplication, 22 to a fused multiply-add, 13 to a comparison, 65 to a
division and 39 to a square root.

Double precision
----------------

The hardware has no double precision arithmetic. The ``agx-soft-f64`` pass
replaces the ``double`` operations with integer arithmetic on their bits:
``fadd``, ``fsub``, ``fmul``, ``fdiv``, ``fneg``, ``fcmp``, ``fpext`` and
``fptrunc`` (from and to ``float``, ``half`` and ``bfloat``), ``sitofp``,
``uitofp``, ``fptosi``, ``fptoui``, and ``llvm.fma``, ``llvm.fmuladd``,
``llvm.sqrt``, ``llvm.fabs``, ``llvm.copysign``, the minimum and maximum
intrinsics, the rounding intrinsics (``floor``, ``ceil``, ``trunc``,
``rint``, ``nearbyint``, ``roundeven``, ``round``), ``llvm.canonicalize``,
``llvm.is.fpclass`` and the saturating conversions. Each operand becomes an
exact sign, 128-bit significand and exponent, the exact result (with a sticky
bit for the bits dropped) is computed with integer operations, and a single
rounding gives the result, so all operations are correctly rounded IEEE 754
operations with denormals, in the four rounding directions (see `Rounding
modes`_; the ``double`` rounding mode is the ``half`` field). Significands are 64-bit, with 128-bit
products in multiplications and fused multiply-adds. Division and the square
root have no loop: a ``llvm.agx.rcp`` or ``llvm.agx.rsqrt`` estimate, lowered
by 2^-19 and taken as a fixed-point value, is refined by two Newton steps on
64-bit and 128-bit integers to a quotient or root within 2 units of the
exact 55-bit one, which the exact remainder then corrects. The estimate may
be off by up to about 2^-19.7 (some 10 units in the last place of ``float``)
without changing a result. NaN results are ``0x7fffffffffffffff`` with
``canonical-nan``, and the first NaN operand, quieted, otherwise. Conversions
to integers saturate, and NaN converts to 0.

Values that are only moved (loads, stores, ``phi``, ``select``, arguments)
stay ``double`` and become ``i64`` in type legalization. ``frem`` and vector
arithmetic on ``double`` are diagnosed as unsupported. The code is large: in
instructions per operation, about 215 for an addition or a multiplication,
430 for a fused multiply-add, 290 for a division or a square root, 100 for a
conversion and 30 for a comparison.

Vector floating point arithmetic is scalarized before the floating point
lowerings (the Scalarizer pass), and constrained vector operations are rounded
lane by lane, so the strict modes, the rounding modes and the software double
precision apply to vectors too. A ``fneg`` or ``fabs`` of ``float`` that is
not a source modifier of another operation is a bit operation, and the
results of ``float`` minimum and maximum are flushed in functions that flush
``float`` denormals (the hardware returns a denormal operand unchanged).
Conversions of ``float`` to ``bfloat`` round to nearest even.

Rounding modes
--------------

The hardware rounds to nearest even only. ``llvm.agx.set.fpmode(i32)`` and
``llvm.agx.get.fpmode()`` access an emulated floating point mode register:

  ===== ======================================================
  Bits  Meaning
  ===== ======================================================
  0-1   ``float`` rounding: 0 toward zero, 1 to nearest even,
        2 upward, 3 downward (the ``FLT_ROUNDS`` numbering)
  2-3   ``half`` (and ``double``) rounding, same encoding
  4     ``float`` denormals are flushed
  5     ``half`` (and ``double``) denormals are flushed
  ===== ======================================================

It holds round to nearest even and the function's ``denormal_fpenv`` at
entry, and is kept in SSA values by the ``agx-lower-fp-env`` pass, which also
lowers the constrained floating point intrinsics
(``llvm.experimental.constrained.*``). Those that round to nearest even,
statically or through the register, become ordinary operations. The others
(``fadd``, ``fsub``, ``fmul``, ``fdiv``, ``fma``, ``sqrt``, ``fptrunc`` to
``half``, ``sitofp``/``uitofp`` and the rounding functions) are computed in
software and correctly rounded in their direction, with the overflow result
of that direction (the largest finite value or infinity) and the sign of an
exact zero sum that it requires. An operation whose denormal mode differs
from its function's flushes its operands and result explicitly. The software
paths do not depend on the ``ieee-fp32`` feature.

SIMD-group matrices
-------------------

``llvm.agx.simdgroup.matrix.fma(a, b, c)`` computes D = A * B + C on 8x8
matrices spread over the SIMD-group, of ``half``, ``bfloat`` or ``float``
elements. Thread t holds the elements (r, c) and (r, c + 1) of each matrix
in a ``<2 x T>``, with r = 4 * (t / 16) + (t / 2) % 4 and
c = 4 * ((t / 8) % 2) + 2 * (t % 2). It is computed with shuffles (each
thread gathers its row of A and its two columns of B, 24 shuffles; four times
as many on subtargets with ``quad-shuffle-index-merge``) and 16 fused
multiply-adds per thread, in order of k starting from C. On G14 the matrix
unit (``simd_matrix_fmadd32`` and ``simd_matrix_fmadd16``, the
``llvm.agx.simd.matrix.fmadd32`` and ``fmadd16`` intrinsics on the ``i64`` or
``i32`` of each ``<2 x T>``) computes the same chain of fused multiply-adds
for the same layout, ``float`` flushing denormals like the ALU and ``half``
keeping them; it is used for ``float`` unless the function has IEEE float
denormals with ``ieee-fp32``, and for ``half`` when the function has IEEE
half denormals, without ``canonical-nan``. Its operands are register pairs
(``r0_r1``, or the halves ``r0l_r0h`` of one register), encoded as their
first register. ``bfloat``, other GPUs and the matrix unit of Apple9 GPUs use
the shuffles.

SIMD-group shuffles
-------------------

``llvm.agx.simd.shuffle(v, i)`` returns ``v`` of thread ``i`` of the
SIMD-group and ``llvm.agx.quad.shuffle(v, i)`` of thread ``i`` of the quad.
The ``xor``, ``up`` and ``down`` variants read the thread at the current
thread index xor, minus or plus ``i``; with ``up`` and ``down``, threads whose
source is outside the SIMD-group (or quad) keep their own value.

On G13 to G15 (``quad-shuffle-index-merge``) the shuffle instructions read
the index of all four threads of a quad, including inactive threads, and use
the OR of their low two bits for the whole quad. The backend therefore:

* expands shuffles whose index is divergent (according to the uniformity
  analysis) into four shuffles, one per quad position, whose low index bits
  are the same in every thread, and selects the result of the thread's
  position; relative shuffles with divergent offsets become shuffles of the
  computed thread;
* passes register indices in ``r0h``, which is zeroed at kernel entry and
  after each shuffle, so that inactive threads contribute zero bits.

Immediate and uniform register indices are used directly.

AIR input
=========

The backend compiles AIR, Apple's LLVM IR for its GPUs, in the form the AIR
target's portable passes leave it (``addAIRPortablePasses``,
``llc -mtriple=air -air-portable``): a module whose kernels are described by
``!air.kernel``. The ``agx-lower-air`` pass, first in the backend's IR passes
and at the start of ``opt``'s pipelines, turns such a module into an AGX
module; it does nothing to a module without ``!air.kernel``. It is not a
reader of ``.air`` files.

The module may only use what AIR has: integers of 8, 16, 32 and 64 bits (and
``i1`` outside memory), ``half``, ``bfloat`` and ``float``, and pointers in
AIR's address spaces, with no casts between them. ``double``, other types,
``i1`` in memory and address space casts are errors, as are kernel
parameters that ``!air.kernel`` does not describe, argument buffers and more
than one threadgroup buffer.

.. table:: AIR address spaces on AGX

   ============= ======================= =================
   AIR           Memory                  AGX address space
   ============= ======================= =================
   0             thread                  5 (private)
   1             device                  1
   2             constant                4
   3             threadgroup             3
   ============= ======================= =================

Each kernel becomes an ``agx_kernel`` whose arguments are its device and
constant buffers, in location order (``air.location_index``, which must
follow the parameter order). A constant buffer that the kernel only loads
from, at constant offsets within the size ``!air.kernel`` gives it
(``air.arg_type_size``, typed by ``air.arg_type_name``, or bytes for a
structure), is passed by reference (``byref``), so that its bytes are
uniform registers; the AIR target's portable passes give each scalar kernel
argument such a buffer. Other buffers are pointers. The threadgroup buffer is
the launch's dynamic threadgroup memory.

System-value parameters become values computed at the kernel's entry:

==================================== ==========================================
AIR system value                     AGX
==================================== ==========================================
``thread_position_in_grid``          ``llvm.agx.thread.position.in.grid.*``
``threadgroup_position_in_grid``     ``llvm.agx.threadgroup.position.in.grid.*``
``thread_position_in_threadgroup``   ``llvm.agx.thread.position.in.threadgroup.*``
``threads_per_threadgroup``          ``llvm.agx.threads.per.threadgroup.*``
``threadgroups_per_grid``            ``llvm.agx.threadgroups.per.grid.*``
``threads_per_grid``                 threadgroups per grid times threads per
                                     threadgroup (grids are whole threadgroups)
``thread_index_in_threadgroup``      ``llvm.agx.thread.index.in.threadgroup``
``thread_index_in_simdgroup``        ``llvm.agx.thread.index.in.simdgroup``
``simdgroup_index_in_threadgroup``   ``llvm.agx.simdgroup.index.in.threadgroup``
``threads_per_simdgroup``            32
``simdgroups_per_threadgroup``       threads per threadgroup / 32, rounded up
==================================== ==========================================

Calls to AIR functions become LLVM intrinsics and instructions:

* the math functions by the inverse of the AIR target's mapping
  (``llvm/lib/Target/AIR/AIRIntrinsicMappings.td``, generated into this
  target): ``air.sin.f32`` is ``llvm.sin.f32``, ``air.fast_sin.f32`` the same
  with ``afn``, ``air.min.u.i32`` is ``llvm.umin.i32``;
* ``air.atomic.<space>.<op>`` and ``air.atomic.<space>.cmpxchg.weak.i32`` are
  ``atomicrmw`` and ``cmpxchg`` in their memory order, with workgroup scope
  for threadgroup memory and ``thread_scope_threadgroup``, volatile when AIR
  says so; the compare-exchange leaves the value it found in its expected
  operand;
* ``air.atomic.fence`` is a ``fence``; an atomic between two ``seq_cst``
  fences (AIR's ordered atomic, which is relaxed) that carries
  ``!agx.atomic.ordering`` (the order AIR's portable passes lowered it from)
  is that ordered atomic, without the fences;
* ``air.wg.barrier`` is ``llvm.agx.threadgroup_barrier``, between
  workgroup-scope release and acquire fences when it orders memory, and
  ``air.simdgroup.barrier`` a workgroup-scope fence (SIMD-groups run in
  lockstep);
* the SIMD-group and quad-group functions are AGX's: shuffles (and
  ``broadcast``), ``shuffle_and_fill_up``/``down``, reductions and scans by
  ``AGX::SIMDOp`` (an inclusive scan adds the element to the hardware's
  exclusive one), ballots widened to AIR's mask type, and ``vote_any`` /
  ``vote_all`` on the mask;
* the OpenCL-style work-item functions (``air.get_global_id`` ...), ``convert``
  (float to integer saturating), ``dot``, ``pack`` / ``unpack`` of
  ``unorm`` / ``snorm`` ``4x8`` and ``2x16``, ``sincos``, ``fract``,
  ``rsqrt``, ``fmod`` and ``mul_hi`` are expanded; device-coherent loads and
  stores are volatile, which the backend encodes coherent (with the barriers
  that makes host-coherent, more than device coherence needs).

* the SIMD-group matrices (``air.simdgroup_matrix_8x8_load``, ``store``,
  ``multiply_accumulate``, ``multiply``, ``init_filled``, ``init_diagonal``)
  are fragments in the layout of ``llvm.agx.simdgroup.matrix.fma`` (see
  `SIMD-group matrices`_): a ``<64 x T>`` becomes the thread's ``<2 x T>``,
  loads and stores move its two elements at
  (c + j + origin.x) * stride.x + (r + origin.y) * stride.y, a
  multiply-accumulate of ``half`` or ``bfloat`` matrices into ``float`` ones
  extends the operands first, and element-wise operations, AIR's and LLVM's
  element-wise functions (``air.fma``, ``llvm.maxnum``), ``fpext`` /
  ``fptrunc``, selects and phis apply to the fragment. Other uses of a
  ``<64 x T>``, which would depend on how AIR orders its elements, are
  errors.

Any other AIR function is an error. The math functions are as accurate as the
backend's lowering of the LLVM intrinsics. Without ``afn`` (and without
``ieee-fp32``, whose AGXStrictFP has its own exact forms), float ``sin``,
``cos``, ``exp``, ``exp2``, ``exp10``, ``log``, ``log2``, ``log10`` and
``pow`` are built in software (``AGXMathExpansion.cpp``: the argument reduced
exactly, Cephes' single precision polynomials; within 1 unit in the last place
on the AIR hardware suite's inputs), where the hardware's approximations,
which the ``afn`` forms keep, are tens of units off; ``sin`` and ``cos`` use
the hardware's reduction beyond 2^22. ``tan``, the inverse and hyperbolic
functions and their inverses, and ``sinpi``/``cospi``/``tanpi`` are built the
same way (half in float). ``sqrt`` and division are correctly rounded (see
`Strict floating point`_). The module then takes AGX's triple and data
layout, and loses the AIR metadata and module flags; address
computations over types that AIR lays out differently (a ``<3 x float>``
takes 16 bytes in AIR, 12 in AGX) become byte offsets with AIR's layout
first, and their stack slots byte arrays of AIR's size.

MLIR
====

MLIR targets AGX through AIR (``mlir/docs/Dialects/AIR.md``):
``-convert-gpu-to-air`` lowers the ``gpu`` dialect and the ``air`` dialect's
SIMD-group matrices to AIR, ``-air-attach-target`` attaches ``#air.target``
attributes, and ``-gpu-module-to-binary`` compiles a module whose target names
an AGX chip through AIR's portable passes and this backend (``agx-lower-air``,
above) to an AGX object.

.. code-block:: console

  $ mlir-opt kernel.mlir --pass-pipeline='builtin.module(air-attach-target{chip=agx-g14g},gpu.module(arith-emulate-f64,convert-gpu-to-air),gpu-module-to-binary)'

Host code from ``gpu-to-llvm`` calls MLIR's GPU runtime wrappers (``mgpu*``).
With ``-DMLIR_ENABLE_AGX_RUNNER=ON -DMLIR_AGX_LIBCUDA=<ZLUDA's
libnvcuda_agx.so>``, ``mlir_agx_runtime`` is ``CudaRuntimeWrappers.cpp`` built
without a CUDA toolkit (``lib/ExecutionEngine/AGXCudaCompat/cuda.h`` declares
the part of the driver API it uses) against ZLUDA's CUDA driver for Asahi
Linux, so ``mlir-runner --shared-libs=libmlir_agx_runtime.so`` runs
``gpu.launch_func`` on the GPU with ZLUDA's tracing: launch records and fault
dumps, and with ``#air.target<chip = ..., features = "+agx-checked">`` checked
mode, whose reports name the ``.mlir`` lines when the kernels have debug
locations (``-ensure-debug-info-scope-on-llvm-func`` and
``-mlir-print-debuginfo`` between pipelines). ``test/Integration/GPU/AGX``
needs the hardware; ``test/Integration/Dialect/AGX/checked.mlir`` runs the
same checked kernel shape on the AGX Performance Simulator.
