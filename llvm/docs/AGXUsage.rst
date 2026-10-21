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
from memory, for example) gets the null and alignment checks only. Bases are
traced through address arithmetic, including the integer form front ends
like ZLUDA use (``inttoptr`` of a 64-bit parameter plus an offset).

Accesses in one basic block at constant offsets from the same base are
checked together, once, before the first of them, and an access whose bytes
an earlier check of its block covers is not checked again. A failure of such
a merged check is reported with the address and size of the whole range, and
before the earlier accesses of the range have run.

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
  ``.reqntid``): at most as many registers as let N threads fit.
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

Kernel ABI
==========

Kernels use the ``agx_kernel`` calling convention and return ``void``. There
are no calls: every other function is inlined, and the backend reports an
error for calls that remain. A module without ``agx_kernel`` functions
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
by-reference arguments, and those beyond 256 bytes per kernel, stay a 64-bit
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
==========================  ==============================================

Each kernel ends with ``stop`` followed by eight ``trap`` instructions, since
the instruction prefetcher reads past the end of the program.

Threadgroup variables are globals in address space 3 without an initializer.
They are allocated per kernel in threadgroup memory, at offsets in order of
first use.

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
  by constants.
* Half and single precision arithmetic, with source modifiers and 8-bit
  floating point immediates folded. Division uses ``rcp`` and a multiply
  with a Newton-Raphson step (omitted with ``arcp``/``afn`` and for
  ``half``); ``sin``/``cos`` reduce to quadrants for ``sin_pt_1`` and
  ``sin_pt_2``; ``exp``, ``log`` and ``pow`` go through ``exp2``/``log2``.
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
precision`_; diagnosed as unsupported), vector memory
operations (vectors are scalarized), textures and graphics shaders, calls
to functions that are not defined in the module, and recursion.

The backend does not use the hardware call and return instructions: all
functions are inlined into the kernels. (G14C does have direct and
register-indirect calls with buffer-relative return targets, but only
calls by a whole, fully active SIMD-group have been measured so far; the
divergent cases that code generation would need are still open, see the
AGX Performance Simulator's ``tasks/unsafe-01-call-ret.md``.) An
indirect call becomes a test of the callee against each address-taken
function of the module with the same type, with a direct call (inlined) for
each, and a trap if none matches. The address of such a function, as a
value (stored, compared or passed), is its number in the module (1, 2, ...).

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
division and square root are not correctly rounded, and NaN results have the
hardware's bit patterns. Two features, off by default, implement IEEE 754 and
PTX behavior in software (``AGXStrictFP.cpp``):

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
products in multiplications and fused multiply-adds; division is a restoring
division and the square root a digit by digit root, each a loop of 55
iterations of 64-bit operations. NaN results are ``0x7fffffffffffffff`` with
``canonical-nan``, and the first NaN operand, quieted, otherwise. Conversions
to integers saturate, and NaN converts to 0.

Values that are only moved (loads, stores, ``phi``, ``select``, arguments)
stay ``double`` and become ``i64`` in type legalization. ``frem`` and vector
arithmetic on ``double`` are diagnosed as unsupported. The code is large: in
instructions per operation, about 290 for an addition, 335 for a
multiplication, 670 for a fused multiply-add, 280 for a division and 230 for
a square root (plus their 55 loop iterations of about 10 instructions), 140
for a conversion and 20 for a comparison.

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
multiply-adds per thread, in order of k starting from C; the matrix unit of
Apple9 GPUs is not used yet.

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

MLIR
====

MLIR targets AGX the way it targets AMDGPU:

* The ``agxir`` dialect wraps the ``llvm.agx.*`` intrinsics one to one, like
  ``rocdl``. Kernels carry the ``agxir.kernel`` attribute and use the
  ``agx_kernelcc`` calling convention.
* The ``agx`` dialect provides higher level operations, like ``amdgpu``:
  ``agx.shuffle`` (any type, SIMD-group or quad scope), ``agx.reduce``,
  ``agx.ballot`` and ``agx.threadgroup_barrier``. ``-convert-agx-to-agxir``
  lowers them.
* ``-convert-gpu-to-agxir`` lowers the ``gpu`` dialect, and
  ``-agxir-attach-target`` attaches ``#agxir.target`` attributes.
  ``-gpu-module-to-binary`` then produces AGX assembly, ELF objects or
  bitcode.

.. code-block:: console

  $ mlir-opt kernel.mlir --pass-pipeline='builtin.module(agxir-attach-target{chip=agx-g14g},gpu.module(convert-gpu-to-agxir),gpu-module-to-binary{format=isa})'
