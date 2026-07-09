===================================================
Scheduling Information for RISC-V VCIX Instructions
===================================================

.. contents::
   :local:

Summary
-------
The purpose of this document is to outline how the scheduling information for RISC-V's ``XSfvcp`` extension -- SiFive Vector Coprocessor Interface (VCIX) -- in LLVM works, why it works the way it does, and how one may modify the code to support their VCIX needs.

SiFive makes no guarantee that modifying the upstream code to describe their VCIX implementations will lead to performance improvements over the default implementation.

Introduction
------------
LLVM uses scheduler models to describe the behavior of processor latencies and resources. The scheduler models are attached to a processor definition (i.e. ``-mcpu=``) or tunings (i.e. ``-mtune=``). The challenge with VCIX is that the same processor definition could be used with different coprocessors that have very different latencies or processor resource usage for a given instruction. As a result, a default implementation is provided, and one may use this document to customize the existing implementation to their needs.

Understanding the VCIX Scheduling Model Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported Scheduling Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

VCIX is supported in the SiFive 7 family scheduling models, for instance ``SiFive7VLEN512Model`` and ``SiFive7VLEN1024X300Model``. These models share a large portion of scheduling information, including those for VCIX instructions. Therefore, when it comes to customizing VCIX scheduling info, which we will walk you through in later sections, you only need to modify a single place.

The SiFive 7 scheduling models are used in the following (tuning) processors:

*   ``-mtune=sifive7-series``
*   ``-mcpu=sifive-x390``
*   ``-mcpu=sifive-x280``
*   ``-mcpu=sifive-e76``
*   ``-mcpu=sifive-s76``
*   ``-mcpu=sifive-u74``

Understanding the Default Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To read the default implementation, please open ``llvm/lib/Target/RISCV/RISCVSchedSiFive7.td`` and navigate to the line that says ``// VCIX``. The line can be found on GitHub `here <https://github.com/llvm/llvm-project/blob/a00278632dfed7b856a0ac11a58423cb6b14a8c1/llvm/lib/Target/RISCV/RISCVSchedSiFive7.td#L1161>`__.

To understand the code here, we first provide a brief overview. VCIX Pseudo-instructions are defined in ``llvm/lib/Target/RISCV/RISCVInstrInfoXSf.td`` which can be found on GitHub `here <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/RISCV/RISCVInstrInfoXSf.td>`__.

For example:

.. code-block::

  multiclass VPseudoVC_X<LMULInfo m, DAGOperand RS1Class,
                         Operand OpClass = payload2> {
    let VLMul = m.value in {
      let Defs = [SF_VCIX_STATE], Uses = [SF_VCIX_STATE] in {
        def "PseudoVC_" # NAME # "_SE_" # m.MX
          : VPseudoVC_X<OpClass, RS1Class>,
            Sched<[!cast<SchedWrite>("WriteVC_" # NAME # "_" # m.MX)]>;
        def "PseudoVC_V_" # NAME # "_SE_" # m.MX
          : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
            Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
      }
      def "PseudoVC_V_" # NAME # "_" # m.MX
        : VPseudoVC_V_X<OpClass, m.vrclass, RS1Class>,
          Sched<[!cast<SchedWrite>("WriteVC_V_" # NAME # "_" # m.MX)]>;
    }
  }

  // snip

  let Predicates = [HasVendorXSfvcp] in {
    foreach m = MxList in {
      defm X : VPseudoVC_X<m, GPR>;

  // snip

In this example, for each LMUL ``m.MX``, there are three pseudos defined:

1.  ``PseudoVC_X_SE_ # m.MX``
2.  ``PseudoVC_V_X_SE_ # m.MX``
3.  ``PseudoVC_V_X_ # m.MX``

Note that ``#`` concatenates the first string with the LMUL ``m.MX``. When ``m.MX`` is ``M2`` for example, the three pseudos would be defined:

1.  ``PseudoVC_X_SE_M2``
2.  ``PseudoVC_V_X_SE_M2``
3.  ``PseudoVC_V_X_M2``

Note that for each of these definitions, there is a ``Sched`` list attached. The ``Sched`` list takes ``SchedWrite`` and ``SchedRead`` objects, which define the behavior of the operands that are written and and read. In the snippet above, the singular write is attached to the pseudo-instruction. It is up to the scheduler model to describe the behavior of each ``SchedWrite``.

Switching back to the scheduling model linked at the start of this section, we explain how behavior is assigned to the VCIX ``SchedWrite`` objects. Let’s take a look at an example:

.. code-block::

  // snip

  defvar Cycles = SiFive7GetCyclesDefault<mx>.c;
  defvar IsWorstCase = SiFive7IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles,
      AcquireAtCycles = [0, 1],
      ReleaseAtCycles = [1, !add(1, Cycles)] in {
      defm "" : LMULWriteResMX<"WriteVC_V_I",   [VCQ, VA1], mx, IsWorstCase>;

  // snip

Here, the ``LMULWriteResMX`` creates a ``WriteRes`` for each supported LMULs, which is represented by ``mx`` above. A ``WriteRes`` associates processor resources, processor resource usage, and latency with each ``SchedWrite``. In this example, the ``SchedWrite`` named ``WriteVC_V_I # mx`` is being said to use the ``VCQ`` (vector command queue) and ``VA1`` (vector arithmetic sequencer) processor resources.
``AcquireAtCycles[i]`` defines a cycle, relative to instruction issue, that processor resource ``i`` in the ``LMULWriteResMX`` below is acquired at. Similarly, ``ReleaseAtCycles[i]`` defines a cycle, relative to instruction issue, that processor resource ``i`` in the ``LMULWriteResMX`` below is released at. For this ``LMULWriteResMX``, we’re saying that the vector command queue is acquired at cycle 0 and released at cycle 1 and the vector arithmetic sequencer is acquired at cycle 1 and released at cycle 1+Cycles. ``Cycles`` gets its value from a function that describes the default behavior. Looking at the entire VCIX default implementation, you can see that all instructions are given this behavior.

From here, you should have enough background on how the default implementation works.

Basic Scheduling Info Customization
-----------------------------------
The default implementation sets the ``Latency``, ``AcquireAtCycles`` and ``ReleaseAtCycles`` the same way for all VCIX instructions. Let’s walk through an example where we *customize* the default implementation for our needs.

Let’s assume that ``WriteVC_V_I`` behaves differently from the default implementation, and all the other VCIX instructions behave the same as the default implementation. We might write something like this:

.. code-block::

  defvar CustomCycles = SiFive7GetCustomCycles<mx>.c;
  defvar IsWorstCase = SiFive7IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CustomCycles,
      AcquireAtCycles = [0, 1],
      ReleaseAtCycles = [1, !add(1, CustomCycles)] in
    defm "" : LMULWriteResMX<"WriteVC_V_I",   [VCQ, VA1], mx, IsWorstCase>;

    // snip rest

In this example, we wrote a new function ``SiFive7GetCustomCycles`` which takes an argument ``mx`` which describes the LMUL we are scheduling for. It is up to you to determine the number of cycles that should be returned, based on the behavior of your implementation. You can read the implementation of ``SiFive7GetCyclesDefault`` to help you write a custom one.

We set ``Latency`` using the result of our new custom function. Then we left ``AcquireAtCycles[0]`` and ``ReleaseAtCycles[0]`` the same because we assume that the ``VCQ`` has the same behavior: it takes one cycle to get dequeued. Then, we use the result of our new custom function to describe how long the arithmetic sequencer is used. In this case, we made the ``Latency`` and occupancy of the arithmetic sequencer the same, but we could have easily written two custom functions instead.

Another thing that can be customized is the processor resources that are used by a VCIX instruction. Currently, these instructions all use the vector command queue and the vector arithmetic sequencer. However, you can add another ``ProcResource`` to the list, and describe when it is acquired and released in ``AcquireAtCycles`` and ``ReleaseAtCycles``.

To do so, first let’s look at the existing ``ProcResource`` we used before, namely ``VCQ`` and ``VA1``. These two instances are actually parameters passed to the enclosing structure, ``SiFive7WriteResBase``. Their actual definitions are placed `here <https://github.com/llvm/llvm-project/blob/e087d428823e1d1d4c00c895bc3b637989764104/llvm/lib/Target/RISCV/RISCVSchedSiFive7.td#L273>`__:

.. code-block::

  def PipeA   : ProcResource<1>;
  def PipeB   : ProcResource<1>;
  def IDiv    : ProcResource<1>; // Int Division
  def FDiv    : ProcResource<1>; // FP Division/Sqrt

  // Arithmetic sequencer(s)
  // VA1 can handle any vector airthmetic instruction.
  def VA1     : ProcResource<1>;
  if dualVALU then {
    // VA2 generally can only handle simple vector arithmetic.
    def VA2     : ProcResource<1>;
  }

  def VL      : ProcResource<1>; // Load sequencer
  def VS      : ProcResource<1>; // Store sequencer
  def VCQ     : ProcResource<1>; // Vector Command Queue

These ``ProcResources`` are instantiated in another class, ``SiFive7SchedResources``, where we also create an alias for each of them (through ``defvar``) so that it's easier to use later:

.. code-block::

  defvar SiFive7PipeA = !cast<ProcResource>(NAME # "SiFive7PipeA");
  defvar SiFive7PipeB = !cast<ProcResource>(NAME # "SiFive7PipeB");
  defvar SiFive7PipeAB = !cast<ProcResGroup>(NAME # "SiFive7PipeAB");
  defvar SiFive7IDiv = !cast<ProcResource>(NAME # "SiFive7IDiv");
  defvar SiFive7FDiv = !cast<ProcResource>(NAME # "SiFive7FDiv");

  defvar SiFive7VA1 = !cast<ProcResource>(NAME # "SiFive7VA1");

  defvar SiFive7VA1OrVA2 = !if (dualVALU,
                                !cast<ProcResGroup>(NAME # "SiFive7VA1OrVA2"),
                                !cast<ProcResource>(NAME # "SiFive7VA1"));

Specifically, ``SiFive7VA1`` here is the alias for ``VA1`` mentioned previously, which is also the instance we’ll eventually pass as a parameter to ``SiFive7WriteResBase`` mentioned earlier.

So if you want to add your own, that might look something like this:

.. code-block::

  // Step 1: create a new ProcResource
  def CustomVCIX          : ProcResource<1>;

  // Step 2: add a new parameter to SiFive7WriteResBase
  multiclass SiFive7WriteResBase<int VLEN,
      ProcResourceKind PipeA, ProcResourceKind PipeB, ProcResourceKind PipeAB,
      ...
      ProcResourceKind VCQ, ProcResourceKind CustomVCIX,
      ...>

  // Step 3: update SiFive7SchedResources
  defvar SiFive7CustomVCIX = !cast<ProcResource>(NAME # SiFive7CustomVCIX);

  defm SiFive7
     : SiFive7WriteResBase<vlen, SiFive7PipeA, SiFive7PipeB, SiFive7PipeAB,
                           ...
                           SiFive7VCQ, SiFive7CustomVCIX, ...>;

  // Final Step: update scheduling info entry

  defvar CustomCycles = SiFive7GetCustomCycles<mx>.c;
  defvar VCIXCycles = SiFive7GetCustomCyclesVCIX<mx>.c;
  defvar IsWorstCase = SiFive7IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CustomCycles,
      AcquireAtCycles = [0, 1, CustomCycles],
      ReleaseAtCycles = [1, !add(1, CustomCycles), !add(1, VCIXCycles) ] in
    defm "" : LMULWriteResMX<"WriteVC_V_I",   [SiFive7VCQ, VA, CustomVCIX], mx, ...>;

Advanced Scheduling Info Customization
--------------------------------------
Another scenario you may be interested in handling is changing scheduling information based on the *value* of the immediate in the first operand of the pseudo-instruction, since it is considered part of the opcode. In order to model this there are a few steps:

1.  Define ``WriteRes`` objects for all pseudo + opcode combinations.
2.  Define ``MCSchedPredicate`` objects for all opcode combinations
3.  Define ``SchedVar`` objects to tie ``MCSchedPredicate`` objects to ``WriteRes`` objects
4.  Define ``SchedWriteVariant`` to aggregate all ``SchedVar`` objects together
5.  Define a ``SchedAlias`` object to tie the behavior of the original ``WriteRes`` name with the ``SchedWriteVariant``

There is an existing helper class, ``LMULWriteResMXVariant``, which can be found on GitHub `here <https://github.com/llvm/llvm-project/blob/36b339b84a98afe7bdf470747a776d0d5f348b64/llvm/lib/Target/RISCV/RISCVScheduleV.td#L74>`__ that implements this process for the case when there is a single predicate.

.. code-block::

  def IsOp0ImmEq2 : MCSchedPredicate<CheckImmOperand<0, 2>>; // true when operand 0 is 2

  defm  : LMULWriteResMXVariant<"WriteVC_V_I", IsOp0ImmEq2,
                              // When IsOp0ImmEq2 is true
                              [VCQ, VA1], 20, [0, 1], [1, 22],
                              // Other cases
                              [VCQ, VA1], 3, [0, 1], [1, 4],
                              mx, ...>;

In the above example, ``WriteVC_V_I`` will be assigned a latency of 20 cycles and hold ``VA1`` for 22 cycles if its first (immediate) operand has a value of 2. Otherwise, the latency would be 3 cycles with an occupancy of 4 cycles on ``VA1``.

To extend it to handle multiple opcodes, you would add add additional ``WriteRes`` definitions for each pseudo + opcode combinations, define additional ``MCSchedPredicate`` objects for each opcode, define additional ``SchedVar`` objects to tie these new objects together, and add these ``SchedVar`` objects to the ``SchedWriteVariant``.

To define a predicate that checks an opcode immediate is 0 or 1 for example, you might write something like this for the ``WriteVC_V_I`` pseudo for LMUL ``mx``:

.. code-block::

  // Define WriteRes objects for all pseudo + opcode combinations
  let Latency = 3, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, 4] in
  def "WriteVC_V_I_" # mx # "_Opc0" : SchedWriteRes<[VCQ, VA1]>
  let Latency = 10, AcquireAtCycles = [0, 1], ReleaseAtCycles = [1, 11] in
  def "WriteVC_V_I_" # mx # "_Opc1" : SchedWriteRes<[VCQ, VA1]>

  // Define MCSchedPredicate objects for all opcode combinations. This toy example shows how to
  // do this with made up opcodes. Please refer to the VCIX manual for opcodes you will want to
  // support.
  def IsOp0ImmEq0 : MCSchedPredicate<CheckImmOperand<0, 0>>; // true when operand 0 is 0
  def IsOp0ImmEq1 : MCSchedPredicate<CheckImmOperand<0, 1>>; // true when operand 0 is 1

  // Define SchedVar objects to tie MCSchedPredicate objects to WriteRes objects
  def "WriteVC_V_I_" # mx # "_Opc0SchedVar"
    : SchedVar<IsOp0ImmEq0, [!cast<SchedWriteRes>("WriteVC_V_I_" # mx # "_Opc0")]>;
  def "WriteVC_V_I_" # mx # "_Opc1SchedVar"
    : SchedVar<IsOp0ImmEq0, [!cast<SchedWriteRes>("WriteVC_V_I_" # mx # "_Opc1")]>;

  // Define SchedWriteVariant to aggregate all SchedVar objects together
  def "WriteVC_V_I_" # mx # "Variant"
    : SchedWriteVariant<["WriteVC_V_I_" # mx # "_Opc0SchedVar",
                         "WriteVC_V_I_" # mx # "_Opc1SchedVar"]>;

  // Define a SchedAlias object to tie the behavior of the original WriteRes name with the SchedWriteVariant
  def : SchedAlias<!cast<SchedReadWrite>("WriteVC_V_I_" # mx),
                       !cast<SchedReadWrite>("WriteVC_V_I_" # mx # "Variant")>;


From here, you should have a strong understanding of how to modify the default implementation of VCIX scheduling in LLVM.
