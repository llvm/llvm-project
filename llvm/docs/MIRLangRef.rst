========================================
Machine IR (MIR) Format Reference Manual
========================================

.. contents::
   :local:

.. warning::
  This is a work in progress.

Introduction
============

This document is a reference manual for the Machine IR (MIR) serialization
format. MIR is a human readable serialization format that is used to represent
LLVM's :ref:`machine specific intermediate representation
<machine code representation>`.

The MIR serialization format is designed to be used for testing the code
generation passes in LLVM.

Overview
========

The MIR serialization format uses a YAML container. YAML is a standard
data serialization language, and the full YAML language spec can be read at
`yaml.org
<http://www.yaml.org/spec/1.2/spec.html#Introduction>`_.

A MIR file is split up into a series of `YAML documents`_. The first document
can contain an optional embedded LLVM IR module, and the rest of the documents
contain the serialized machine functions.

.. _YAML documents: http://www.yaml.org/spec/1.2/spec.html#id2800132

MIR Testing Guide
=================

You can use the MIR format for testing in two different ways:

- You can write MIR tests that invoke a single code generation pass using the
  ``-run-pass`` option in llc.

- You can use llc's ``-stop-after`` option with existing or new LLVM assembly
  tests and check the MIR output of a specific code generation pass.

Testing Individual Code Generation Passes
-----------------------------------------

The ``-run-pass`` option in llc allows you to create MIR tests that invoke just
a single code generation pass. When this option is used, llc will parse an
input MIR file, run the specified code generation pass(es), and output the
resulting MIR code.

You can generate an input MIR file for the test by using the ``-stop-after`` or
``-stop-before`` option in llc. For example, if you would like to write a test
for the post register allocation pseudo instruction expansion pass, you can
specify the machine copy propagation pass in the ``-stop-after`` option, as it
runs just before the pass that we are trying to test:

   ``llc -stop-after=machine-cp bug-trigger.ll > test.mir``

After generating the input MIR file, you'll have to add a run line that uses
the ``-run-pass`` option to it. In order to test the post register allocation
pseudo instruction expansion pass on X86-64, a run line like the one shown
below can be used:

    ``# RUN: llc -o - %s -mtriple=x86_64-- -run-pass=postrapseudos | FileCheck %s``

The MIR files are target dependent, so they have to be placed in the target
specific test directories (``lib/CodeGen/TARGETNAME``). They also need to
specify a target triple or a target architecture either in the run line or in
the embedded LLVM IR module.

Simplifying MIR files
^^^^^^^^^^^^^^^^^^^^^

The MIR code coming out of ``-stop-after``/``-stop-before`` is very verbose;
Tests are more accessible and future proof when simplified:

- Use the ``-simplify-mir`` option with llc.

- Machine function attributes often have default values or the test works just
  as well with default values. Typical candidates for this are: `alignment:`,
  `exposesReturnsTwice`, `legalized`, `regBankSelected`, `selected`.
  The whole `frameInfo` section is often unnecessary if there is no special
  frame usage in the function. `tracksRegLiveness` on the other hand is often
  necessary for some passes that care about block livein lists.

- The (global) `liveins:` list is typically only interesting for early
  instruction selection passes and can be removed when testing later passes.
  The per-block `liveins:` on the other hand are necessary if
  `tracksRegLiveness` is true.

- Branch probability data in block `successors:` lists can be dropped if the
  test doesn't depend on it. Example:
  `successors: %bb.1(0x40000000), %bb.2(0x40000000)` can be replaced with
  `successors: %bb.1, %bb.2`.

- MIR code contains a whole IR module. This is necessary because there are
  no equivalents in MIR for global variables, references to external functions,
  function attributes, metadata, debug info. Instead some MIR data references
  the IR constructs. You can often remove them if the test doesn't depend on
  them.

- Alias Analysis is performed on IR values. These are referenced by memory
  operands in MIR. Example: `:: (load 8 from %ir.foobar, !alias.scope !9)`.
  If the test doesn't depend on (good) alias analysis the references can be
  dropped: `:: (load 8)`

- MIR blocks can reference IR blocks for debug printing, profile information
  or debug locations. Example: `bb.42.myblock` in MIR references the IR block
  `myblock`. It is usually possible to drop the `.myblock` reference and simply
  use `bb.42`.

- If there are no memory operands or blocks referencing the IR then the
  IR function can be replaced by a parameterless dummy function like
  `define @func() { ret void }`.

- It is possible to drop the whole IR section of the MIR file if it only
  contains dummy functions (see above). The .mir loader will create the
  IR functions automatically in this case.

.. _limitations:

Limitations
-----------

Currently the MIR format has several limitations in terms of which state it
can serialize:

- The target-specific state in the target-specific ``MachineFunctionInfo``
  subclasses isn't serialized at the moment.

- The target-specific ``MachineConstantPoolValue`` subclasses (in the ARM and
  SystemZ backends) aren't serialized at the moment.

- The ``MCSymbol`` machine operands are only printed, they can't be parsed.

- A lot of the state in ``MachineModuleInfo`` isn't serialized - only the CFI
  instructions and the variable debug information from MMI is serialized right
  now.

These limitations impose restrictions on what you can test with the MIR format.
For now, tests that would like to test some behaviour that depends on the state
of certain ``MCSymbol``  operands or the exception handling state in MMI, can't
use the MIR format. As well as that, tests that test some behaviour that
depends on the state of the target specific ``MachineFunctionInfo`` or
``MachineConstantPoolValue`` subclasses can't use the MIR format at the moment.

High Level Structure
====================

.. _embedded-module:

Embedded Module
---------------

When the first YAML document contains a `YAML block literal string`_, the MIR
parser will treat this string as an LLVM assembly language string that
represents an embedded LLVM IR module.
Here is an example of a YAML document that contains an LLVM module:

.. code-block:: llvm

       define i32 @inc(i32* %x) {
       entry:
         %0 = load i32, i32* %x
         %1 = add i32 %0, 1
         store i32 %1, i32* %x
         ret i32 %1
       }

.. _YAML block literal string: http://www.yaml.org/spec/1.2/spec.html#id2795688

Machine Functions
-----------------

The remaining YAML documents contain the machine functions. This is an example
of such YAML document:

.. code-block:: text

     ---
     name:            inc
     tracksRegLiveness: true
     liveins:
       - { reg: '%rdi' }
     body: |
       bb.0.entry:
         liveins: %rdi

         %eax = MOV32rm %rdi, 1, _, 0, _
         %eax = INC32r killed %eax, implicit-def dead %eflags
         MOV32mr killed %rdi, 1, _, 0, _, %eax
         RETQ %eax
     ...

The document above consists of attributes that represent the various
properties and data structures in a machine function.

The attribute ``name`` is required, and its value should be identical to the
name of a function that this machine function is based on.

The attribute ``body`` is a `YAML block literal string`_. Its value represents
the function's machine basic blocks and their machine instructions.

Machine Instructions Format Reference
=====================================

The machine basic blocks and their instructions are represented using a custom,
human readable serialization language. This language is used in the
`YAML block literal string`_ that corresponds to the machine function's body.

A source string that uses this language contains a list of machine basic
blocks, which are described in the section below.

Machine Basic Blocks
--------------------

A machine basic block is defined in a single block definition source construct
that contains the block's ID.
The example below defines two blocks that have an ID of zero and one:

.. code-block:: text

    bb.0:
      <instructions>
    bb.1:
      <instructions>

A machine basic block can also have a name. It should be specified after the ID
in the block's definition:

.. code-block:: text

    bb.0.entry:       ; This block's name is "entry"
       <instructions>

The block's name should be identical to the name of the IR block that this
machine block is based on.

.. _block-references:

Block References
^^^^^^^^^^^^^^^^

The machine basic blocks are identified by their ID numbers. Individual
blocks are referenced using the following syntax:

.. code-block:: text

    %bb.<id>

Example:

.. code-block:: llvm

    %bb.0

The following syntax is also supported, but the former syntax is preferred for
block references:

.. code-block:: text

    %bb.<id>[.<name>]

Example:

.. code-block:: llvm

    %bb.1.then

Successors
^^^^^^^^^^

The machine basic block's successors have to be specified before any of the
instructions:

.. code-block:: text

    bb.0.entry:
      successors: %bb.1.then, %bb.2.else
      <instructions>
    bb.1.then:
      <instructions>
    bb.2.else:
      <instructions>

The branch weights can be specified in brackets after the successor blocks.
The example below defines a block that has two successors with branch weights
of 32 and 16:

.. code-block:: text

    bb.0.entry:
      successors: %bb.1.then(32), %bb.2.else(16)

.. _bb-liveins:

Live In Registers
^^^^^^^^^^^^^^^^^

The machine basic block's live in registers have to be specified before any of
the instructions:

.. code-block:: text

    bb.0.entry:
      liveins: %edi, %esi

The list of live in registers and successors can be empty. The language also
allows multiple live in register and successor lists - they are combined into
one list by the parser.

Miscellaneous Attributes
^^^^^^^^^^^^^^^^^^^^^^^^

The attributes ``IsAddressTaken``, ``IsLandingPad`` and ``Alignment`` can be
specified in brackets after the block's definition:

.. code-block:: text

    bb.0.entry (address-taken):
      <instructions>
    bb.2.else (align 4):
      <instructions>
    bb.3(landing-pad, align 4):
      <instructions>

.. TODO: Describe the way the reference to an unnamed LLVM IR block can be
   preserved.

Machine Instructions
--------------------

A machine instruction is composed of a name,
:ref:`machine operands <machine-operands>`,
:ref:`instruction flags <instruction-flags>`, and machine memory operands.

The instruction's name is usually specified before the operands. The example
below shows an instance of the X86 ``RETQ`` instruction with a single machine
operand:

.. code-block:: text

    RETQ %eax

However, if the machine instruction has one or more explicitly defined register
operands, the instruction's name has to be specified after them. The example
below shows an instance of the AArch64 ``LDPXpost`` instruction with three
defined register operands:

.. code-block:: text

    %sp, %fp, %lr = LDPXpost %sp, 2

The instruction names are serialized using the exact definitions from the
target's ``*InstrInfo.td`` files, and they are case sensitive. This means that
similar instruction names like ``TSTri`` and ``tSTRi`` represent different
machine instructions.

.. _instruction-flags:

Instruction Flags
^^^^^^^^^^^^^^^^^

The flag ``frame-setup`` can be specified before the instruction's name:

.. code-block:: text

    %fp = frame-setup ADDXri %sp, 0, 0

.. _registers:

Registers
---------

Registers are one of the key primitives in the machine instructions
serialization language. They are primarly used in the
:ref:`register machine operands <register-operands>`,
but they can also be used in a number of other places, like the
:ref:`basic block's live in list <bb-liveins>`.

The physical registers are identified by their name. They use the following
syntax:

.. code-block:: text

    %<name>

The example below shows three X86 physical registers:

.. code-block:: text

    %eax
    %r15
    %eflags

The virtual registers are identified by their ID number. They use the following
syntax:

.. code-block:: text

    %<id>

Example:

.. code-block:: text

    %0

The null registers are represented using an underscore ('``_``'). They can also be
represented using a '``%noreg``' named register, although the former syntax
is preferred.

.. _machine-operands:

Machine Operands
----------------

There are seventeen different kinds of machine operands, and all of them, except
the ``MCSymbol`` operand, can be serialized. The ``MCSymbol`` operands are
just printed out - they can't be parsed back yet.

Immediate Operands
^^^^^^^^^^^^^^^^^^

The immediate machine operands are untyped, 64-bit signed integers. The
example below shows an instance of the X86 ``MOV32ri`` instruction that has an
immediate machine operand ``-42``:

.. code-block:: text

    %eax = MOV32ri -42

An immediate operand is also used to represent a subregister index when the
machine instruction has one of the following opcodes:

- ``EXTRACT_SUBREG``

- ``INSERT_SUBREG``

- ``REG_SEQUENCE``

- ``SUBREG_TO_REG``

In case this is true, the Machine Operand is printed according to the target.

For example:

In AArch64RegisterInfo.td:

.. code-block:: text

  def sub_32 : SubRegIndex<32>;

If the third operand is an immediate with the value ``15`` (target-dependent
value), based on the instruction's opcode and the operand's index the operand
will be printed as ``%subreg.sub_32``:

.. code-block:: text

    %1:gpr64 = SUBREG_TO_REG 0, %0, %subreg.sub_32

For integers > 64bit, we use a special machine operand, ``MO_CImmediate``,
which stores the immediate in a ``ConstantInt`` using an ``APInt`` (LLVM's
arbitrary precision integers).

.. TODO: Describe the FPIMM immediate operands.

.. _register-operands:

Register Operands
^^^^^^^^^^^^^^^^^

The :ref:`register <registers>` primitive is used to represent the register
machine operands. The register operands can also have optional
:ref:`register flags <register-flags>`,
:ref:`a subregister index <subregister-indices>`,
and a reference to the tied register operand.
The full syntax of a register operand is shown below:

.. code-block:: text

    [<flags>] <register> [ :<subregister-idx-name> ] [ (tied-def <tied-op>) ]

This example shows an instance of the X86 ``XOR32rr`` instruction that has
5 register operands with different register flags:

.. code-block:: text

  dead %eax = XOR32rr undef %eax, undef %eax, implicit-def dead %eflags, implicit-def %al

.. _register-flags:

Register Flags
~~~~~~~~~~~~~~

The table below shows all of the possible register flags along with the
corresponding internal ``llvm::RegState`` representation:

.. list-table::
   :header-rows: 1

   * - Flag
     - Internal Value

   * - ``implicit``
     - ``RegState::Implicit``

   * - ``implicit-def``
     - ``RegState::ImplicitDefine``

   * - ``def``
     - ``RegState::Define``

   * - ``dead``
     - ``RegState::Dead``

   * - ``killed``
     - ``RegState::Kill``

   * - ``undef``
     - ``RegState::Undef``

   * - ``internal``
     - ``RegState::InternalRead``

   * - ``early-clobber``
     - ``RegState::EarlyClobber``

   * - ``debug-use``
     - ``RegState::Debug``

   * - ``renamable``
     - ``RegState::Renamable``

.. _subregister-indices:

Subregister Indices
~~~~~~~~~~~~~~~~~~~

The register machine operands can reference a portion of a register by using
the subregister indices. The example below shows an instance of the ``COPY``
pseudo instruction that uses the X86 ``sub_8bit`` subregister index to copy 8
lower bits from the 32-bit virtual register 0 to the 8-bit virtual register 1:

.. code-block:: text

    %1 = COPY %0:sub_8bit

The names of the subregister indices are target specific, and are typically
defined in the target's ``*RegisterInfo.td`` file.

Constant Pool Indices
^^^^^^^^^^^^^^^^^^^^^

A constant pool index (CPI) operand is printed using its index in the
function's ``MachineConstantPool`` and an offset.

For example, a CPI with the index 1 and offset 8:

.. code-block:: text

    %1:gr64 = MOV64ri %const.1 + 8

For a CPI with the index 0 and offset -12:

.. code-block:: text

    %1:gr64 = MOV64ri %const.0 - 12

A constant pool entry is bound to a LLVM IR ``Constant`` or a target-specific
``MachineConstantPoolValue``. When serializing all the function's constants the
following format is used:

.. code-block:: text

    constants:
      - id:               <index>
        value:            <value>
        alignment:        <alignment>
        isTargetSpecific: <target-specific>

where ``<index>`` is a 32-bit unsigned integer, ``<value>`` is a `LLVM IR Constant
<https://www.llvm.org/docs/LangRef.html#constants>`_, alignment is a 32-bit
unsigned integer, and ``<target-specific>`` is either true or false.

Example:

.. code-block:: text

    constants:
      - id:               0
        value:            'double 3.250000e+00'
        alignment:        8
      - id:               1
        value:            'g-(LPC0+8)'
        alignment:        4
        isTargetSpecific: true

Global Value Operands
^^^^^^^^^^^^^^^^^^^^^

The global value machine operands reference the global values from the
:ref:`embedded LLVM IR module <embedded-module>`.
The example below shows an instance of the X86 ``MOV64rm`` instruction that has
a global value operand named ``G``:

.. code-block:: text

    %rax = MOV64rm %rip, 1, _, @G, _

The named global values are represented using an identifier with the '@' prefix.
If the identifier doesn't match the regular expression
`[-a-zA-Z$._][-a-zA-Z$._0-9]*`, then this identifier must be quoted.

The unnamed global values are represented using an unsigned numeric value with
the '@' prefix, like in the following examples: ``@0``, ``@989``.

Target-dependent Index Operands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A target index operand is a target-specific index and an offset. The
target-specific index is printed using target-specific names and a positive or
negative offset.

For example, the ``amdgpu-constdata-start`` is associated with the index ``0``
in the AMDGPU backend. So if we have a target index operand with the index 0
and the offset 8:

.. code-block:: text

    %sgpr2 = S_ADD_U32 _, target-index(amdgpu-constdata-start) + 8, implicit-def _, implicit-def _

Jump-table Index Operands
^^^^^^^^^^^^^^^^^^^^^^^^^

A jump-table index operand with the index 0 is printed as following:

.. code-block:: text

    tBR_JTr killed %r0, %jump-table.0

A machine jump-table entry contains a list of ``MachineBasicBlocks``. When serializing all the function's jump-table entries, the following format is used:

.. code-block:: text

    jumpTable:
      kind:             <kind>
      entries:
        - id:             <index>
          blocks:         [ <bbreference>, <bbreference>, ... ]

where ``<kind>`` is describing how the jump table is represented and emitted (plain address, relocations, PIC, etc.), and each ``<index>`` is a 32-bit unsigned integer and ``blocks`` contains a list of :ref:`machine basic block references <block-references>`.

Example:

.. code-block:: text

    jumpTable:
      kind:             inline
      entries:
        - id:             0
          blocks:         [ '%bb.3', '%bb.9', '%bb.4.d3' ]
        - id:             1
          blocks:         [ '%bb.7', '%bb.7', '%bb.4.d3', '%bb.5' ]

External Symbol Operands
^^^^^^^^^^^^^^^^^^^^^^^^^

An external symbol operand is represented using an identifier with the ``$``
prefix. The identifier is surrounded with ""'s and escaped if it has any
special non-printable characters in it.

Example:

.. code-block:: text

    CALL64pcrel32 $__stack_chk_fail, csr_64, implicit %rsp, implicit-def %rsp

MCSymbol Operands
^^^^^^^^^^^^^^^^^

A MCSymbol operand is holding a pointer to a ``MCSymbol``. For the limitations
of this operand in MIR, see :ref:`limitations <limitations>`.

The syntax is:

.. code-block:: text

    EH_LABEL <mcsymbol Ltmp1>

CFIIndex Operands
^^^^^^^^^^^^^^^^^

A CFI Index operand is holding an index into a per-function side-table,
``MachineFunction::getFrameInstructions()``, which references all the frame
instructions in a ``MachineFunction``. A ``CFI_INSTRUCTION`` may look like it
contains multiple operands, but the only operand it contains is the CFI Index.
The other operands are tracked by the ``MCCFIInstruction`` object.

The syntax is:

.. code-block:: text

    CFI_INSTRUCTION offset %w30, -16

which may be emitted later in the MC layer as:

.. code-block:: text

    .cfi_offset w30, -16

.. TODO: Describe the parsers default behaviour when optional YAML attributes
   are missing.
.. TODO: Describe the syntax for the bundled instructions.
.. TODO: Describe the syntax for virtual register YAML definitions.
.. TODO: Describe the machine function's YAML flag attributes.
.. TODO: Describe the syntax for the register mask machine operands.
.. TODO: Describe the frame information YAML mapping.
.. TODO: Describe the syntax of the stack object machine operands and their
   YAML definitions.
.. TODO: Describe the syntax of the block address machine operands.
.. TODO: Describe the syntax of the metadata machine operands, and the
   instructions debug location attribute.
.. TODO: Describe the syntax of the register live out machine operands.
.. TODO: Describe the syntax of the machine memory operands.
