=========================
 RISC-V Vector Extension
=========================

.. contents::
   :local:

The RISC-V target readily supports the 1.0 version of the `RISC-V Vector Extension (RVV) <https://github.com/riscv/riscv-v-spec/blob/v1.0/v-spec.adoc>`_, but requires some tricks to handle its unique design.
This guide gives an overview of how RVV is modelled in LLVM IR and how the backend generates code for it.

Mapping to LLVM IR types
========================

RVV adds 32 ``VLEN`` sized registers, where ``VLEN`` is an unknown constant to the compiler. To be able to represent ``VLEN`` sized values, the RISC-V backend takes the same approach as AArch64's SVE and uses `scalable vector types <https://llvm.org/docs/LangRef.html#t-vector>`_.

Scalable vector types are of the form ``<vscale x n x ty>``, which indicate a vector with a multiple of ``n`` elements of type ``ty``. ``n`` and ``ty`` then end up controlling LMUL and SEW respectively.

LLVM supports only ``ELEN=32`` or ``ELEN=64``, so ``vscale`` is defined as ``VLEN/64`` (see ``RISCV::RVVBitsPerBlock``).
This makes the LLVM IR types stable between the two ``ELEN`` s considered, i.e. every LLVM IR scalable vector type has exactly one corresponding pair of element type and LMUL, and vice-versa.

+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
|                   | LMUL=⅛        | LMUL=¼         | LMUL=½           | LMUL=1            | LMUL=2            | LMUL=4            | LMUL=8            |
+===================+===============+================+==================+===================+===================+===================+===================+
| i64 (ELEN=64)     | N/A           | N/A            | N/A              | <v x 1 x i64>     | <v x 2 x i64>     | <v x 4 x i64>     | <v x 8 x i64>     |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| i32               | N/A           | N/A            | <v x 1 x i32>    | <v x 2 x i32>     | <v x 4 x i32>     | <v x 8 x i32>     | <v x 16 x i32>    |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| i16               | N/A           | <v x 1 x i16>  | <v x 2 x i16>    | <v x 4 x i16>     | <v x 8 x i16>     | <v x 16 x i16>    | <v x 32 x i16>    |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| i8                | <v x 1 x i8>  | <v x 2 x i8>   | <v x 4 x i8>     | <v x 8 x i8>      | <v x 16 x i8>     | <v x 32 x i8>     | <v x 64 x i8>     |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| double (ELEN=64)  | N/A           | N/A            | N/A              | <v x 1 x double>  | <v x 2 x double>  | <v x 4 x double>  | <v x 8 x double>  |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| float             | N/A           | N/A            | <v x 1 x float>  | <v x 2 x float>   | <v x 4 x float>   | <v x 8 x float>   | <v x 16 x float>  |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+
| half              | N/A           | <v x 1 x half> | <v x 2 x half>   | <v x 4 x half>    | <v x 8 x half>    | <v x 16 x half>   | <v x 32 x half>   |
+-------------------+---------------+----------------+------------------+-------------------+-------------------+-------------------+-------------------+

(Read ``<v x k x ty>`` as ``<vscale x k x ty>``)


Mask vector types
-----------------

As for mask vectors, they are physically represented using a layout of densely packed bits in a vector register.
They are mapped to the following LLVM IR types:

- <vscale x 1 x i1>
- <vscale x 2 x i1>
- <vscale x 4 x i1>
- <vscale x 8 x i1>
- <vscale x 16 x i1>
- <vscale x 32 x i1>
- <vscale x 64 x i1>

Two types with the same ratio SEW/LMUL will have the same related mask type. For instance, two different comparisons one under SEW=64, LMUL=2 and the other under SEW=32, LMUL=1 will both generate a mask <vscale x 2 x i1>.

Representation in LLVM IR
=========================

Vector instructions can be represented in three main ways in LLVM IR:

1. Regular instructions on both fixed and scalable vector types

   .. code-block:: llvm

       %c = add <vscale x 4 x i32> %a, %b

2. RISC-V vector intrinsics, which mirror the `C intrinsics specification <https://github.com/riscv-non-isa/rvv-intrinsic-doc>`_

   These come in unmasked variants:

   .. code-block:: llvm

       %c = call @llvm.riscv.vadd.nxv4i32.nxv4i32(
              <vscale x 4 x i32> %passthru,
	      <vscale x 4 x i32> %a,
	      <vscale x 4 x i32> %b,
	      i64 %avl
	    )

   As well as masked variants:

   .. code-block:: llvm

       %c = call @llvm.riscv.vadd.nxv4i32.nxv4i32(
              <vscale x 4 x i32> %passthru,
	      <vscale x 4 x i32> %a,
	      <vscale x 4 x i32> %b,
	      i64 %avl
	    )

   Both allow setting the AVL as well as controlling the inactive/tail elements via the passthru operand, but the masked variant also provides operands for the mask and ``vta``/``vma`` policy bits.

   The only valid types are scalable vector types.

3. :doc:`Vector predication (VP) intrinsics </Proposals/VectorPredication>`

   .. code-block:: llvm

       %c = call @llvm.vp.add.nxv4i32(
	      <vscale x 4 x i32> %a,
	      <vscale x 4 x i32> %b,
	      <vscale x 4 x i1> %m
	      i32 %evl
	    )

   Unlike RISC-V intrinsics, VP intrinsics are target agnostic so they can be emitted from other optimisation passes in the middle-end (like the loop vectorizer). They also support fixed length vector types.

SelectionDAG lowering
=====================

For regular **scalable** vector LLVM IR instructions, their corresponding SelectionDAG nodes are legal on RISC-V and don't require any custom lowering.

.. code-block::

   t5: nxv4i32 = add t2, t4

RISC-V vector intrinsics are also always scalable and so don't need custom lowering:

.. code-block::

   t12: nxv4i32 = llvm.riscv.vadd TargetConstant:i64<10056>, undef:nxv4i32, t2, t4, t6

Fixed length vectors
--------------------

The only legal vector MVTs on RISC-V are scalable, so fixed length vectors need to be custom lowered and performed in a scalable container type:

1. The fixed length vector operands are inserted into scalable containers via ``insert_subvector``. The container size is chosen to have a minimum size big enough to fit the fixed length vector (see ``getContainerForFixedLengthVector``).
2. The operation is then performed via a scalable **VL (vector length) node**. These are custom nodes that contain an AVL operand which is set to the size of the fixed length vector, and are defined in RISCVInstrInfoVVLPatterns.td.
3. The result is put back into a fixed length vector via ``extract_subvector``.

.. code-block::

   t2: nxv2i32,ch = CopyFromReg t0, Register:nxv2i32 %0
     t4: v4i32 = extract_subvector t2, Constant:i64<0>
       t6: nxv2i32,ch = CopyFromReg t0, Register:nxv2i32 %1
     t7: v4i32 = extract_subvector t6, Constant:i64<0>
   t8: v4i32 = add t4, t7

   // custom lowered to:

       t2: nxv2i32,ch = CopyFromReg t0, Register:nxv2i32 %0
       t6: nxv2i32,ch = CopyFromReg t0, Register:nxv2i32 %1
       t15: nxv2i1 = RISCVISD::VMSET_VL Constant:i64<4>
     t16: nxv2i32 = RISCVISD::ADD_VL t2, t6, undef:nxv2i32, t15, Constant:i64<4>
   t17: v4i32 = extract_subvector t16, Constant:i64<0>

VL nodes often have a passthru or mask operand, which are usually set to undef and all ones for fixed length vectors.

The ``insert_subvector`` and ``extract_subvector`` nodes responsible for wrapping and unwrapping will get combined away, and eventually we will lower all fixed vector types to scalable. Note that the vectors at the interface of a function are always scalable vectors.

.. note::

   The only ``insert_subvector`` and ``extract_subvector`` nodes that make it through lowering are those that can be performed as an exact subregister insert or extract. This means that any fixed length vector ``insert_subvector`` and ``extract_subvector`` nodes that aren't legalized must lie on a register group boundary, so the exact ``VLEN`` must be known at compile time (i.e. compiled with ``-mrvv-vector-bits=zvl`` or ``-mllvm -riscv-v-vector-bits-max=VLEN``, or have an exact ``vscale_range`` attribute).

Vector predication intrinsics
-----------------------------

VP intrinsics also get custom lowered via VL nodes in order to set the EVL and mask.

.. code-block::

   t12: nxv2i32 = vp_add t2, t4, t6, Constant:i64<8>

   // custom lowered to:

   t18: nxv2i32 = RISCVISD::ADD_VL t2, t4, undef:nxv2i32, t6, Constant:i64<8>


Instruction selection
=====================

VL and VTYPE need to be configured correctly, so we can't just directly select the underlying vector MachineInstrs. Instead a layer of pseudo instructions get selected which carry the extra information needed to emit the necessary ``vsetvli`` instructions later.

.. code-block::

   %c:vrm2 = PseudoVADD_VV_M2 %passthru:vrm2(tied-def 0), %a:vrm2, %b:vrm2, %vl:gpr, 5

Each vector instruction has multiple pseudo instructions defined in ``RISCVInstrInfoVPseudos.td``.

The pseudos have operands for the AVL and SEW (encoded as a power of 2), as well as potentially the mask, policy or rounding mode if applicable.
The passhthru operand is tied to the destination register to control the inactive/tail elements.

For each possible LMUL there is a variant of the pseudo instruction, as it affects the register class needed for the operands, and similarly there are ``_MASK`` variants that control whether or not the instruction is masked.

For scalable vectors that should use VLMAX, the AVL is set to a sentinel value of -1.

There are patterns for target agnostic SelectionDAG nodes in ``RISCVInstrInfoVSDPatterns.td``, VL nodes in ``RISCVInstrInfoVVLPatterns.td`` and RVV intrinsics in ``RISCVInstrInfoVPseudos.td``.

Mask patterns
-------------

For the VL patterns we only match to masked pseudos to reduce the size of the match table, even if the node's mask is all ones and could be an unmasked pseudo. The ``RISCVDAGToDAGISel::doPeepholeMaskedRVV`` will detects that the mask is all ones during post-processing and convert it into its unmasked form.

.. code-block::

     t15: nxv4i1 = RISCVISD::VMSET_VL Constant:i32<-1>
   t16: nxv4i32 = PseudoVADD_MASK_VV_M2 t0, t2, t4, t15, -1, 5

   // gets optimized to:

   t16: nxv4i32 = PseudoVADD_VV_M2 t0, t2, t4, 4, 5

.. note::

   Any vmset_vl can be treated as an all ones mask since the tail elements past VL are undef and can be replaced with ones.

For masked pseudos the mask operand is copied to the physical ``$v0`` register with a glued ``CopyToReg`` node:

.. code-block::

     t23: ch,glue = CopyToReg t0, Register:nxv4i1 $v0, t6
   t25: nxv4i32 = PseudoVADD_VV_M2_MASK Register:nxv4i32 $noreg, t2, t4, Register:nxv4i1 $v0, TargetConstant:i64<8>, TargetConstant:i64<5>, TargetConstant:i64<1>, t23:1

Register allocation
===================

Register allocation is split between vector and scalar registers, with vector allocation running first:

.. code-block::

  $v8m2 = PseudoVADD_VV_M2 $v8m2(tied-def 0), $v8m2, $v10m2, %vl:gpr, 5

.. note::

   We split register allocation between vectors and scalars so that :ref:`RISCVInsertVSETVLI` can run after vector register allocation, but still before scalar register allocation as it may need to create a new virtual register to set the AVL to VLMAX.

   Performing RISCVInsertVSETVLI after vector register allocation imposes fewer constraints on the machine scheduler since it cannot schedule instructions past vsetvlis, and it allows us to emit further vector pseudos during spilling or constant rematerialization.

There are four register classes for vectors:

- ``VR`` for vector registers (``v0``, ``v1,``, ..., ``v32``). Used when :math:`\text{LMUL} \leq 1` and mask registers.
- ``VRM2`` for vector groups of length 2 i.e. :math:`\text{LMUL}=2` (``v0m2``, ``v2m2``, ..., ``v30m2``)
- ``VRM4`` for vector groups of length 4 i.e. :math:`\text{LMUL}=4` (``v0m4``, ``v4m4``, ..., ``v28m4``)
- ``VRM8`` for vector groups of length 8 i.e. :math:`\text{LMUL}=8` (``v0m8``, ``v8m8``, ..., ``v24m8``)

:math:`\text{LMUL} \lt 1` types and mask types do not benefit from having a dedicated class, so ``VR`` is used in their case.

Some instructions have a constraint that a register operand cannot be ``V0`` or overlap with ``V0``, so for these cases we also have ``VRNoV0`` variants.

.. _RISCVInsertVSETVLI:

RISCVInsertVSETVLI
==================

After vector registers are allocated, the RISCVInsertVSETVLI pass will insert the necessary vsetvlis for the pseudos.

.. code-block::

  dead $x0 = PseudoVSETVLI %vl:gpr, 209, implicit-def $vl, implicit-def $vtype
  $v8m2 = PseudoVADD_VV_M2 $v8m2(tied-def 0), $v8m2, $v10m2, $noreg, 5, implicit $vl, implicit $vtype

The physical ``$vl`` and ``$vtype`` registers are implicitly defined by the ``PseudoVSETVLI``, and are implicitly used by the ``PseudoVADD``.
The VTYPE operand (``209`` in this example) is encoded as per the specification via ``RISCVVType::encodeVTYPE``.

RISCVInsertVSETVLI performs dataflow analysis to emit as few vsetvlis as possible. It will also try to minimize the number of vsetvlis that set VL, i.e. it will emit ``vsetvli x0, x0`` if only VTYPE needs changed but VL doesn't.

Pseudo expansion and printing
=============================

After scalar register allocation, the ``RISCVExpandPseudoInsts.cpp`` pass expands out the ``PseudoVSETVLI``.

.. code-block::

   dead $x0 = VSETVLI $x1, 209, implicit-def $vtype, implicit-def $vl
   renamable $v8m2 = PseudoVADD_VV_M2 $v8m2(tied-def 0), $v8m2, $v10m2, $noreg, 5, implicit $vl, implicit $vtype

Note that the vector pseudo remains as it's needed to encode the register class for the LMUL, so the VL and SEW operands are unused.

``RISCVAsmPrinter`` will then lower the pseudo instructions into real ``MCInsts``.

.. code-block:: nasm

   vsetvli a0, zero, e32, m2, ta, ma
   vadd.vv v8, v8, v10


See also
========

- `2023 LLVM Dev Mtg - Vector codegen in the RISC-V backend <https://youtu.be/-ox8iJmbp0c?feature=shared>`_
- `2023 LLVM Dev Mtg - How to add an C intrinsic and code-gen it, using the RISC-V vector C intrinsics <https://youtu.be/t17O_bU1jks?feature=shared>`_
- `2021 LLVM Dev Mtg “Optimizing code for scalable vector architectures” <https://youtu.be/daWLCyhwrZ8?feature=shared>`_
