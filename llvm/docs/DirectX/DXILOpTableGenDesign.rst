==============================================================
Specification of DXIL Operations using TableGen Representation
==============================================================
.. contents::
   :local:

.. toctree
   :hidden

Introduction
============

`DirectXShaderCompiler <https://github.com/microsoft/DirectXShaderCompiler>`_
encapsulates, among other information, various DXIL Operations in
`hctdb.py <https://github.com/microsoft/DirectXShaderCompiler/blob/main/utils/hct/hctdb.py>`_.
DXIL Operations are represented in one of the following `two ways
<https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#operations>`_:

#. Using LLVM instructions.
#. Using LLVM External functions. These are represented in LLVM IR as follows:

   * "Standard" LLVM intrinsics (e.g., ``llvm.sin.*``) and
   * HLSL intrinsics (defined as LLVM intrinsics in ``llvm/include/llvm/IR/IntrinsicsDirectX.td``, e.g., ``llvm.dx.*``)

   These are  collectively referred to as `LLVM Intrinsics` in this note.

Following is the complete list of attributes of DXIL Ops with the corresponding field name
as used in ``hctdb.py``. A DXIL Op is represented by a set of associated attributes. These
are consumed in DXIL backend passes as well as in other usage scenarios such as validation, 
DXIL reader, etc.

A. Attributes consumed in DXIL backend passes

   1. Name of operation (``dxil_op``)
   2. A string that documents the operation (``doc``) - This is not strictly necessary but is included
      for readability and documentation of the operation.
   3. The generic or HLSL-specific intrinsic that maps to the operation (``llvm_name``).
   4. Unique Integer ID (``dxil_opid``)
   5. Operation Class signifying the name and function signature of the operation (``dxil_class``).
      This string is an integral part of the DXIL Op function name and is constructed in
      the format ``dx.op.<class-name>.<overload-type>``. Each DXIL Op call target function name 
      is required to conform to this format per existing contract with the driver.
   6. List of valid overload types for the operation (``oload_types``).
   7. Required DXIL Version with support for the operation.
   8. Required minimum Shader Model (``shader_model``).
   9. Minimum shader model required with translation by linker (``shader_model_translated``)
   10.  List of shader stages applicable to (``shader_stages``), empty, if applicable to all stages.
   11.  Memory access attributes of the operation (``fn_attr``).
   12.  Boolean attributes of operation to indicate if it

        * is some kind of a derivative (``is_derivative``)
        * requires gradient calculation (``is_gradient``)
        * is a sampler feedback (``is_feedback``)
        * requires in-wave, cross-lane functionality (``is_wave``)
        * requires that all of its inputs are uniform across the wave (``requires_uniform_inputs``).
        * is a barrier operation (``is_barrier``).

Motivation
==========

DXIL backend passes depend on various attributes of DXIL Operations. For example, ``DXILOpLowering``
pass will need information such as the DXIL operation an LLVM intrinsic is to be lowered to,
along with valid overload and parameter types etc. The TableGen file -
``llvm/lib/Target/DirectX/DXIL.td`` - is used to represent DXIL Operations
by specifying their attributes listed above. ``DXIL.td`` is designed to be the single source
of reference of DXIL Operations primarily for the implementation of passes in DXIL backend in 
``llvm-project`` repo - analogous to ``hctdb.py`` for ``DirectXShadeCompiler`` repo. However, 
the current design does not intend to encapsulate various validation rules, present in ``hctdb.py``, 
but do not pertain to DXIL Operations. It needs to have a rich representation capabilities that 
TableGen backends (such as ``DXILEmitter``) can rely on. Additionally, the DXIL Op specification 
should be easy to read and comprehend.

This note provides the design of the specification DXIL Ops as TableGen class ``DXILOp``
by specifying its attributes identified above.

DXIL Operation Specification
============================

The DXIL Operation is represented using the TableGen class ``DXILOp``. The DXIL operation
attributes are specified as fields of the ``DXILOp`` class as described below.

1. Each DXIL Operation is represented as a TableGen record. The name of each of the records
   signifies operation name.
2. A documentation string for the operation.
3. The LLVM Intrinsic that maps to the operation is represented as ``Intrinsic`` defined in
   `Intrinsics.td <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Intrinsics.td>`_.
4. The unique operation id is represented by an integer.
5. DXIL Operation Class is represented as follows

   .. code-block::

        // Abstraction of DXIL Operation class.
        class DXILOpClass;

   Concrete operation records, such as ``unary`` are defined by inheriting from ``DXILOpClass``.
6. Return and argument types of the operation are represented as ``dag``s using the
   special markers ``out`` and ``ins``. An overload type, if supported by the operation, is
   denoted as the positional type ``dxil_overload_ty`` in the argument or in the result, where
   ``dxil_overload_ty`` is defined to be synonymous to ``llvm_any_ty``.

   .. code-block::

      defvar dxil_overload_ty = llvm_any_ty


7. Valid overload types and shader stages predicated on Shader Model version are specified
   as a list of ``Constraint`` records. Representation of ``Constraints`` class is described
   a later section.
8. Various attributes of the DXIL Operation that are not predicated on Shader Model version
   are represented as a ``dag`` using the special marker ``attrs``. Representation of ``Attributes`` 
   class is described in a later section.

A DXIL Operation is represented by the following TableGen class by encapsulating the various
TableGen representations of its attributes described above.

.. code-block::

   // Abstraction DXIL Operation
   class DXILOp {
     // A short description of the operation
     string Doc = "";

     // Opcode of DXIL Operation
     int OpCode = 0;

     // Class of DXIL Operation.
     DXILOpClass OpClass = UnknownOpClass;

     // LLVM Intrinsic DXIL Operation maps to
     Intrinsic LLVMIntrinsic = ?;

     // Dag containing the arguments of the op. Default to 0 arguments.
     dag arguments = (ins);

     // Results of the op. Default to 0 results.
     dag result = (out);

     // List of constraints predicated on Shader Model version
     list<SMVersionConstraints> sm_constraints;

     // Non-predicated operation attributes
     dag attrtibutes = (attrs);
     Version DXILVersion = ?;
   }

Constraint Specification
========================

DXIL Operation attributes such as valid overload types and valid shader stages are
predicated on Shader Model version. These are represented as list of constrained
attributes.

Following is the definition of a generic constraint and the associated predicate

.. code-block::

   // Primitive predicate
   class Pred;

   // Generic constraint
   class Constraint<Pred pred> {
     Pred predicate = pred;
   }

Shader Model version is represented as follows:

.. code-block::

   // Abstract class to represent major and minor version values
   class Version<int major, int minor> {
     int Major = major;
     int Minor = minor;
   }

   // Valid Shader model version records

   // Definition of Shader Model 6.0 - 6.8 and DXIL Version 1.0 - 1.8
   foreach i = 0...8 in {
     def SM6_#i : Version<6, i>;
     def DX1_#i : Version<1, i>;
   }

A shader model version predicate class is defined as

.. code-block::

   class SMVersion<Version ver> : Pred {
     Version SMVersion = ver;
   }

A constraint class to represent overload types and shader stages predicated on shader
model version is defined as

.. code-block::

   class SMVersionConstraints<SMVersion smver, dag oloads, dag stages> : Constraint<smver> {
     dag overload_types = oloads;
     dag stage_kinds = stages;
   }

The ``dag overload_types`` and ``dag shader_kinds`` use a special markers ``overloads``
and ``stages``, respectively.

Examples of Constraints
-----------------------

Consider a DXIL Operation that is valid in Shader Model 6.2 and later,

1. with valid overload types ``half``, ``float``, ``i16`` and ``i32``
2. is valid for stages ``pixel`` and ``compute``
3. with valid overload types ``double`` and ``i614`` if Shader Model version 6.3 and later
4. is valid for all stages if Shader Model version 6.3 and later

This is represented as

.. code-block::

   [SMVersionConstraints<SMVersion<SM6_2>,
                          (overloads llvm_half_ty, llvm_float_ty, llvm_i16_ty, llvm_i32_ty),
                          (stages pixel, compute)>,
    SMVersionConstraints<SMVersion<SM6_3>,
                          (overloads llvm_half_ty, llvm_float_ty, llvm_double_ty,
                                 llvm_i16_ty, llvm_i32_ty, llvm_i64_ty),
                          (stages allKinds)>];

Consider a DXIL operation that is valid in Shader Model version 6.2 and later,

1. with no overload types, i.e., all argument typess and result type are fixed.
2. is valid for all stages.

This is represented as

.. code-block::

     [SMVersionConstraints<SMVersion<SM6_2>, (overloads), (stages allKinds)>];


Specifying attributes predicated on Shader Model version using the single field 
``sm_constraints`` not only allows for all of them to be specified together but
also allows for a single place to specify minimum shader model version that supports
the operation. Thus, a separate fiels is not needed to specify minimum shader model 
version.

Attribute Specification
=======================

DXIL Operation attributes that are not predicated on any constraint, are represented as
a ``dag`` of Attribute records of the following abstract ``DXILAttributes`` class.

.. code-block::

  class DXILAttributes;

Following example records represent memory attributes 

.. code-block::

  def ReadOnly : DXILOpAttributes;
  def ReadNone : DXILOpAttributes;

DXIL Operation Specification Example
====================================
Following illustrates the specification of the DXIL Op ``Sin``

.. code-block::

  def Sin  : DXILOp {
    let Doc ="Returns sine(theta) for theta in radians.";
    let OpCode = 13;
    let OpClass = unary;
    let LLVMIntrinsic = int_sin;
    let arguments = (ins LLVMMatchType<0>);
    let result = (out dxil_overload_ty);
    let sm_constraints = [SMVersionConstraints<SMVersion<SM6_0>,
                          (overloads llvm_half_ty, llvm_float_ty),
                          (stages allKinds)>];
    let attributes = (attrs ReadNone);
    let DXILVersion = DX1_0;
  }

Summary
=======

This note sketches the design of a readable and maintainable TableGen specification of
DXIL Ops in ``DXIL.td`` intended to serve as a single source of reference for TableGen
backends (such as ``DXILEmitter``) that generate C++ representations used in DXIL
backend passes.
