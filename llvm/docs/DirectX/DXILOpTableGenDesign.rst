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

Following is the complete list of properties of DXIL Ops with the corresponding field name
as used in ``hctdb.py``. A DXIL Op is represented by a set of associated properties. These
are consumed in DXIL backend passes as well as in other usage scenarios such as validation,
DXIL reader, etc.

A. Properties consumed in DXIL backend passes

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

DXIL backend passes depend on various properties of DXIL Operations. For example, ``DXILOpLowering``
pass will need information such as the DXIL operation an LLVM intrinsic is to be lowered to,
along with valid overload and argument types etc. The TableGen file -
``llvm/lib/Target/DirectX/DXIL.td`` - is used to represent DXIL Operations
by specifying their properties listed above. ``DXIL.td`` is designed to be the single source
of reference of DXIL Operations primarily for the implementation of passes in DXIL backend in
``llvm-project`` repo - analogous to ``hctdb.py`` for ``DirectXShadeCompiler`` repo. However,
the current design does not intend to encapsulate various validation rules, present in ``hctdb.py``,
but do not pertain to DXIL Operations. It needs to have a rich representation capabilities that
TableGen backends (such as ``DXILEmitter``) can rely on. Additionally, the DXIL Op specification
should be easy to read and comprehend.

This note provides the design of the specification DXIL Ops as TableGen class ``DXILOp``
by specifying its properties identified above.

DXIL Operation Specification
============================

The DXIL Operation is represented using the TableGen class ``DXILOp``. The DXIL operation
properties are specified as fields of the ``DXILOp`` class as described below.

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
6. A non-``void`` return type of the operation is represented as a list with one ``LLVMType``. 
   A ``void`` return is represented as a null-list, ``[]``.
7. Non-zero count of operation arguments are represented as a list of ``LLVMType`` with each type
   corresponding to the argument position. An overload type, if supported by the operation, is
   denoted as the positional type ``dxil_overload_ty`` in the argument or in the result, where
   ``dxil_overload_ty`` is defined to be synonymous to ``llvm_any_ty``.

   .. code-block::

      defvar dxil_overload_ty = llvm_any_ty

   Use of TableGen class ``LLVMMatchType`` is supported to match type of another argument.

8. Valid overload types and shader stages predicated on Shader Model version are specified
   as a list of ``Constraints`` records. Representation of ``Constraints`` class is described
   a later section.
9. Various attributes of the DXIL Operation are represented as a ``list`` of ``Attribute`` class 
   records. Representation of ``Attribute`` class is described in a later section.

10. DXIL Version is represented as a record of class ``DXILVersion``, whose details are provided
    in a later section.

A DXIL Operation is represented by the following TableGen class by encapsulating the various
TableGen representations of its properties described above.

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
     Intrinsic LLVMIntrinsic = ? ;

     // List of argument types of the op. Default to 0 arguments.
     list<LLVMType> arguments = [];

     // List of result types of the op. Default to 0 results.
     list<LLVMType> result = [];

     list<Constraints> constraints = [];

     // Operation attributes
     list<DXILAttribute> attributes = [];

     Version DXILVersion = ? ;
   }

Version Specification
=====================

A ``Version`` class encapsulating ``Major`` and ``Minor`` version number is defined
as follows:

.. code-block::

   // Abstract class to represent major and minor version values
   class Version<int major, int minor> {
     int Major = major;
     int Minor = minor;
   }


Concrete representations of valid Shader Model and DXIL versions are defined as follows:

.. code-block::

   // Definition of Shader Model 6.0 - 6.8 and DXIL Version 1.0 - 1.8
   foreach i = 0...8 in {
     def SM6_#i : Version<6, i>;
     def DX1_#i : Version<1, i>;
   }

Shader Stage Specification
==========================

Various shader stages such as ``compute``, ``pixel``, ``vertex``, etc., are represented as
follows

.. code-block::

   // Shader stages
   class ShaderStage;

   def compute : ShaderStage;
   def pixel : ShaderStage;
   def vertex : ShaderStage;
   ...

Constraint Specification
========================

DXIL Operation properties such as valid overload types and valid shader stages are
predicated on Shader Model version. These are represented as list of constraints.

Following is the class representing a predicate and a constraint class representing
a ``list<Pred> l`` of properties applicable to the DXIL Operation predicated on
``Pred p``.

.. code-block::

   // Primitive predicate
   class Pred;

   // Generic constraint
   class Constraints<Pred p, list<Pred> l = []> : Pred {
     Pred pred = p;
     list<Pred> constraints = l;
   }


A shader model version predicate class is defined as

.. code-block::

   class SMVersion<Version ver> : Pred {
     Version SMVersion = ver;
   }

Overload type predicates are represented as records of the class

.. code-block::

   class Overloads<list<LLVMType> tys> : Pred {
    list<LLVMType> overload_types = tys;
  }

Shader Stage predicates are represented as records of class

.. code-block::

   class Stages<list<ShaderStage> st> : Pred {
    list<ShaderStage> stage_kinds = st;
   }

Overload and shader stages constrained by Shader Model version are expressed by
composing the above predicate records.

If no constraints are specified for a DXIL operation, it is assumed to 

a) be supported in Shader Model 6.0 and later.
b) have no overload types
c) be supported in all shader stage kinds

If a constraint is specified, one or both of the Overload type and shader kind 
constraints can be omitted when appropriate.

Examples of Constraint Specification
------------------------------------

1. Consider a DXIL Operation with the following properties,

   1. In Shader Model 6.2 and later

      a) supports overload types ``half``, ``float``, ``i16`` and ``i32`` and
      b) is valid for stages ``pixel`` and ``compute``

   2. In Shader Model 6.3 and later

      a) supports additional valid overload types ``double`` and ``i64`` and
      b) is valid for all stages

   The constraints of such an operation are represented as

   .. code-block::

      constraints = [Constraints<SMVersion<SM6_2>,
                    [Overloads<[llvm_half_ty, llvm_float_ty, llvm_i16_ty, llvm_i32_ty]>,
                     Stages<[pixel, compute]>]>,
       Constraints<SMVersion<SM6_3>,
                     [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty,
                                    llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]];

   Note that ``Stage<>`` predicate is not specified for the constraint predicated for 
   ``SM6_3`` to signify that the operation is valid in all shader stages in Shader Model 
   version 6.3.

2. Consider a DXIL operation that is valid in Shader Model version 6.2 and later,

   1. with no overload types, i.e., all argument types and result type are fixed.
   2. is valid for all stages.

   This is represented as

   .. code-block::

        [Constraints<SMVersion<SM6_2>, []];

Specifying properties predicated on Shader Model version using the field
``constraints`` not only allows for all of them to be specified together but
also allows for a single place to specify minimum shader model version that supports
the operation. Thus, a separate field is not needed to specify minimum shader model
version.

Attribute Specification
=======================

DXIL Operation attributes that are not predicated on any constraint, are represented as
a ``list`` of Attribute records of ``DXILAttributes`` class.

.. code-block::

  class DXILAttributes;

Following example records represent memory attributes

.. code-block::

  def ReadOnly : DXILOpAttributes;
  def ReadNone : DXILOpAttributes;

DXIL Operation Specification Examples
=====================================

A convenience class ``DXILOpAndCLass`` is defined to specify the minimal properties
of a ``DXILOp`` as follows.

.. code-block::

  class DXILOpAndClass<int opcode, DXILOpClass opcalss> : DXILOp {
    int OpCode = opcode;
    DXILOpClass OpClass = opcalss;
  }

Following examples illustrate the specification of some of the DXIL Ops

``Sin`` operation valid in SM 6.0 for all shader stages but with overload types constraints.

.. code-block::

  def Sin : DXILOpAndClass<13, unary> {
    let Doc = "Returns sine(theta) for theta in radians.";
    let LLVMIntrinsic = int_sin;
    let arguments = [LLVMMatchType<0>];
    let result = [dxil_overload_ty];
    let constraints = [
      Constraints<SMVersion<SM6_0>, [Overloads<[llvm_half_ty, llvm_float_ty]>]>
    ];
    let attributes = [ReadNone];
    let DXILVersion = DX1_0;
  }


``FlattenedThreadIdInGroup`` operation valid in SM 6.0 with shader stage validity
constraints; with  fixed argument type, hence no valid overload type and ``void`` 
return type, hence ``result`` field not specified.

.. code-block::

   def FlattenedThreadIdInGroup : DXILOpAndClass<96, flattenedThreadIdInGroup> {
     let Doc = "Provides a flattened index for a given thread within a given "
               "group (SV_GroupIndex)";
     let LLVMIntrinsic = int_dx_flattened_thread_id_in_group;
     let arguments = [llvm_i32_ty];
     let constraints =
         [Constraints<SMVersion<SM6_0>,
                      [Stages<[compute, mesh, amplification, node]>]>];
     let attributes = [ReadNone];
     let DXILVersion = DX1_0;
   }

``RawBufferStore`` operation with different valid overload types for SM 6.2+ and SM 6.3+.

.. code-block::

   def RawBufferStore : DXILOpAndClass<140, rawBufferStore> {
     let Doc = "Writes to a RWByteAddressBuffer or RWStructuredBuffer.";
     let LLVMIntrinsic = int_rwbuffer_store;
     let arguments = [llvm_i32_ty, dxil_resource_ty, llvm_i32_ty, llvm_i32_ty, dxil_overload_ty,
                      dxil_overload_ty, dxil_overload_ty, dxil_overload_ty, llvm_i8_ty, llvm_i32_ty];
     let constraints = [Constraints<SMVersion<SM6_2>,
                          [Overloads<[llvm_half_ty, llvm_float_ty, llvm_i16_ty, llvm_i32_ty]>]>,
                        Constraints<SMVersion<SM6_3>,
                          [Overloads<[llvm_half_ty, llvm_float_ty, llvm_double_ty,
                                 llvm_i16_ty, llvm_i32_ty, llvm_i64_ty]>]>];
     let DXILVersion = DX1_2;
   }


Summary
=======

This note sketches the design of a readable and maintainable TableGen specification of
DXIL Ops in ``DXIL.td`` intended to serve as a single source of reference for TableGen
backends (such as ``DXILEmitter``) that generate C++ representations used in DXIL
backend passes.
