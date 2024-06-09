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
<https://github.com/microsoft/DirectXShaderCompiler/blob/130877392c263888ef06bab768856d3dab1f1c9a/docs/DXIL.rst#L1978>`_:

#. Using LLVM instructions.
#. Using LLVM External functions. These are represented in LLVM IR as follows:
   * "Standard" LLVM intrinsics (e.g., ``llvm.sin.*``) and
   * HLSL intrinsics (defined as LLVM intrinsics in ``llvm/include/llvm/IR/IntrinsicsDirectX.td``, e.g., ``llvm.dx.*``)

   These are  collectively referred to as `LLVM Intrinsics` in this note.

Following is the complete list of properties of DXIL Ops with the corresponding field name
as used in ``hctdb.py``. A DXIL Op is represented by a set of associated properties. These
are categorized into two groups - viz., those that are (1) consumed in DXIL backend passes
and (2) consumed in other usage scenarios such as validation, DXIL reader, etc.

A. Properties consumed in DXIL backend passes

   1. Name of operation (``dxil_op``)
   2. The generic or HLSL-specific intrinsic that maps to the operation (``llvm_name``).
   3. Unique Integer ID (``dxil_opid``)
   4. Operation Class signifying the name and function signature of the operation (``dxil_class``).
      This string is an integral part of the DXIL Op function name and is constructed in
      the format ``dx.op.<class-name>.<overload-type>``. The DXIL validator checks for any
      deviation from this for each of the DXIL Op call.
   5. List of valid overload types for the operation (``oload_types``).
   6. Required DXIL Version with support for the operation.
   7. A string that documents the operation (``doc``) - This is not strictly necessary but is included
      for readability and documentation of the operation.

B. Properties consumed in other usage scenarios

   1. Required minimum Shader Model (``shader_model``).
   2. Minimum shader model required with translation by linker (``shader_model_translated``)
   3.  List of shader stages applicable to (``shader_stages``), empty for all.
   4.  Memory access attributes of the operation (``fn_attr``).
   5.  Boolean attributes of operation to indicate if it

       * is some kind of a derivative (``is_derivative``)
       * requires gradient calculation (``is_gradient``)
       * is a sampler feedback (``is_feedback``)
       * requires in-wave, cross-lane functionality (``is_wave``)
       * requires that all of its inputs are uniform across the wave (``requires_uniform_inputs``).
       * is a barrier operation (``is_barrier``).

Motivation
==========

DXIL backend passes depend on various properties of DXIL Operations. For example, ``DXILLowering``
pass will need information such as the DXIL operation an LLVM intrinsic is to be lowered to,
along with valid overload and parameter types etc. The TableGen file -
``llvm/lib/Target/DirectX/DXIL.td`` - is used to represent DXIL Operations
by specifying their properties listed above. ``DXIL.td`` is designed to be the single source
of reference of DXIL Operations for DXIL backend implementation in ``llvm-project`` repo -
analogous to ``hctdb.py`` for ``DirectXShadeCompiler`` repo. It needs to have a rich
representation capabilities that TableGen backends (such as ``DXILEmitter``) can rely on.
Additionally, the DXIL Op specification should be easy to read and comprehend.

This note focuses on specification of the set of properties consumed by DXIL backend
passes identified above in category A. Any of the properties from category B are expected to be
included as deemed necessary during implementation.

Design
======

1. Each DXIL Operation is represented as a TableGen record. The name of each of the records
   signifies operation name.
2. The LLVM Intrinsic that maps to the operation is represented using ``Intrinsic::*``.
3. The unique operation id is represented by an integer.
4. DXIL Operation Class is represented as follows

   .. code-block::

        // Abstraction of DXIL Operation class.
        // It encapsulates an associated function signature viz.,
        // returnTy(param1Ty, param2Ty, ...) represented as a list of LLVMTypes.
        // DXIL Ops that belong to a DXILOpClass record the signature of that DXILOpClass

        class DXILOpClass<list<LLVMType> OpSig> {
          list<LLVMType> OpSignature = OpSig;
        }

   Concrete operation classes, such as ``unary`` are defined inheriting from ``DXILOpClass``.
5. Valid overload types are represented as a list of ``LLVMType``.
6. Concrete records of DXIL versions and are defined by inheriting from the class

   .. code-block::

        // Abstract class to represent major and minor version values
        class Version<int major, int minor> {
          int Major = major;
          int Minor = minor;
        }

7. A documentation string for the operation.


A DXIL Operation is represented by the following TableGen class by encapsulating the various
TableGen representations of its properties described above.

.. code-block::

  // Abstraction DXIL Operation
  class DXILOpPropertiesBase {
    int OpCode = 0;                           // Opcode of DXIL Operation
    DXILOpClass OpClass = UnknownOpClass;     // Class of DXIL Operation.
    Intrinsic LLVMIntrinsic = ?;              // LLVM Intrinsic DXIL Operation maps to
    list<LLVMType> OpOverloadTypes = ?; // Valid overload type
                                              // of DXIL Operation
    Version DXILVer = ?;                      // Min DXIL version
    string Doc = "";                          // A short description of the operation
  }


The following convenience class, definitions of ``unary`` and ``DXVer1_0`` are used to
illustrate the definitions of ``Sin`` and ``Cos`` operations:

  .. code-block::

      class DXILOpProperties<int opCode,
                    Intrinsic intrinsic,
                    list<LLVMType> overloadTypes,
                    string doc> : DXILOpPropertiesBase {
        int OpCode = opCode;
        Intrinsic LLVMIntrinsic = intrinsic;
        list<LLVMType> OpOverloadTypes = overloadTypes;
        string Doc = doc;
      }

      def unary : DXILOpClass<[llvm_any_ty, LLVMMatchType<0>]>;
      def DXVer1_0 : Version<1, 0>;

      let OpClass = unary, DXILVer = DXVer1_0 in {
        def Cos  : DXILOpProperties<12, int_cos, [llvm_half_ty, llvm_float_ty],
                                   "Returns cosine(theta) for theta in radians.">;
        def Sin  : DXILOpProperties<13, int_sin, [llvm_half_ty, llvm_float_ty],
                                   "Returns sine(theta) for theta in radians.">;
      }

Summary
=======

This note sketches the design of a readable and maintainable TableGen specification of
DXIL Ops in ``DXIL.td`` intended to serve as a single source of reference for TableGen
backends (such as ``DXILEmitter``) that generates C++ representations used in DXIL
backend passes.
