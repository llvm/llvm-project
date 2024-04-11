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

#. Using LLVM instructions
#. Using LLVM External functions. These are represented in LLVM IR as follows:

   * "Standard" LLVM intrinsics (e.g., ``llvm.sin.*``) and
   * HLSL intrinsics (defined as LLVM intrinsics in ``llvm/include/llvm/IR/IntrinsicsDirectX.td``, e.g., ``llvm.dx.*``)

These are  collectively referred to as `LLVM Intrinsics` in this note.

DXIL Ops, as currently represented in ``hctdb.py`` have the following attributes

#. ``name`` - A short, unique name
#. ``llvm_id`` - ID of LLVM instruction. This is just an arbitrary, yet fixed, number that indicates LLVM's ``CallInst`` for all LLVM intrinsics
#. ``llvm_name`` - String name of LLVM instruction type
#. ``is_dxil_op`` - A bool indicating whether this is a call into a built-in DXIL function
#. ``dxil_op`` - String name of DXIL operation
#. ``dxil_opid`` - ID of DXIL operation
#. ``dxil_class`` - String name of the opcode class
#. ``category`` - String classification for this instruction
#. ``doc`` - String documentation description of this instruction
#. ``remarks`` - String long-form remarks on this instruction
#. ``ops`` - List of operands that this instruction takes
#. ``is_allowed`` - Bool indicating whether this instruction is allowed in a DXIL program
#. ``oload_types`` - String denoting overload types if applicable (e.g., "hf", "iwl")
#. ``fn_attr`` - Attribute shorthand strings: rn=does not access memory,ro=only reads from memory,
#. ``is_deriv`` - Bool indicating whether this is some kind of derivative
#. ``is_gradient`` - Bool indicating whether this requires a gradient calculation
#. ``is_feedback`` - Bool indicating whether this is a sampler feedback op
#. ``is_wave``  - Bool indicating whether this requires in-wave, cross-lane functionality
#. ``requires_uniform_inputs``  - Bool indicating whether this operation requires that all of its inputs are uniform across the wave
#. ``is_barrier``  - Bool indicating whether this is a barrier operation
#. ``shader_stages`` - shader stages to which this applies, empty for all.
#. ``shader_model`` - minimum shader model required (e.g., 6, 0)
#. ``inst_helper_prefix`` - None
#. ``fully_qualified_name_prefix`` - Constant string ``"hlsl::OP::OpCode"``
#. ``is_dxil_op`` - Bool that evaluates (dxil_op != "") indicating whether this is a DXIL operation
#. ``is_reserved`` - Bool that evaluates (dxil_class == "Reserved")
#. ``shader_model_translated`` - minimum shader model required with translation by linker
#. ``props`` - extra properties

Core DXIL Operation information is encapsulated in ``utils/hct/hctdb.py``. Additional
refinements of this information, such as supported overload types specific to
Shader Model version, is embedded in various other source locations in
`DirectXShaderCompiler <https://github.com/microsoft/DirectXShaderCompiler>`_
repo. Additional conditions that refine the DXIL Op properties are also encoded
in the DXIL Validator component.

Motivation
==========

``DXILLowering`` pass needs to lower the LLVM intrinsics. TableGen file -
``llvm/lib/Target/DirectX/DXIL.td`` - is used to specify the properties of DXIL
Ops including the mapping of each of them to LLVM intrinsics they correspond to,
if any. This purpose is served by ``utils/hct/hctdb.py`` in ``DirectXShaderCompiler``
repo. Analogously, ``DXIL.td`` is planned to be the single source of reference
for the properties and LLVM intrinsic mapping of DXIL Ops for DXIL backend
implementation in ``llvm-project`` repo. Additionally, the refinements of DXIL Op
properties based on aspects such as target Shader Model version, Shader kind (viz.,
Compute, Pixel, Vertex etc) should also be represented in TableGen specification
of DXIL Ops - as much as possible. This will allow generation of valid DXIL code
in all passes and potentially not require (or reduce the complexity of) a post-compile
validation step that ensures valid DXIL binary.

As a result, specification in ``DXIL.td`` needs to have a rich representation
abilities that TableGen backends (such as ``DXILEmitter``) can rely on. The DXIL
Op specification should be declarative - as much as possible - making it
easy to comprehend and amenable to specification of constraints that refine
DXIL Op properties.

.. _DXIL Operation Attributes:

DXIL Operation Attributes
=========================

Distilling the essential attributes of DXIL Op from the above, following
attributes form the core of its specification.

#. ``dxil_opid`` or ``OpCode``
#. ``dxil_class`` or ``OpClass`` - this string is an integral part of the DXIL Op
   function name and is constructed in the format ``dx.op.<class-name>.<overload-type>``.
   The DXIL validator checks for any deviation from this for each of the DXIL Op call.
#. ``ops`` - list of operands encapsulating the index and valid (fixed or overload) types
#. ``oload_types`` - Valid overload types of the DXIL op
#. Rest of the attributes represented using ``is_*`` booleans along with
   ``shader_model_translated`` and ``shader_stages``

Each of the LLVM intrinsics maps to an function represented by a call to an
external function of the form ``dx.op.<class-name>.<overload-type>`` as noted above.

TableGen Specification
======================

Shader Model
^^^^^^^^^^^^

DirectX Shader Models are distinguished by their ``major.minor`` number.
This is represented by the following TableGen class.

.. code-block::

  class DXILShaderModel<int major, int minor> {
    int Major = major;
    int Minor = minor;
  }

Each of the valid shader models is defined as TableGen records. For
example following is the definition of SM 6.0 and SM 6.2,

.. code-block::

  // Shader Model 6.x
  def SM6_0 : DXILShaderModel<6, 0>;
  def SM6_2 : DXILShaderModel<6, 2>;

DXIL Class
^^^^^^^^^^

Each DXIL Op belongs to a class represented by ``dxil_class`` field value. A
DXIL class represents DXIL Ops with the same function prototype (or signature).
This is represented using the following TableGen class.

.. code-block::

  class DXILOpClass<list<LLVMType> OpSig> {
    list<LLVMType> OpSignature = OpSig;
  }

Each of the valid classes is represented by a concrete TableGen record.
For example, following is the definition of the ``unary`` class

.. code-block::

  def unary : DXILOpClass<[llvm_any_ty, LLVMMatch<0>]>;


Overload Types
^^^^^^^^^^^^^^

Each DXIL Op has a set of valid overload types denoted by ``oload_types``.
Valid overload types of a DXIL OP are represented as a list. However, overload
types supported by DXIL Ops may vary depending on minimum target shader model
version. So, the list of supported overload types are tagged with the minimum
shader model in which they are valid for the DXIL Op being specified.
Following TableGen class is defined to encapsulate such as representation.

.. code-block::

  class DXILOpOverload<DXILShaderModel minsm, list<LLVMType> overloads> {
    DXILShaderModel ShaderModel = minsm;
    list<LLVMType> OpOverloads = overloads;
  }

Specification of DXIL Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specification of DXIL Op is abstracted using the following TableGen class to
represent the core attributes outlined in  `DXIL Operation Attributes`_ section.

.. code-block::

  // Abstraction DXIL Operation to LLVM intrinsic
  class DXILOpMappingBase {
    int OpCode = 0;                      // Opcode of DXIL Operation
    DXILOpClass OpClass = UnknownOpClass;// Class of DXIL Operation.
    Intrinsic LLVMIntrinsic = ?;         // LLVM Intrinsic DXIL Operation maps to
    list<DXILOpOverload> OpOverloadTypes = ?; // Valid overload type
                                       // of DXIL Operation
    string Doc = "";                     // A short description of the operation
  }

Following is a convenience TableGen class that inherits from ``DXILOpMappingBase``
with templatized parameters. It is used to define various DXIL Ops.

.. code-block::

  class DXILOpMapping<int opCode,
                    Intrinsic intrinsic,
                    list<DXILOpOverload> overloadTypes,
                    string doc> : DXILOpMappingBase {
    int OpCode = opCode;
    Intrinsic LLVMIntrinsic = intrinsic;
    list<DXILOpOverload> OpOverloadTypes = overloadTypes;
    string Doc = doc;
  }

The DXIL Op ``Sin`` is defined as follows:

.. code-block::

  let OpClass = unary in
    def Sin  : DXILOpMapping<13, int_sin,
                             [DXILOpOverload<SM6_3, [llvm_half_ty, llvm_float_ty]>,
                              DXILOpOverload<SM6_0, [llvm_float_ty]>],
                             "Returns sine(theta) for theta in radians.">;


Note that validity of overload type ``float`` in SM 6.0 and later, and
that of ``half`` and ``float`` in SM 6.2 and later, is specified.

Summary
=======

This note describes design and implementation of a TableGen representation of
DXIL Ops in ``DXIL.td``. ``DXIL.td`` is intended to (a) serve as a single source
of reference for TableGen backends (such as ``DXILEmitter``- specific to DXIL
backend), (b) have an accurate and rich specification including the ability to
represent refinement constraints, and (c) be declarative as much as possible for
readability and maintainability.