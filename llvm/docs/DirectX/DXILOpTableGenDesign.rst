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

Motivation
==========

``DXILLowering`` pass needs to lower the LLVM intrinsics. TableGen file -
``llvm/lib/Target/DirectX/DXIL.td`` - is used to specify the properties of DXIL
Ops including the mapping of each of them to LLVM intrinsics they correspond to, if any.
``utils/hct/hctdb.py`` serves this purpose  in ``DirectXShaderCompier`` repo.
Anologously, ``DXIL.td`` is planned to be the single source of reference
for the properties and LLVM intrinsic mapping of DXIL Ops for DXIL backend
implemetation in ``llvm-project`` repo. It needs to have a rich representation
abilities that TableGen backends (such as ``DXILEmitter``) can rely on. Additionally,
the DXIL Op specification should be easy to read and comprehend.

Design
======

Distilling the essential attributes of DXIL Op from the above (as a start), following
attributes form the core of its specification.

#. ``dxil_opid`` or ``OpCode``
#. ``dxil_class`` or ``OpClass`` - this string is an integral part of the DXIL Op function name and is constructed in the format ``dx.op.<class-name>.<overload-type>``. The DXIL validator checks for any deviation from this for each of the DXIL Op call.
#. ``ops`` - list of operands encapsulating the index and valid (fixed or overload) types
#. ``oload_types`` - Valid overload types of the DXIL op

Each of the LLVM intrinsics maps to an external function represented by a call to an
external function of the form ``dx.op.<class-name>.<overload-type>`` as noted above.

Following is a basic TableGen class structure to encapsulate the mapping of LLVM Intrinsics to DXIL Ops.

.. code-block::

    // Abstraction of DXIL Operation to LLVM Intrinsic mapping
    class DXILOpMappingBase {
      int OpCode = 0;                      // Opcode of DXIL Operation
      DXILOpClass OpClass = UnknownOpClass;// Class of DXIL Operation.
      Intrinsic LLVMIntrinsic = ?;         // LLVM Intrinsic DXIL Operation maps to
      string Doc = "";                     // A short description of the operation
      list<LLVMType> OpTypes = ?;          // Valid types of DXIL Operation in the
                                           // format [returnTy, param1ty, ...]
    }

Various options considered to represent this mapping - keeping the goals of rich
representation and readability stated above - are discussed in the remainder
of the note. The basic difference between these options is the way return and
parameter types are represented for DXIL Ops with valid overload types.
Valid overload types for several DXIL Ops would be over-specified using LLVM's
``llvm_any*_ty`` types. For example, ``half`` and ``float`` are only valid for
DXIL ``Sin`` and would be overspecified using ``llvm_anyfloat_ty``. The options
listed below address the need to model such overload types specific types
precisely for correct code generation. They each provide specifications with
varying levels in (a) ease of readablility and maintainability and
(b) of compactness / richness.

Option 1 : Specify ``OpType`` as a list of valid fixed types.
-------------------------------------------------------------

``OpTypes`` for ``Sin`` may be specified as
``[[llvm_i16, llvm_i32], [llvm_i16, llvm_i32]]``. Repeating such lists for each
of the DXIL Ops - not all of which are unary - reduces readability and increases
the proclivity for errors in specification and maintenance. Even if one can
consider usage of TableGen definitions to create shorthand concrete record
defs for these, above stated problems are barely mitigated.

Option 2 : Specify a function to validate accepted overload types
-----------------------------------------------------------------

Specify a validation function to verify/generate the accepted set of overload
types for DXIL Ops as a field of ``class DXILOpMappingBase``. This function is
expected to be invoked by the TableGen backends to generate relevant ``*.inc``
files with accurate content. Such a specification can provide relief from the
need to specify and maintain long lists of OpTypes. However, having such set
of functions fits rather awkwardly with the TableGen API usage of being able
to directly work with the content of a record. Further, these validation
functions add to the maintenace overlead while not not necessarily making
the specification more readable.

Option 3a : Specify ``OpTypes`` as an override of list valid fixed types
------------------------------------------------------------------------
[**Current strawman implementation**]

Inherit the valid types of the LLVM Intrinsic being lowered as valid for the
DXIL Op, by default. This will reduce the need to specify a ``OpTypes`` list
for those DXIL Ops with the same valid types as the LLVM Intrinsic. In cases
where it is not the case (such as ``Sin``), an optional list that overrides
the default inheritence should be specified. This improves the readability
by eliminating specification of ``OpType`` lists, when not needed. A
relatively small set of precise overload types that are specific to DXIL Ops
are defined to further improve readability. Such types
(e.g., ``llvm_halforfloat_ty``) are defined using standard LLVM MVT
kinds (viz., ``MVT::Other``).

For example, following is the specification of ``Sin`` where the default
type inheritence is overridden via explicit specification of valid overload
types that are more precise.

.. code-block::

    def Sin  : DXILOpMapping<13, unary, int_sin,
                             "Returns sine(theta) for theta in radians.",
                             [llvm_halforfloat_ty, LLVMMatchType<0>]>;

Following is the specification of ``ThreadId`` where the types of the LLVM
intrinsic (``int_dx_thread_id``) are valid for ``dx.op.threadId.*`` and
need not be overridden.

.. code-block::

    def ThreadId : DXILOpMapping<93, threadId, int_dx_thread_id,
                                 "Reads the thread ID">;

The backends can consume such specification without needing to execute
additional validation function or code. Such specification provides better
readability and the necessary type information. It does not completely
eliminate the mechanism of using lists as ``OpTypes``, but DXIL Ops that
need ``OpTypes`` lists could be lesser.

Option 3b : Specify ``OpTypes`` as an exclusion list of valid fixed types
-------------------------------------------------------------------------

Another variant of the Option 3a is to specify an exclusion list. An
exclusion list instead of an override list provides a list of fixed types
not valid for an DXIL Op and thus need to be excluded from a valid overload
type lsit of LLVM Intrinsic. The benefits and downsides of this are the same
as those of specifying an override list as in Option 3a.

Option 4 : Specify accepted overload types as ``Attr`` records
---------------------------------------------------------------

LLVM's TableGen infrastructure defines a base ``class Attr``
(``llvm/include/llvm/IR/Attributes.td``) with an associated
``AttrProperty``. Valid overload types of a DXIL Op can be represented as
``Attr`` records, distinguished via ``AttrProperty`` to model precise types
as needed. This can provide the necessary readability and the richness of
specification. Additionally, the other properties of a DXIL Op (such as the
``bool is_*``) can also be uniformly represented as ``Attr`` records.

Summary
=======

This note discusses various design options that have been explored to implement
a representation of DXIL Ops in ``DXIL.td``. ``DXIL.td`` is intended to serve as
a single source of reference for TableGen backends (such as ``DXILEmitter`` -
specific to DXIL backend), have an accurate and rich specification, be readable
and maintainable. The current implementation employs Option 3a. It is in place,
primarily, to facilitate lowering of new LLVM intrinsics for HLSL functions
being added in the front end. It serves to expose any additional considerations
necessary for an improved design of ``DXIL.td``. The current plan is to design
and implement **Opton 4** to improve readability and maintainablity while
leveraging constructs in LLVM TableGen infrastructure for a potentially rich
specification.
