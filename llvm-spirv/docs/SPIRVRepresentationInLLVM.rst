================================
SPIR-V representation in LLVM IR
================================
.. contents::
   :local:

Overview
========

As one of the goals of SPIR-V is to `"map easily to other IRs, including LLVM
IR" <https://cvs.khronos.org/svn/repos/SPIRV/trunk/specs/SPIRV.html#_goals>`_,
most of SPIR-V entities (global variables, constants, types, functions, basic
blocks, instructions) have straightforward counterparts in LLVM. Therefore the
focus of this document is those entities in SPIR-V which do not map to LLVM in
an obvious way. These include:

 * SPIR-V types mapped to LLVM types
 * SPIR-V instructions mapped to LLVM function calls
 * SPIR-V extended instructions mapped to LLVM function calls
 * SPIR-V builtins variables mapped to LLVM global variables
 * SPIR-V instructions mapped to LLVM metadata
 * SPIR-V types mapped to LLVM opaque types
 * SPIR-V decorations mapped to LLVM metadata or named attributes

SPIR-V Types Mapped to LLVM Types
=================================
Limited to this section, we define the following common postfix.

* {Access} - Postifix indicating the access qualifier.
{Access} take integer literal values which are defined by the SPIR-V spec.

OpTypeImage
-----------
OpTypeImage is mapped to LLVM opaque type
spirv.Image._{SampledType}_{Dim}_{Depth}_{Arrayed}_{MS}_{Sampled}_{Format}_{Access}
and mangled as __spirv_Image__{SampledType}_{Dim}_{Depth}_{Arrayed}_{MS}_{Sampled}_{Format}_{Access},

where

* {SampledType}={float|half|int|uint|void} - Postfix indicating the sampled data type
  - void for unknown sampled data type
* {Dim} - Postfix indicating the dimension of the image
* {Depth} - Postfix indicating whether the image is a depth image
* {Arrayed} - Postfix indicating whether the image is arrayed image
* {MS} - Postfix indicating whether the image is multi-sampled
* {Sampled} - Postfix indicating whether the image is associated with sampler
* {Format} - Postfix indicating the image format

Postfixes {Dim}, {Depth}, {Arrayed}, {MS}, {Sampled} and {Format} take integer
literal values which are defined by the SPIR-V spec.

OpTypeSampledImage
------------------
OpTypeSampledImage is mapped to LLVM opaque type
spirv.SampledImage._{Postfixes} and mangled as __spirv_SampledImage__{Postfixes},
where {Postfixes} are the same as the postfixes of the original image type, as
defined above in this section.

OpTypePipe
----------
OpTypePipe is mapped to LLVM opaque type
spirv.Pipe._{Access} and mangled as __spirv_Pipe__{Access}.

Other SPIR-V Types
------------------
* OpTypeEvent
* OpTypeDeviceEvent
* OpTypeReserveId
* OpTypeQueue
* OpTypeSampler
* OpTypePipeStorage (SPIR-V 1.1)
The above SPIR-V types are mapped to LLVM opaque type spirv.{TypeName} and
mangled as __spirv_{TypeName}, where {TypeName} is the name of the SPIR-V
type with "OpType" removed, e.g., OpTypeEvent is mapped to spirv.Event and
mangled as __spirv_Event.

SPIR-V Instructions Mapped to LLVM Function Calls
=================================================

Some SPIR-V instructions which can be included in basic blocks do not have
corresponding LLVM instructions or intrinsics. These SPIR-V instructions are
represented by function calls in LLVM. The function corresponding to a SPIR-V
instruction is termed SPIR-V builtin function and its name is `IA64 mangled
<https://mentorembedded.github.io/cxx-abi/abi.html#mangling>`_ with extensions
for SPIR-V specific types. The unmangled name of a SPIR-V builtin function
follows the convention

.. code-block:: c

  __spirv_{OpCodeName}{_OptionalPostfixes}

where {OpCodeName} is the op code name of the SPIR-V instructions without the
"Op" prefix, e.g. EnqueueKernel. {OptionalPostfixes} are optional postfixes to
specify decorations for the SPIR-V instruction. The SPIR-V op code name and
each postfix does not contain "_".

SPIR-V builtin functions accepts all argument types accepted by the
corresponding SPIR-V instructions. The literal operands of extended
instruction are mapped to function call arguments with type i32.

Optional Postfixes for SPIR-V Builtin Function Names
----------------------------------------------------

SPIR-V builtin functions corresponding to the following SPIR-V instructions are
postfixed following the order specified as below:

 * Instructions having identical argument types but different return types are postfixed with "_R{ReturnType}" where
    - {ReturnType} = {ScalarType}|{VectorType}
    - {ScalarType} = char|uchar|short|ushort|int|uint|long|ulong|half|float|double|bool
    - {VectorType} = {ScalarType}{2|3|4|8|16}
 * Instructions with saturation decoration are postfixed with "_sat"
 * Instructions with floating point rounding mode decoration are postfixed with "_rtp|_rtn|_rtz|_rte"

SPIR-V Builtin Conversion Function Names
----------------------------------------

The unmangled names of SPIR-V builtin conversion functions follow the convention:

.. code-block:: c

  __spirv_{ConversionOpCodeName}_R{ReturnType}{_sat}{_rtp|_rtn|_rtz|_rte}

where

 * {ConversionOpCodeName} = ConvertFToU|ConvertFToS|ConvertUToF|ConvertUToS|UConvert|SConvert|FConvert|SatConvertSToU|SatConvertUToS

SPIR-V Builtin Reinterpret / Bitcast Function Names
---------------------------------------------------

The unmangled names of SPIR-V builtin reinterpret / bitcast functions follow the convention:

.. code-block:: c

  __spirv_{BitcastOpCodeName}_R{ReturnType}

SPIR-V Builtin ImageSample Function Names
----------------------------------------

The unmangled names of SPIR-V builtin ImageSample functions follow the convention:

.. code-block:: c

  __spirv_{ImageSampleOpCodeName}_R{ReturnType}

SPIR-V Builtin GenericCastToPtr Function Name
----------------------------------------

The unmangled names of SPIR-V builtin GenericCastToPtrExplicit function follow the convention:

.. code-block:: c

  __spirv_GenericCastToPtrExplicit_To{Global|Local|Private}
  
SPIR-V 1.1 Builtin CreatePipeFromPipeStorage Function Name 
----------------------------------------

The unmangled names of SPIR-V builtin CreatePipeFromPipeStorage function follow the convention:

.. code-block:: c

  __spirv_CreatePipeFromPipeStorage_{read|write}

SPIR-V Extended Instructions Mapped to LLVM Function Calls
==========================================================

SPIR-V extended instructions are mapped to LLVM function calls. The function
name is IA64 mangled and the unmangled name has the format

.. code-block:: c

  __spirv_{ExtendedInstructionSetName}_{ExtendedInstrutionName}{__OptionalPostfixes}

where {ExtendedInstructionSetName} for OpenCL is "ocl".

The translated functions accepts all argument types accepted by the
corresponding SPIR-V instructions. The literal operands of extended
instruction are mapped to function call arguments with type i32.

The optional postfixes take the same format as SPIR-V builtin functions. The first postfix
starts with two underscores to facilitate identification since extended instruction name
may contain underscore. The remaining postfixes start with one underscore.

OpenCL Extended Builtin Vector Load Function Names
----------------------------------------

The unmangled names of OpenCL extended vector load functions follow the convention:

.. code-block:: c

  __spirv_ocl_{VectorLoadOpCodeName}__R{ReturnType}

where

 * {VectorLoadOpCodeName} = vloadn|vload_half|vload_halfn|vloada_halfn


SPIR-V Builtins Variables Mapped to LLVM Global Variables
=========================================================

SPIR-V builtin variables are mapped to LLVM global variables with unmangled
name __spirv_BuiltIn{Name}.

SPIR-V instructions mapped to LLVM metadata
===========================================

SPIR-V specification allows multiple module scope instructions, whereas LLVM
named metadata must be unique, so encoding of such instructions has the
following format:

.. code-block:: llvm

  !spirv.<OpCodeName> = !{!<InstructionMetadata1>, <InstructionMetadata2>, ..}
  !<InstructionMetadata1> = !{<Operand1>, <Operand2>, ..}
  !<InstructionMetadata2> = !{<Operand1>, <Operand2>, ..}

For example:

.. code-block:: llvm

  !spirv.Source = !{!0}
  !spirv.SourceExtension = !{!2, !3}
  !spirv.Extension = !{!2}
  !spirv.Capability = !{!4}
  !spirv.MemoryModel = !{!5}
  !spirv.EntryPoint = !{!6 ,!7}
  !spirv.ExecutionMode = !{!8, !9}
  !spirv.Generator = !{!10 }

  ; 3 - OpenCL_C, 102000 - OpenCL version 1.2, !1 - optional file id.
  !0 = !{i32 3, i32 102000, !1}
  !1 = !{!"/tmp/opencl/program.cl"}
  !2 = !{!"cl_khr_fp16"}
  !3 = !{!"cl_khr_gl_sharing"}
  !4 = !{i32 10}                ; Float64 - program uses doubles
  !5 = !{i32 1, i32 2}     ; 1 - 32-bit addressing model, 2 - OpenCL memory model
  !6 = !{i32 6, TBD, !"kernel1", TBD}
  !7 = !{i32 6, TBD, !"kernel2", TBD}
  !8 = !{!6, i32 18, i32 16, i32 1, i32 1}     ; local size hint <16, 1, 1> for 'kernel1'
  !9 = !{!7, i32 32}     ; independent forward progress is required for 'kernel2'
  !10 = !{i16 6, i16 123} ; 6 - Generator Id, 123 - Generator Version 

