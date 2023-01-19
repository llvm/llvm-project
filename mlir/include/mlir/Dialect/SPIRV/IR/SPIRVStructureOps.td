//===-- SPIRVStructureOps.td - MLIR SPIR-V Structure Ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains ops for defining the SPIR-V structure: module, function,
// and module-level operations. The representational form of these ops deviate
// from the SPIR-V binary format in order to utilize MLIR mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_STRUCTURE_OPS
#define MLIR_DIALECT_SPIRV_IR_STRUCTURE_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPIRV_AddressOfOp : SPIRV_Op<"mlir.addressof",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     InFunctionScope, Pure]> {
  let summary = "Get the address of a global variable.";

  let description = [{
    Variables in module scope are defined using symbol names. This op generates
    an SSA value that can be used to refer to the symbol within function scope
    for use in ops that expect an SSA value. This operation has no corresponding
    SPIR-V instruction; it's merely used for modelling purpose in the SPIR-V
    dialect. Since variables in module scope in SPIR-V dialect are of pointer
    type, this op returns a pointer type as well, and the type is the same as
    the variable referenced.

    <!-- End of AutoGen section -->

    ```
    spv-address-of-op ::= ssa-id `=` `spirv.mlir.addressof` symbol-ref-id
                                     `:` spirv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.mlir.addressof @global_var : !spirv.ptr<f32, Input>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$variable
  );

  let results = (outs
    SPIRV_AnyPtr:$pointer
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let builders = [OpBuilder<(ins "spirv::GlobalVariableOp":$var)>];

  let assemblyFormat = "$variable attr-dict `:` type($pointer)";
}

// -----

def SPIRV_ConstantOp : SPIRV_Op<"Constant",
    [ConstantLike,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     Pure]> {
  let summary = [{
    Declare a new integer-type or floating-point-type scalar constant.
  }];

  let description = [{
    This op declares a SPIR-V normal constant. SPIR-V has multiple constant
    instructions covering different constant types:

    * `OpConstantTrue` and `OpConstantFalse` for boolean constants
    * `OpConstant` for scalar constants
    * `OpConstantComposite` for composite constants
    * `OpConstantNull` for null constants
    * ...

    Having such a plethora of constant instructions renders IR transformations
    more tedious. Therefore, we use a single `spirv.Constant` op to represent
    them all. Note that conversion between those SPIR-V constant instructions
    and this op is purely mechanical; so it can be scoped to the binary
    (de)serialization process.

    <!-- End of AutoGen section -->

    ```
    spirv.Constant-op ::= ssa-id `=` `spirv.Constant` attribute-value
                        (`:` spirv-type)?
    ```

    #### Example:

    ```mlir
    %0 = spirv.Constant true
    %1 = spirv.Constant dense<[2, 3]> : vector<2xf32>
    %2 = spirv.Constant [dense<3.0> : vector<2xf32>] : !spirv.array<1xvector<2xf32>>
    ```

    TODO: support constant structs
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    SPIRV_Type:$constant
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns true if a constant can be built for the given `type`.
    static bool isBuildableWith(Type type);

    // Creates a constant zero/one of the given `type` at the current insertion
    // point of `builder` and returns it.
    static spirv::ConstantOp getZero(Type type, Location loc,
                                     OpBuilder &builder);
    static spirv::ConstantOp getOne(Type type, Location loc,
                                    OpBuilder &builder);
  }];

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPIRV_EntryPointOp : SPIRV_Op<"EntryPoint", [InModuleScope]> {
  let summary = [{
    Declare an entry point, its execution model, and its interface.
  }];

  let description = [{
    Execution Model is the execution model for the entry point and its
    static call tree. See Execution Model.

    Entry Point must be the Result <id> of an OpFunction instruction.

    Name is a name string for the entry point. A module cannot have two
    OpEntryPoint instructions with the same Execution Model and the same
    Name string.

    Interface is a list of symbol references to `spirv.GlobalVariable`
    operations. These declare the set of global variables from a
    module that form the interface of this entry point. The set of
    Interface symbols must be equal to or a superset of the
    `spirv.GlobalVariable`s referenced by the entry point’s static call
    tree, within the interface’s storage classes.  Before version 1.4,
    the interface’s storage classes are limited to the Input and
    Output storage classes. Starting with version 1.4, the interface’s
    storage classes are all storage classes used in declaring all
    global variables referenced by the entry point’s call tree.

    <!-- End of AutoGen section -->

    ```
    execution-model ::= "Vertex" | "TesellationControl" |
                        <and other SPIR-V execution models...>

    entry-point-op ::= ssa-id `=` `spirv.EntryPoint` execution-model
                       symbol-reference (`, ` symbol-reference)*
    ```

    #### Example:

    ```mlir
    spirv.EntryPoint "GLCompute" @foo
    spirv.EntryPoint "Kernel" @foo, @var1, @var2

    ```
  }];

  let arguments = (ins
    SPIRV_ExecutionModelAttr:$execution_model,
    FlatSymbolRefAttr:$fn,
    SymbolRefArrayAttr:$interface
  );

  let results = (outs);

  let autogenSerialization = 0;

  let builders = [
    OpBuilder<(ins "spirv::ExecutionModel":$executionModel,
      "spirv::FuncOp":$function, "ArrayRef<Attribute>":$interfaceVars)>];
}

// -----

def SPIRV_ExecutionModeOp : SPIRV_Op<"ExecutionMode", [InModuleScope]> {
  let summary = "Declare an execution mode for an entry point.";

  let description = [{
    Entry Point must be the Entry Point <id> operand of an OpEntryPoint
    instruction.

    Mode is the execution mode. See Execution Mode.

    This instruction is only valid when the Mode operand is an execution
    mode that takes no Extra Operands, or takes Extra Operands that are not
    <id> operands.

    <!-- End of AutoGen section -->

    ```
    execution-mode ::= "Invocations" | "SpacingEqual" |
                       <and other SPIR-V execution modes...>

    execution-mode-op ::= `spirv.ExecutionMode ` ssa-use execution-mode
                          (integer-literal (`, ` integer-literal)* )?
    ```

    #### Example:

    ```mlir
    spirv.ExecutionMode @foo "ContractionOff"
    spirv.ExecutionMode @bar "LocalSizeHint", 3, 4, 5
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    SPIRV_ExecutionModeAttr:$execution_mode,
    I32ArrayAttr:$values
  );

  let results = (outs);

  let hasVerifier = 0;

  let autogenSerialization = 0;

  let builders = [
    OpBuilder<(ins "spirv::FuncOp":$function,
      "spirv::ExecutionMode":$executionMode, "ArrayRef<int32_t>":$params)>];
}

// -----

def SPIRV_FuncOp : SPIRV_Op<"func", [
    AutomaticAllocationScope, DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface, InModuleScope, IsolatedFromAbove
  ]> {
  let summary = "Declare or define a function";

  let description = [{
    This op declares or defines a SPIR-V function using one region, which
    contains one or more blocks.

    Different from the SPIR-V binary format, this op is not allowed to
    implicitly capture global values, and all external references must use
    function arguments or symbol references. This op itself defines a symbol
    that is unique in the enclosing module op.

    This op itself takes no operands and generates no results. Its region
    can take zero or more arguments and return zero or one values.

    <!-- End of AutoGen section -->

    ```
    spv-function-control ::= "None" | "Inline" | "DontInline" | ...
    spv-function-op ::= `spirv.func` function-signature
                         spv-function-control region
    ```

    #### Example:

    ```mlir
    spirv.func @foo() -> () "None" { ... }
    spirv.func @bar() -> () "Inline|Pure" { ... }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    StrAttr:$sym_name,
    SPIRV_FunctionControlAttr:$function_control
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let hasVerifier = 0;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"spirv::FunctionControl", "spirv::FunctionControl::None">:$control,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumArguments, and getNumResults can be called safely
    LogicalResult verifyType();

    /// Hook for FunctionOpInterface, called after verifying the function
    /// type and the presence of the (potentially empty) function body.
    /// Ensures SPIR-V specific semantics.
    LogicalResult verifyBody();
  }];
}

// -----

def SPIRV_GlobalVariableOp : SPIRV_Op<"GlobalVariable", [InModuleScope, Symbol]> {
  let summary = [{
    Allocate an object in memory at module scope. The object is
    referenced using a symbol name.
  }];

  let description = [{
    The variable type must be an OpTypePointer. Its type operand is the type of
    object in memory.

    Storage Class is the Storage Class of the memory holding the object. It
    cannot be Generic. It must be the same as the Storage Class operand of
    the variable types. Only those storage classes that are valid at module
    scope (like Input, Output, StorageBuffer, etc.) are valid.

    Initializer is optional.  If Initializer is present, it will be
    the initial value of the variable’s memory content. Initializer
    must be an symbol defined from a constant instruction or other
    `spirv.GlobalVariable` operation in module scope. Initializer must
    have the same type as the type of the defined symbol.

    <!-- End of AutoGen section -->

    ```
    variable-op ::= `spirv.GlobalVariable` spirv-type symbol-ref-id
                    (`initializer(` symbol-ref-id `)`)?
                    (`bind(` integer-literal, integer-literal `)`)?
                    (`built_in(` string-literal `)`)?
                    attribute-dict?
    ```

    where `initializer` specifies initializer and `bind` specifies the
    descriptor set and binding number. `built_in` specifies SPIR-V
    BuiltIn decoration associated with the op.

    #### Example:

    ```mlir
    spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input> @var0
    spirv.GlobalVariable @var1 initializer(@var0) : !spirv.ptr<f32, Output>
    spirv.GlobalVariable @var2 bind(1, 2) : !spirv.ptr<f32, Uniform>
    spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    ```
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    OptionalAttr<FlatSymbolRefAttr>:$initializer,
    OptionalAttr<I32Attr>:$location,
    OptionalAttr<I32Attr>:$binding,
    OptionalAttr<I32Attr>:$descriptor_set,
    OptionalAttr<StrAttr>:$builtin
  );

  let results = (outs);

  let builders = [
    OpBuilder<(ins "TypeAttr":$type,
                   "StringAttr":$sym_name,
                   CArg<"FlatSymbolRefAttr", "nullptr">:$initializer),
    [{
      $_state.addAttribute("type", type);
      $_state.addAttribute(getSymNameAttrName($_state.name), sym_name);
      if (initializer)
        $_state.addAttribute(getInitializerAttrName($_state.name), initializer);
    }]>,
    OpBuilder<(ins "TypeAttr":$type, "ArrayRef<NamedAttribute>":$namedAttrs),
    [{
      $_state.addAttribute("type", type);
      $_state.addAttributes(namedAttrs);
    }]>,
    OpBuilder<(ins "Type":$type, "StringRef":$name,
      "unsigned":$descriptorSet, "unsigned":$binding)>,
    OpBuilder<(ins "Type":$type, "StringRef":$name,
      "spirv::BuiltIn":$builtin)>,
    OpBuilder<(ins "Type":$type,
                   "StringRef":$sym_name,
                    CArg<"FlatSymbolRefAttr", "{}">:$initializer),
    [{
      $_state.addAttribute("type", TypeAttr::get(type));
      $_state.addAttribute(getSymNameAttrName($_state.name), $_builder.getStringAttr(sym_name));
      if (initializer)
        $_state.addAttribute(getInitializerAttrName($_state.name), initializer);
    }]>
  ];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    ::mlir::spirv::StorageClass storageClass() {
      return this->getType().cast<::mlir::spirv::PointerType>().getStorageClass();
    }
  }];
}

// -----

def SPIRV_ModuleOp : SPIRV_Op<"module",
    [IsolatedFromAbove, NoRegionArguments, NoTerminator,
     SingleBlock, SymbolTable, Symbol]> {
  let summary = "The top-level op that defines a SPIR-V module";

  let description = [{
    This op defines a SPIR-V module using a MLIR region. The region contains
    one block. Module-level operations, including functions definitions,
    are all placed in this block.

    Using an op with a region to define a SPIR-V module enables "embedding"
    SPIR-V modules in other dialects in a clean manner: this op guarantees
    the validity and serializability of a SPIR-V module and thus serves as
    a clear-cut boundary.

    This op takes no operands and generates no results. This op should not
    implicitly capture values from the enclosing environment.

    This op has only one region, which only contains one block. The block
    has no terminator.

    <!-- End of AutoGen section -->

    ```
    addressing-model ::= `Logical` | `Physical32` | `Physical64` | ...
    memory-model ::= `Simple` | `GLSL450` | `OpenCL` | `Vulkan` | ...
    spv-module-op ::= `spirv.module` addressing-model memory-model
                      (requires  spirv-vce-attribute)?
                      (`attributes` attribute-dict)?
                      region
    ```

    #### Example:

    ```mlir
    spirv.module Logical GLSL450  {}

    spirv.module Logical Vulkan
        requires #spirv.vce<v1.0, [Shader], [SPV_KHR_vulkan_memory_model]>
        attributes { some_additional_attr = ... } {
      spirv.func @do_nothing() -> () {
        spirv.Return
      }
    }
    ```
  }];

  let arguments = (ins
    SPIRV_AddressingModelAttr:$addressing_model,
    SPIRV_MemoryModelAttr:$memory_model,
    OptionalAttr<SPIRV_VerCapExtAttr>:$vce_triple,
    OptionalAttr<StrAttr>:$sym_name
  );

  let results = (outs);

  let regions = (region AnyRegion);

  let builders = [
    OpBuilder<(ins CArg<"std::optional<StringRef>", "std::nullopt">:$name)>,
    OpBuilder<(ins "spirv::AddressingModel":$addressing_model,
                   "spirv::MemoryModel":$memory_model,
                   CArg<"std::optional<spirv::VerCapExtAttr>", "std::nullopt">:$vce_triple,
                   CArg<"std::optional<StringRef>", "std::nullopt">:$name)>
  ];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{

    bool isOptionalSymbol() { return true; }

    std::optional<StringRef> getName() { return getSymName(); }

    static StringRef getVCETripleAttrName() { return "vce_triple"; }
  }];

  let hasVerifier = 0;
  let hasRegionVerifier = 1;
}

// -----

def SPIRV_ReferenceOfOp : SPIRV_Op<"mlir.referenceof", [Pure]> {
  let summary = "Reference a specialization constant.";

  let description = [{
    Specialization constants in module scope are defined using symbol names.
    This op generates an SSA value that can be used to refer to the symbol
    within function scope for use in ops that expect an SSA value.
    This operation has no corresponding SPIR-V instruction; it's merely used
    for modelling purpose in the SPIR-V dialect. This op's return type is
    the same as the specialization constant.

    <!-- End of AutoGen section -->

    ```
    spv-reference-of-op ::= ssa-id `=` `spirv.mlir.referenceof` symbol-ref-id
                                       `:` spirv-scalar-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.mlir.referenceof @spec_const : f32
    ```

    TODO Add support for composite specialization constants.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$spec_const
  );

  let results = (outs
    SPIRV_Type:$reference
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let assemblyFormat = "$spec_const attr-dict `:` type($reference)";
}

// -----

def SPIRV_SpecConstantOp : SPIRV_Op<"SpecConstant", [InModuleScope, Symbol]> {
  let summary = [{
    Declare a new integer-type or floating-point-type scalar specialization
    constant.
  }];

  let description = [{
    This op declares a SPIR-V scalar specialization constant. SPIR-V has
    multiple constant instructions covering different scalar types:

    * `OpSpecConstantTrue` and `OpSpecConstantFalse` for boolean constants
    * `OpSpecConstant` for scalar constants

    Similar as `spirv.Constant`, this op represents all of the above cases.
    `OpSpecConstantComposite` and `OpSpecConstantOp` are modelled with
    separate ops.

    <!-- End of AutoGen section -->

    ```
    spv-spec-constant-op ::= `spirv.SpecConstant` symbol-ref-id
                             `spec_id(` integer `)`
                             `=` attribute-value (`:` spirv-type)?
    ```

    where `spec_id` specifies the SPIR-V SpecId decoration associated with
    the op.

    #### Example:

    ```mlir
    spirv.SpecConstant @spec_const1 = true
    spirv.SpecConstant @spec_const2 spec_id(5) = 42 : i32
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypedAttrInterface:$default_value
  );

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPIRV_SpecConstantCompositeOp : SPIRV_Op<"SpecConstantComposite", [
    InModuleScope, Symbol]> {
  let summary = "Declare a new composite specialization constant.";

  let description = [{
    This op declares a SPIR-V composite specialization constant. This covers
    the `OpSpecConstantComposite` SPIR-V instruction. Scalar constants are
    covered by `spirv.SpecConstant`.

    A constituent of a spec constant composite can be:
    - A symbol referring of another spec constant.
    - The SSA ID of a non-specialization constant (i.e. defined through
      `spirv.SpecConstant`).
    - The SSA ID of a `spirv.Undef`.

    ```
    spv-spec-constant-composite-op ::= `spirv.SpecConstantComposite` symbol-ref-id ` (`
                                       symbol-ref-id (`, ` symbol-ref-id)*
                                       `) :` composite-type
    ```

     where `composite-type` is some non-scalar type that can be represented in the `spv`
     dialect: `spirv.struct`, `spirv.array`, or `vector`.

     #### Example:

     ```mlir
     spirv.SpecConstant @sc1 = 1   : i32
     spirv.SpecConstant @sc2 = 2.5 : f32
     spirv.SpecConstant @sc3 = 3.5 : f32
     spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<i32, f32, f32>
     ```

    TODO Add support for constituents that are:
    - regular constants.
    - undef.
    - spec constant composite.
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    SymbolRefArrayAttr:$constituents
  );

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPIRV_SpecConstantOperationOp : SPIRV_Op<"SpecConstantOperation", [
       Pure, InFunctionScope,
       SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = [{
    Declare a new specialization constant that results from doing an operation.
  }];

  let description = [{
    This op declares a SPIR-V specialization constant that results from
    doing an operation on other constants (specialization or otherwise).

    In the `spv` dialect, this op is modelled as follows:

    ```
    spv-spec-constant-operation-op ::= `spirv.SpecConstantOperation` `wraps`
                                         generic-spirv-op `:` function-type
    ```

    In particular, an `spirv.SpecConstantOperation` contains exactly one
    region. In turn, that region, contains exactly 2 instructions:
    - One of SPIR-V's instructions that are allowed within an
    OpSpecConstantOp.
    - An `spirv.mlir.yield` instruction as the terminator.

    The following SPIR-V instructions are valid:
    - OpSConvert,
    - OpUConvert,
    - OpFConvert,
    - OpSNegate,
    - OpNot,
    - OpIAdd,
    - OpISub,
    - OpIMul,
    - OpUDiv,
    - OpSDiv,
    - OpUMod,
    - OpSRem,
    - OpSMod
    - OpShiftRightLogical,
    - OpShiftRightArithmetic,
    - OpShiftLeftLogical
    - OpBitwiseOr,
    - OpBitwiseXor,
    - OpBitwiseAnd
    - OpVectorShuffle,
    - OpCompositeExtract,
    - OpCompositeInsert
    - OpLogicalOr,
    - OpLogicalAnd,
    - OpLogicalNot,
    - OpLogicalEqual,
    - OpLogicalNotEqual
    - OpSelect
    - OpIEqual,
    - OpINotEqual
    - OpULessThan,
    - OpSLessThan
    - OpUGreaterThan,
    - OpSGreaterThan
    - OpULessThanEqual,
    - OpSLessThanEqual
    - OpUGreaterThanEqual,
    - OpSGreaterThanEqual

    TODO Add capability-specific ops when supported.

    #### Example:
    ```mlir
    %0 = spirv.Constant 1: i32
    %1 = spirv.Constant 1: i32

    %2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%0, %1) : (i32, i32) -> i32
    ```
  }];

  let arguments = (ins);

  let results = (outs AnyType:$result);

  let regions = (region SizedRegion<1>:$body);

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let hasVerifier = 0;
  let hasRegionVerifier = 1;
}

// -----

def SPIRV_YieldOp : SPIRV_Op<"mlir.yield", [
    HasParent<"SpecConstantOperationOp">, Pure, Terminator]> {
  let summary = [{
    Yields the result computed in `spirv.SpecConstantOperation`'s
    region back to the parent op.
  }];

  let description = [{
    This op is a special terminator whose only purpose is to terminate
    an `spirv.SpecConstantOperation`'s enclosed region. It accepts a
    single operand produced by the preceeding (and only other) instruction
    in its parent block (see SPIRV_SpecConstantOperation for further
    details). This op has no corresponding SPIR-V instruction.

    ```
    spirv.mlir.yield ::= `spirv.mlir.yield` ssa-id : spirv-type
    ```

    #### Example:
    ```mlir
    %0 = ... (some op supported by SPIR-V OpSpecConstantOp)
    spirv.mlir.yield %0
    ```
  }];

  let arguments = (ins AnyType:$operand);

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let assemblyFormat = "attr-dict $operand `:` type($operand)";

  let hasVerifier = 0;
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_STRUCTURE_OPS
