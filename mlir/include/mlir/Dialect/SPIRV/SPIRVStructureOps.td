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

#ifndef SPIRV_STRUCTURE_OPS
#define SPIRV_STRUCTURE_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPV_AddressOfOp : SPV_Op<"mlir.addressof", [InFunctionScope, NoSideEffect]> {
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
    spv-address-of-op ::= ssa-id `=` `spv.mlir.addressof` symbol-ref-id
                                     `:` spirv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.mlir.addressof @global_var : !spv.ptr<f32, Input>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$variable
  );

  let results = (outs
    SPV_AnyPtr:$pointer
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let builders = [OpBuilderDAG<(ins "spirv::GlobalVariableOp":$var)>];

  let assemblyFormat = "$variable attr-dict `:` type($pointer)";
}

// -----

def SPV_ConstantOp : SPV_Op<"constant", [ConstantLike, NoSideEffect]> {
  let summary = "The op that declares a SPIR-V normal constant";

  let description = [{
    This op declares a SPIR-V normal constant. SPIR-V has multiple constant
    instructions covering different constant types:

    * `OpConstantTrue` and `OpConstantFalse` for boolean constants
    * `OpConstant` for scalar constants
    * `OpConstantComposite` for composite constants
    * `OpConstantNull` for null constants
    * ...

    Having such a plethora of constant instructions renders IR transformations
    more tedious. Therefore, we use a single `spv.constant` op to represent
    them all. Note that conversion between those SPIR-V constant instructions
    and this op is purely mechanical; so it can be scoped to the binary
    (de)serialization process.

    <!-- End of AutoGen section -->

    ```
    spv-constant-op ::= ssa-id `=` `spv.constant` attribute-value
                        (`:` spirv-type)?
    ```

    #### Example:

    ```mlir
    %0 = spv.constant true
    %1 = spv.constant dense<[2, 3]> : vector<2xf32>
    %2 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>
    ```

    TODO: support constant structs
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    SPV_Type:$constant
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

def SPV_EntryPointOp : SPV_Op<"EntryPoint", [InModuleScope]> {
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

    Interface is a list of symbol references to `spv.globalVariable`
    operations. These declare the set of global variables from a
    module that form the interface of this entry point. The set of
    Interface symbols must be equal to or a superset of the
    `spv.globalVariable`s referenced by the entry point’s static call
    tree, within the interface’s storage classes.  Before version 1.4,
    the interface’s storage classes are limited to the Input and
    Output storage classes. Starting with version 1.4, the interface’s
    storage classes are all storage classes used in declaring all
    global variables referenced by the entry point’s call tree.

    <!-- End of AutoGen section -->

    ```
    execution-model ::= "Vertex" | "TesellationControl" |
                        <and other SPIR-V execution models...>

    entry-point-op ::= ssa-id `=` `spv.EntryPoint` execution-model
                       symbol-reference (`, ` symbol-reference)*
    ```

    #### Example:

    ```mlir
    spv.EntryPoint "GLCompute" @foo
    spv.EntryPoint "Kernel" @foo, @var1, @var2

    ```
  }];

  let arguments = (ins
    SPV_ExecutionModelAttr:$execution_model,
    FlatSymbolRefAttr:$fn,
    SymbolRefArrayAttr:$interface
  );

  let results = (outs);

  let autogenSerialization = 0;

  let builders = [
    OpBuilderDAG<(ins "spirv::ExecutionModel":$executionModel,
      "spirv::FuncOp":$function, "ArrayRef<Attribute>":$interfaceVars)>];
}

// -----

def SPV_FuncOp : SPV_Op<"func", [
    AutomaticAllocationScope, DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionLike, InModuleScope, IsolatedFromAbove, Symbol
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
    spv-function-op ::= `spv.func` function-signature
                         spv-function-control region
    ```

    #### Example:

    ```mlir
    spv.func @foo() -> () "None" { ... }
    spv.func @bar() -> () "Inline|Pure" { ... }
    ```
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    SPV_FunctionControlAttr:$function_control
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let verifier = [{ return success(); }];

  let builders = [
    OpBuilderDAG<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"spirv::FunctionControl", "spirv::FunctionControl::None">:$control,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FuncOp>;

    /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }

    /// Returns the number of results. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called safely
    LogicalResult verifyType();

    /// Hook for OpTrait::FunctionLike, called after verifying the function
    /// type and the presence of the (potentially empty) function body.
    /// Ensures SPIR-V specific semantics.
    LogicalResult verifyBody();
  }];
}

// -----

def SPV_GlobalVariableOp : SPV_Op<"globalVariable", [InModuleScope, Symbol]> {
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
    `spv.globalVariable` operation in module scope. Initializer must
    have the same type as the type of the defined symbol.

    <!-- End of AutoGen section -->

    ```
    variable-op ::= `spv.globalVariable` spirv-type symbol-ref-id
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
    spv.globalVariable @var0 : !spv.ptr<f32, Input> @var0
    spv.globalVariable @var1 initializer(@var0) : !spv.ptr<f32, Output>
    spv.globalVariable @var2 bind(1, 2) : !spv.ptr<f32, Uniform>
    spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    ```
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    OptionalAttr<FlatSymbolRefAttr>:$initializer
  );

  let results = (outs);

  let builders = [
    OpBuilderDAG<(ins "TypeAttr":$type, "ArrayRef<NamedAttribute>":$namedAttrs),
    [{
      $_state.addAttribute("type", type);
      $_state.addAttributes(namedAttrs);
    }]>,
    OpBuilderDAG<(ins "Type":$type, "StringRef":$name,
      "unsigned":$descriptorSet, "unsigned":$binding)>,
    OpBuilderDAG<(ins "Type":$type, "StringRef":$name,
      "spirv::BuiltIn":$builtin)>
  ];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    ::mlir::spirv::StorageClass storageClass() {
      return this->type().cast<::mlir::spirv::PointerType>().getStorageClass();
    }
  }];
}

// -----

def SPV_ModuleOp : SPV_Op<"module",
                          [IsolatedFromAbove,
                           SingleBlockImplicitTerminator<"ModuleEndOp">,
                           SymbolTable, Symbol]> {
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
    must be terminated via the `spv.mlir.endmodule` op.

    <!-- End of AutoGen section -->

    ```
    addressing-model ::= `Logical` | `Physical32` | `Physical64` | ...
    memory-model ::= `Simple` | `GLSL450` | `OpenCL` | `Vulkan` | ...
    spv-module-op ::= `spv.module` addressing-model memory-model
                      (requires  spirv-vce-attribute)?
                      (`attributes` attribute-dict)?
                      region
    ```

    #### Example:

    ```mlir
    spv.module Logical GLSL450  {}

    spv.module Logical Vulkan
        requires #spv.vce<v1.0, [Shader], [SPV_KHR_vulkan_memory_model]>
        attributes { some_additional_attr = ... } {
      spv.func @do_nothing() -> () {
        spv.Return
      }
    }
    ```
  }];

  let arguments = (ins
    SPV_AddressingModelAttr:$addressing_model,
    SPV_MemoryModelAttr:$memory_model,
    OptionalAttr<SPV_VerCapExtAttr>:$vce_triple,
    OptionalAttr<StrAttr>:$sym_name
  );

  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilderDAG<(ins CArg<"Optional<StringRef>", "llvm::None">:$name)>,
    OpBuilderDAG<(ins "spirv::AddressingModel":$addressing_model,
      "spirv::MemoryModel":$memory_model,
      CArg<"Optional<StringRef>", "llvm::None">:$name)>
  ];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{

    bool isOptionalSymbol() { return true; }

    Optional<StringRef> getName() { return sym_name(); }

    static StringRef getVCETripleAttrName() { return "vce_triple"; }

    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
  }];
}

// -----

def SPV_ModuleEndOp : SPV_Op<"mlir.endmodule", [InModuleScope, Terminator]> {
  let summary = "The pseudo op that ends a SPIR-V module";

  let description = [{
    This op terminates the only block inside a `spv.module`'s only region.
    This op does not have a corresponding SPIR-V instruction and thus will
    not be serialized into the binary format; it is used solely to satisfy
    the structual requirement that an block must be ended with a terminator.
  }];

  let arguments = (ins);

  let results = (outs);

  let assemblyFormat = "attr-dict";

  let verifier = [{ return success(); }];

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPV_ReferenceOfOp : SPV_Op<"mlir.referenceof", [NoSideEffect]> {
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
    spv-reference-of-op ::= ssa-id `=` `spv.mlir.referenceof` symbol-ref-id
                                       `:` spirv-scalar-type
    ```

    #### Example:

    ```mlir
    %0 = spv.mlir.referenceof @spec_const : f32
    ```

    TODO Add support for composite specialization constants.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$spec_const
  );

  let results = (outs
    SPV_Type:$reference
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let assemblyFormat = "$spec_const attr-dict `:` type($reference)";
}

// -----

def SPV_SpecConstantOp : SPV_Op<"specConstant", [InModuleScope, Symbol]> {
  let summary = "The op that declares a SPIR-V specialization constant";

  let description = [{
    This op declares a SPIR-V scalar specialization constant. SPIR-V has
    multiple constant instructions covering different scalar types:

    * `OpSpecConstantTrue` and `OpSpecConstantFalse` for boolean constants
    * `OpSpecConstant` for scalar constants

    Similar as `spv.constant`, this op represents all of the above cases.
    `OpSpecConstantComposite` and `OpSpecConstantOp` are modelled with
    separate ops.

    <!-- End of AutoGen section -->

    ```
    spv-spec-constant-op ::= `spv.specConstant` symbol-ref-id
                             `spec_id(` integer `)`
                             `=` attribute-value (`:` spirv-type)?
    ```

    where `spec_id` specifies the SPIR-V SpecId decoration associated with
    the op.

    #### Example:

    ```mlir
    spv.specConstant @spec_const1 = true
    spv.specConstant @spec_const2 spec_id(5) = 42 : i32
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    AnyAttr:$default_value
  );

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

def SPV_SpecConstantCompositeOp : SPV_Op<"specConstantComposite", [InModuleScope, Symbol]> {
  let summary = "Declare a new composite specialization constant.";

  let description = [{
    This op declares a SPIR-V composite specialization constant. This covers
    the `OpSpecConstantComposite` SPIR-V instruction. Scalar constants are
    covered by `spv.specConstant`.

    A constituent of a spec constant composite can be:
    - A symbol referring of another spec constant.
    - The SSA ID of a non-specialization constant (i.e. defined through
      `spv.specConstant`).
    - The SSA ID of a `spv.undef`.

    ```
    spv-spec-constant-composite-op ::= `spv.specConstantComposite` symbol-ref-id ` (`
                                       symbol-ref-id (`, ` symbol-ref-id)*
                                       `) :` composite-type
    ```

     where `composite-type` is some non-scalar type that can be represented in the `spv`
     dialect: `spv.struct`, `spv.array`, or `vector`.

     #### Example:

     ```mlir
     spv.specConstant @sc1 = 1   : i32
     spv.specConstant @sc2 = 2.5 : f32
     spv.specConstant @sc3 = 3.5 : f32
     spv.specConstantComposite @scc (@sc1, @sc2, @sc3) : !spv.struct<i32, f32, f32>
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

#endif // SPIRV_STRUCTURE_OPS
