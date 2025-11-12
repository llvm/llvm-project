//===- SPIRVGraphOps.td - Graph extended insts spec file -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of Graph extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_GRAPH_OPS
#define MLIR_DIALECT_SPIRV_IR_GRAPH_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// SPIR-V Graph opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all Graph ops.
class SPIRV_GraphARMOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_ArmVendorOp<mnemonic, traits> {

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_ARM_graph, SPV_ARM_tensors]>,
    Capability<[SPIRV_C_GraphARM]>
  ];
}

def SPIRV_GraphARMOp : SPIRV_GraphARMOp<"Graph", [
    AutomaticAllocationScope, DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface, InModuleScope, IsolatedFromAbove
  ]> {

  let summary = "Declare or define a SPIR-V graph";

  let description = [{
    This op declares or defines a SPIR-V graph using one region, which
    contains one or more blocks.

    This op is not allowed to implicitly capture global values, and all external
    references must use function arguments or symbol references. This op itself
    defines a symbol that is unique in the enclosing module op.

    Note that this op does not have a 1:1 mapping to the SPIR-V ops representing
    a graph. Indeed during serialization a single GraphARMOp is serialized into
    several different SPIR-V ops: OpGraphARM, OpGraphInputARM and OpGraphEndARM.
    There are as many occurences of OpGraphInputARM ops as many inputs in the
    graph. Deserialization maps that set of operations into a single GraphARMOp.

    This op itself takes no operands and generates no results. Its region
    can take zero or more arguments and return one or more values.

    ```
    spv-graph-arm-op ::= `spirv.ARM.Graph` function-signature
                        region
    ```

    #### Example:

    ```mlir
    spirv.ARM.Graph @graph(%arg0: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
        spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<GraphType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<BoolAttr>:$entry_point,
    StrAttr:$sym_name
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let hasVerifier = 0;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "GraphType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,  CArg<"bool", "false">:$entry_point)>];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
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

// Check that an op can only be used within the scope of a spirv.ARM.Graph op.
def InGraphScope : PredOpTrait<
  "op must appear in a spirv.ARM.Graph op's block",
  CPred<"isNestedInGraphARMOpInterface($_op.getParentOp())">>;

// -----

def SPIRV_GraphConstantARMOp : SPIRV_GraphARMOp<"GraphConstant", [InGraphScope, Pure, ConstantLike]> {
  let summary = "Declare a graph constant.";

  let description = [{
    Declare a graph constant.
    Result Type must be an OpTypeTensorARM.
    GraphConstantID must be a 32-bit integer literal.

    #### Example:

    ```mlir
    %0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
    ```

    GraphConstantID is a unique identifier which is use to map the contants
    defined by GraphConstantARM in the SPIRV module with the one provided at
    shader creation time via the VkDataGraphPipelineShaderModuleCreateInfoARM.
    That Vulkan structure provides a list of VkDataGraphPipelineConstantARM
    which contains the bindings from id to data. (For more details see
    https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#graphs)
  }];

  let arguments = (ins
    I32Attr: $graph_constant_id
  );

  let results = (outs
    SPIRV_AnyTensorArm:$output
  );

  let hasVerifier = 0;

  let autogenSerialization = 0;

  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
}

// -----

def SPIRV_GraphEntryPointARMOp : SPIRV_GraphARMOp<"GraphEntryPoint", [InModuleScope]> {
  let summary = [{
    Declare a graph entry point and its interface.
  }];

  let description = [{
    Graph Entry Point must be the Result <id> of an OpGraphARM instruction.

    Name is a name string for the graphentry point. A module cannot have two
    OpGraphEntryPointARM instructions with the same Name string.

    Interface is a list of symbol references to `spirv.GlobalVariable`
    operations. These declare the set of global variables from a
    module that form the interface of this entry point. The set of
    Interface symbols must be equal to or a superset of the
    `spirv.GlobalVariable`s referenced by the entry point’s static call
    tree, within the interface’s storage classes.

    #### Example:

    ```mlir
    spirv.GlobalVariable @arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
    spirv.GlobalVariable @res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
    spirv.ARM.GraphEntryPoint @graph, @arg_0, @res_0
    spirv.ARM.Graph @graph(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
        ...
    }
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    SymbolRefArrayAttr:$interface
  );

  let results = (outs);

  // Checks for graph and interface symbol reference are done in spirv::ModuleOp verification.
  let hasVerifier = 0;

  let autogenSerialization = 0;

  let builders = [
    OpBuilder<(ins "spirv::GraphARMOp":$graph, "ArrayRef<Attribute>":$interfaceVars)>];
}

// -----

def SPIRV_GraphOutputsARMOp : SPIRV_GraphARMOp<"GraphOutputs", [InGraphScope, Pure,
                                               Terminator]> {

  let summary = "Define graph outputs.";

  let description = [{
    Values are the graph outputs values and must match the GraphOutputs Type
    operand of the OpTypeGraphARM type of the OpGraphARM body this
    instruction is in.

    This instruction must be the last instruction in a block.

    #### Example:

    ```mlir
    spirv.ARM.Graph @graph(%arg0 : !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
        spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
    }
    ```
  }];

  let arguments = (ins
    Variadic<SPIRV_AnyTensorArm>:$value
  );

  let results = (outs);

  let autogenSerialization = 0;

  let hasOpcode = 0;

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // MLIR_DIALECT_SPIRV_IR_GRAPH_OPS
