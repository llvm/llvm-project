//===- TargetAndABI.td - SPIR-V Target and ABI definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for supporting lowering to SPIR-V dialect. This file
// defines SPIR-V attributes used for specifying the shader interface or ABI.
// This is because SPIR-V module is expected to work in an execution environment
// as specified by a client API. A SPIR-V module needs to "link" correctly with
// the execution environment regarding the resources that are used in the SPIR-V
// module and get populated with data via the client API. The shader interface
// (or ABI) is passed into SPIR-V lowering path via attributes defined in this
// file. A compilation flow targeting SPIR-V is expected to attach such
// attributes to resources and other suitable places.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_TARGET_AND_ABI
#define MLIR_DIALECT_SPIRV_IR_TARGET_AND_ABI

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

class SPV_Attr<string attrName, string attrMnemonic>
    : AttrDef<SPIRV_Dialect, attrName> {
  let mnemonic = attrMnemonic;
}

// For entry functions, this attribute specifies information related to entry
// points in the generated SPIR-V module:
// 1) WorkGroup Size.
def SPV_EntryPointABIAttr : SPV_Attr<"EntryPointABI", "entry_point_abi"> {
  let parameters = (ins OptionalParameter<"DenseIntElementsAttr">:$local_size);
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPV_ExtensionArrayAttr : TypedArrayAttrBase<
    SPV_ExtensionAttr, "SPIR-V extension array attribute">;

def SPV_CapabilityArrayAttr : TypedArrayAttrBase<
    SPV_CapabilityAttr, "SPIR-V capability array attribute">;

// Description of cooperative matrix operations supported on the
// target. Represents `VkCooperativeMatrixPropertiesNV`. See
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkCooperativeMatrixPropertiesNV.html
def SPV_CooperativeMatrixPropertiesNVAttr :
    SPV_Attr<"CooperativeMatrixPropertiesNV", "coop_matrix_props"> {
  let parameters = (ins
    "int":$m_size,
    "int":$n_size,
    "int":$k_size,
    "mlir::Type":$a_type,
    "mlir::Type":$b_type,
    "mlir::Type":$c_type,
    "mlir::Type":$result_type,
    "mlir::spirv::ScopeAttr":$scope
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPV_CooperativeMatrixPropertiesNVArrayAttr :
    TypedArrayAttrBase<SPV_CooperativeMatrixPropertiesNVAttr,
                       "CooperativeMatrixPropertiesNV array attribute">;

// Description of the supported joint matrix operations. See
// https://github.com/intel/llvm/blob/sycl/sycl/doc/design/spirv-extensions/SPV_INTEL_joint_matrix.asciidoc
def SPV_JointMatrixPropertiesINTELAttr :
    SPV_Attr<"JointMatrixPropertiesINTEL", "joint_matrix_props"> {
  let parameters = (ins
    "int":$m_size,
    "int":$n_size,
    "int":$k_size,
    "mlir::Type":$a_type,
    "mlir::Type":$b_type,
    "mlir::Type":$c_type,
    "mlir::Type":$result_type,
    "mlir::spirv::ScopeAttr":$scope
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPV_JointMatrixPropertiesINTELArrayAttr :
    TypedArrayAttrBase<SPV_JointMatrixPropertiesINTELAttr,
                       "JointMatrixPropertiesINTEL array attribute">;

// This attribute specifies the limits for various resources on the target
// architecture.
//
// See https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits
// for the complete list of limits and their explanation for the Vulkan API.
// The following ones are those affecting SPIR-V CodeGen. Their default value
// are the from Vulkan limit requirements:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax
def SPV_ResourceLimitsAttr : SPV_Attr<"ResourceLimits", "resource_limits"> {
  let parameters = (ins
    // The maximum total storage size, in bytes, available for variables
    // declared with the Workgroup storage class.
    DefaultValuedParameter<"int", "16384">:$max_compute_shared_memory_size,

    // The maximum total number of compute shader invocations in a single local
    // workgroup.
    DefaultValuedParameter<"int", "128">:$max_compute_workgroup_invocations,
    // The maximum size of a local compute workgroup, per dimension.
    DefaultValuedParameter<
      "ArrayAttr",
      "$_builder.getI32ArrayAttr({128, 128, 64})"
    >:$max_compute_workgroup_size,

    // The default number of invocations in each subgroup.
    DefaultValuedParameter<"int", "32">:$subgroup_size,

    // The configurations of cooperative matrix operations
    // supported. Default is an empty list.
    DefaultValuedParameter<
      "ArrayAttr",
      "nullptr"
    >:$cooperative_matrix_properties_nv
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // MLIR_DIALECT_SPIRV_IR_TARGET_AND_ABI
