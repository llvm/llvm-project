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

class SPIRV_Attr<string attrName, string attrMnemonic>
    : AttrDef<SPIRV_Dialect, attrName> {
  let mnemonic = attrMnemonic;
}

// For entry functions, this attribute specifies information related to entry
// points in the generated SPIR-V module:
// 1) [optional] Requested workgroup size.
// 2) [optional] Requested subgroup size.
// 3) [optional] Requested target width.
def SPIRV_EntryPointABIAttr : SPIRV_Attr<"EntryPointABI", "entry_point_abi"> {
  let parameters = (ins
    OptionalParameter<"DenseI32ArrayAttr">:$workgroup_size,
    OptionalParameter<"std::optional<int>">:$subgroup_size,
    OptionalParameter<"std::optional<int>">:$target_width
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPIRV_ExtensionArrayAttr : TypedArrayAttrBase<
    SPIRV_ExtensionAttr, "SPIR-V extension array attribute">;

def SPIRV_CapabilityArrayAttr : TypedArrayAttrBase<
    SPIRV_CapabilityAttr, "SPIR-V capability array attribute">;

def SPIRV_LinkageAttributesAttr : SPIRV_Attr<"LinkageAttributes", "linkage_attributes"> {
  let parameters = (ins
    "StringAttr":$linkage_name,
    "mlir::spirv::LinkageTypeAttr":$linkage_type
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// Description of cooperative matrix operations supported on the
// target. Represents `VkCooperativeMatrixPropertiesKHR`. See
// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCooperativeMatrixPropertiesKHR.html
def SPIRV_CooperativeMatrixPropertiesKHRAttr :
    SPIRV_Attr<"CooperativeMatrixPropertiesKHR", "coop_matrix_props_khr"> {
  let parameters = (ins
    "uint32_t":$m_size,
    "uint32_t":$n_size,
    "uint32_t":$k_size,
    "mlir::Type":$a_type,
    "mlir::Type":$b_type,
    "mlir::Type":$c_type,
    "mlir::Type":$result_type,
    "bool":$acc_sat,
    "mlir::spirv::ScopeAttr":$scope
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPIRV_CooperativeMatrixPropertiesKHRArrayAttr :
    TypedArrayAttrBase<SPIRV_CooperativeMatrixPropertiesKHRAttr,
                       "CooperativeMatrixPropertiesKHR array attribute">;

// Description of cooperative matrix operations supported on the
// target. Represents `VkCooperativeMatrixPropertiesNV`. See
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkCooperativeMatrixPropertiesNV.html
def SPIRV_CooperativeMatrixPropertiesNVAttr :
    SPIRV_Attr<"CooperativeMatrixPropertiesNV", "coop_matrix_props_nv"> {
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

def SPIRV_CacheControlLoadINTELAttr :
    SPIRV_Attr<"CacheControlLoadINTEL", "cache_control_load_intel"> {
  let parameters = (ins "unsigned":$cache_level,
                        "mlir::spirv::LoadCacheControl":$load_cache_control);
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPIRV_CacheControlStoreINTELAttr :
    SPIRV_Attr<"CacheControlStoreINTEL", "cache_control_store_intel"> {
  let parameters = (ins "unsigned":$cache_level,
                        "mlir::spirv::StoreCacheControl":$store_cache_control);
  let assemblyFormat = "`<` struct(params) `>`";
}

def SPIRV_CooperativeMatrixPropertiesNVArrayAttr :
    TypedArrayAttrBase<SPIRV_CooperativeMatrixPropertiesNVAttr,
                       "CooperativeMatrixPropertiesNV array attribute">;

// This attribute specifies the limits for various resources on the target
// architecture.
//
// See https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits
// for the complete list of limits and their explanation for the Vulkan API.
// The following ones are those affecting SPIR-V CodeGen. Their default value
// are the from Vulkan limit requirements:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax
def SPIRV_ResourceLimitsAttr : SPIRV_Attr<"ResourceLimits", "resource_limits"> {
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

    // The minimum supported size if the subgroup size is controllable.
    OptionalParameter<"std::optional<int>">:$min_subgroup_size,
    // The maximum supported size if the subgroup size is controllable.
    OptionalParameter<"std::optional<int>">:$max_subgroup_size,

    // The configurations of cooperative matrix operations
    // supported. Default is an empty list.
    DefaultValuedParameter<
      "ArrayAttr",
      "nullptr"
    >:$cooperative_matrix_properties_khr,

    DefaultValuedParameter<
      "ArrayAttr",
      "nullptr"
    >:$cooperative_matrix_properties_nv
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // MLIR_DIALECT_SPIRV_IR_TARGET_AND_ABI
