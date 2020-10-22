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

#ifndef SPIRV_TARGET_AND_ABI
#define SPIRV_TARGET_AND_ABI

include "mlir/Dialect/SPIRV/SPIRVBase.td"

// For entry functions, this attribute specifies information related to entry
// points in the generated SPIR-V module:
// 1) WorkGroup Size.
def SPV_EntryPointABIAttr : StructAttr<"EntryPointABIAttr", SPIRV_Dialect, [
    StructFieldAttr<"local_size", I32ElementsAttr>
]>;

def SPV_ExtensionArrayAttr : TypedArrayAttrBase<
    SPV_ExtensionAttr, "SPIR-V extension array attribute">;

def SPV_CapabilityArrayAttr : TypedArrayAttrBase<
    SPV_CapabilityAttr, "SPIR-V capability array attribute">;

// Description of cooperative matrix operations supported on the
// target. Represents `VkCooperativeMatrixPropertiesNV`. See
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkCooperativeMatrixPropertiesNV.html
def SPV_CooperativeMatrixPropertiesNVAttr :
    StructAttr<"CooperativeMatrixPropertiesNVAttr", SPIRV_Dialect, [
    StructFieldAttr<"m_size", I32Attr>,
    StructFieldAttr<"n_size", I32Attr>,
    StructFieldAttr<"k_size", I32Attr>,
    StructFieldAttr<"a_type", TypeAttr>,
    StructFieldAttr<"b_type", TypeAttr>,
    StructFieldAttr<"c_type", TypeAttr>,
    StructFieldAttr<"result_type", TypeAttr>,
    StructFieldAttr<"scope", SPV_ScopeAttr>
]>;

def SPV_CooperativeMatrixPropertiesNVArrayAttr :
    TypedArrayAttrBase<SPV_CooperativeMatrixPropertiesNVAttr,
                       "CooperativeMatrixPropertiesNV array attribute">;

// This attribute specifies the limits for various resources on the target
// architecture.
//
// See https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits
// for the complete list of limits and their explanation for the Vulkan API.
// The following ones are those affecting SPIR-V CodeGen. Their default value
// are the from Vulkan limit requirements:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax
def SPV_ResourceLimitsAttr : StructAttr<"ResourceLimitsAttr", SPIRV_Dialect, [
    // The maximum total storage size, in bytes, available for variables
    // declared with the Workgroup storage class.
    StructFieldAttr<"max_compute_shared_memory_size",
                    DefaultValuedAttr<I32Attr, "16384">>,

    // The maximum total number of compute shader invocations in a single local
    // workgroup.
    StructFieldAttr<"max_compute_workgroup_invocations",
                    DefaultValuedAttr<I32Attr, "128">>,
    // The maximum size of a local compute workgroup, per dimension.
    StructFieldAttr<"max_compute_workgroup_size",
                    DefaultValuedAttr<I32ElementsAttr, "{128, 128, 64}">>,

    // The default number of invocations in each subgroup.
    // 0x7FFFFFFF means unknown.
    StructFieldAttr<"subgroup_size", DefaultValuedAttr<I32Attr, "0x7FFFFFFF">>,

    // The configurations of cooperative matrix operations
    // supported. Default is an empty list.
    StructFieldAttr<
        "cooperative_matrix_properties_nv",
        DefaultValuedAttr<SPV_CooperativeMatrixPropertiesNVArrayAttr, "{}">>
]>;

#endif // SPIRV_TARGET_AND_ABI
