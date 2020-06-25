//===- SPIRVBase.td - MLIR SPIR-V Op Definitions Base file -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for SPIR-V operation definition specification.
// This file defines the SPIR-V dialect, common SPIR-V types, and utilities
// for facilitating defining SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_BASE
#define SPIRV_BASE

include "mlir/IR/OpBase.td"
include "mlir/Dialect/SPIRV/SPIRVAvailability.td"

//===----------------------------------------------------------------------===//
// SPIR-V dialect definitions
//===----------------------------------------------------------------------===//

def SPIRV_Dialect : Dialect {
  let name = "spv";

  let summary = "The SPIR-V dialect in MLIR.";

  let description = [{
    SPIR-V is a binary intermediate language for representing graphical-shader
    stages and compute kernels for multiple Khronos APIs, including OpenCL,
    OpenGL, and Vulkan.
    See https://www.khronos.org/registry/spir-v for more details regarding
    SPIR-V itself.

    The SPIR-V dialect aims to be a proper compiler intermediate representation
    to facilitate transformations. Ops in this dialect stay at the same semantic
    level as the SPIR-V specification and try to have one-to-one mapping to the
    corresponding SPIR-V instructions; but they may deviate representationally
    to utilize MLIR mechanisms if it results in better representation and thus
    benefits transformations. The dialect also aims to maintain straightforward
    serialization into and deserialization from the SPIR-V binary format.
    See https://mlir.llvm.org/docs/Dialects/SPIR-V/ for more details regarding
    high-level designs and implementation structures of the SPIR-V dialect.
  }];

  let cppNamespace = "spirv";
  let hasConstantMaterializer = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Attribute
    //===------------------------------------------------------------------===//

    /// Returns the attribute name to use when specifying decorations on results
    /// of operations.
    static std::string getAttributeName(Decoration decoration);
  }];
}

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

// A predicate that checks whether `$_self` is a known enum case for the
// enum class with `name`.
class SPV_IsKnownEnumCaseFor<string name> :
    CPred<"::mlir::spirv::symbolize" # name # "("
          "$_self.cast<IntegerAttr>().getValue().getZExtValue()).hasValue()">;

// Wrapper over base BitEnumAttr to set common fields.
class SPV_BitEnumAttr<string name, string description,
                      list<BitEnumAttrCase> cases> :
    BitEnumAttr<name, description, cases> {
  let predicate = And<[
    I32Attr.predicate,
    SPV_IsKnownEnumCaseFor<name>,
  ]>;
  let cppNamespace = "::mlir::spirv";
}

// Wrapper over base I32EnumAttr to set common fields.
class SPV_I32EnumAttr<string name, string description,
                      list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, description, cases> {
  let predicate = And<[
    I32Attr.predicate,
    SPV_IsKnownEnumCaseFor<name>,
  ]>;
  let cppNamespace = "::mlir::spirv";
}

// Wrapper over base StrEnumAttr to set common fields.
class SPV_StrEnumAttr<string name, string description,
                      list<StrEnumAttrCase> cases> :
    StrEnumAttr<name, description, cases> {
  let predicate = And<[
    StrAttr.predicate,
    CPred<"::mlir::spirv::symbolize" # name # "("
          "$_self.cast<StringAttr>().getValue()).hasValue()">,
  ]>;
  let cppNamespace = "::mlir::spirv";
}

//===----------------------------------------------------------------------===//
// SPIR-V availability definitions
//===----------------------------------------------------------------------===//

def SPV_V_1_0 : I32EnumAttrCase<"V_1_0", 0, "v1.0">;
def SPV_V_1_1 : I32EnumAttrCase<"V_1_1", 1, "v1.1">;
def SPV_V_1_2 : I32EnumAttrCase<"V_1_2", 2, "v1.2">;
def SPV_V_1_3 : I32EnumAttrCase<"V_1_3", 3, "v1.3">;
def SPV_V_1_4 : I32EnumAttrCase<"V_1_4", 4, "v1.4">;
def SPV_V_1_5 : I32EnumAttrCase<"V_1_5", 5, "v1.5">;

def SPV_VersionAttr : SPV_I32EnumAttr<"Version", "valid SPIR-V version", [
    SPV_V_1_0, SPV_V_1_1, SPV_V_1_2, SPV_V_1_3, SPV_V_1_4, SPV_V_1_5]>;

class MinVersion<I32EnumAttrCase min> : MinVersionBase<
    "QueryMinVersionInterface", SPV_VersionAttr, min> {
  let interfaceDescription = [{
    Querying interface for minimal required SPIR-V version.

    This interface provides a `getMinVersion()` method to query the minimal
    required version for the implementing SPIR-V operation. The returned value
    is a `mlir::spirv::Version` enumerant.
  }];
}

class MaxVersion<I32EnumAttrCase max> : MaxVersionBase<
    "QueryMaxVersionInterface", SPV_VersionAttr, max> {
  let interfaceDescription = [{
    Querying interface for maximal supported SPIR-V version.

    This interface provides a `getMaxVersion()` method to query the maximal
    supported version for the implementing SPIR-V operation. The returned value
    is a `mlir::spirv::Version` enumerant.
  }];
}

class Extension<list<StrEnumAttrCase> extensions> : Availability {
  let interfaceName = "QueryExtensionInterface";
  let interfaceDescription = [{
    Querying interface for required SPIR-V extensions.

    This interface provides a `getExtensions()` method to query the required
    extensions for the implementing SPIR-V operation. The returned value
    is a nested vector whose element is `mlir::spirv::Extension`s. The outer
    vector's elements (which are vectors) should be interpreted as conjunction
    while the inner vector's elements (which are `mlir::spirv::Extension`s)
    should be interpreted as disjunction. For example, given

    ```
    {{Extension::A, Extension::B}, {Extension::C}, {{Extension::D, Extension::E}}
    ```

    The operation instance is available when (`Extension::A` OR `Extension::B`)
    AND (`Extension::C`) AND (`Extension::D` OR `Extension::E`) is enabled.
  }];

  // TODO(antiagainst): Returning SmallVector<ArrayRef<...>> is not recommended.
  // Find a better way for this.
  let queryFnRetType = "::llvm::SmallVector<::llvm::ArrayRef<"
                          "::mlir::spirv::Extension>, 1>";
  let queryFnName = "getExtensions";

  let mergeAction = !if(
      !empty(extensions), "", "$overall.emplace_back($instance)");
  let initializer = "{}";
  let instanceType = "::llvm::ArrayRef<::mlir::spirv::Extension>";

  // Pack all extensions as a static array and get its reference.
  let instancePreparation = !if(!empty(extensions), "",
    "static const ::mlir::spirv::Extension exts[] = {" #
    StrJoin<!foreach(ext, extensions,
      "::mlir::spirv::Extension::" # ext.symbol)>.result #
    "}; " #
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    "ArrayRef<::mlir::spirv::Extension> " #
      "ref(exts, ::llvm::array_lengthof(exts));");
  let instance = "ref";
}

class Capability<list<I32EnumAttrCase> capabilities> : Availability {
  let interfaceName = "QueryCapabilityInterface";
  let interfaceDescription = [{
    Querying interface for required SPIR-V capabilities.

    This interface provides a `getCapabilities()` method to query the required
    capabilities for the implementing SPIR-V operation. The returned value
    is a nested vector whose element is `mlir::spirv::Capability`s. The outer
    vector's elements (which are vectors) should be interpreted as conjunction
    while the inner vector's elements (which are `mlir::spirv::Capability`s)
    should be interpreted as disjunction. For example, given

    ```
    {{Capability::A, Capability::B}, {Capability::C}, {{Capability::D, Capability::E}}
    ```

    The operation instance is available when (`Capability::A` OR `Capability::B`)
    AND (`Capability::C`) AND (`Capability::D` OR `Capability::E`) is enabled.
  }];

  let queryFnRetType = "::llvm::SmallVector<::llvm::ArrayRef<"
                          "::mlir::spirv::Capability>, 1>";
  let queryFnName = "getCapabilities";

  let mergeAction = !if(
      !empty(capabilities), "", "$overall.emplace_back($instance)");
  let initializer = "{}";
  let instanceType = "::llvm::ArrayRef<::mlir::spirv::Capability>";

  // Pack all capabilities as a static array and get its reference.
  let instancePreparation = !if(!empty(capabilities), "",
    "static const ::mlir::spirv::Capability caps[] = {" #
    StrJoin<!foreach(cap, capabilities,
      "::mlir::spirv::Capability::" # cap.symbol)>.result #
    "}; " #
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    "ArrayRef<::mlir::spirv::Capability> " #
      "ref(caps, ::llvm::array_lengthof(caps));");
  let instance = "ref";
}

// TODO(antiagainst): the following interfaces definitions are duplicating with
// the above. Remove them once we are able to support dialect-specific contents
// in ODS.
def QueryMinVersionInterface : OpInterface<"QueryMinVersionInterface"> {
  let methods = [InterfaceMethod<"", "::mlir::spirv::Version", "getMinVersion">];
}
def QueryMaxVersionInterface : OpInterface<"QueryMaxVersionInterface"> {
  let methods = [InterfaceMethod<"", "::mlir::spirv::Version", "getMaxVersion">];
}
def QueryExtensionInterface : OpInterface<"QueryExtensionInterface"> {
  let methods = [InterfaceMethod<
    "",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1>",
    "getExtensions">];
}
def QueryCapabilityInterface : OpInterface<"QueryCapabilityInterface"> {
  let methods = [InterfaceMethod<
    "",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1>",
    "getCapabilities">];
}

//===----------------------------------------------------------------------===//
// SPIR-V extension definitions
//===----------------------------------------------------------------------===//

// Extensions known to the SPIR-V dialect.
// https://github.com/KhronosGroup/SPIRV-Registry has the full list.
def SPV_KHR_16bit_storage                : StrEnumAttrCase<"SPV_KHR_16bit_storage">;
def SPV_KHR_8bit_storage                 : StrEnumAttrCase<"SPV_KHR_8bit_storage">;
def SPV_KHR_device_group                 : StrEnumAttrCase<"SPV_KHR_device_group">;
def SPV_KHR_float_controls               : StrEnumAttrCase<"SPV_KHR_float_controls">;
def SPV_KHR_physical_storage_buffer      : StrEnumAttrCase<"SPV_KHR_physical_storage_buffer">;
def SPV_KHR_multiview                    : StrEnumAttrCase<"SPV_KHR_multiview">;
def SPV_KHR_no_integer_wrap_decoration   : StrEnumAttrCase<"SPV_KHR_no_integer_wrap_decoration">;
def SPV_KHR_post_depth_coverage          : StrEnumAttrCase<"SPV_KHR_post_depth_coverage">;
def SPV_KHR_shader_atomic_counter_ops    : StrEnumAttrCase<"SPV_KHR_shader_atomic_counter_ops">;
def SPV_KHR_shader_ballot                : StrEnumAttrCase<"SPV_KHR_shader_ballot">;
def SPV_KHR_shader_clock                 : StrEnumAttrCase<"SPV_KHR_shader_clock">;
def SPV_KHR_shader_draw_parameters       : StrEnumAttrCase<"SPV_KHR_shader_draw_parameters">;
def SPV_KHR_storage_buffer_storage_class : StrEnumAttrCase<"SPV_KHR_storage_buffer_storage_class">;
def SPV_KHR_subgroup_vote                : StrEnumAttrCase<"SPV_KHR_subgroup_vote">;
def SPV_KHR_variable_pointers            : StrEnumAttrCase<"SPV_KHR_variable_pointers">;
def SPV_KHR_vulkan_memory_model          : StrEnumAttrCase<"SPV_KHR_vulkan_memory_model">;

def SPV_EXT_demote_to_helper_invocation  : StrEnumAttrCase<"SPV_EXT_demote_to_helper_invocation">;
def SPV_EXT_descriptor_indexing          : StrEnumAttrCase<"SPV_EXT_descriptor_indexing">;
def SPV_EXT_fragment_fully_covered       : StrEnumAttrCase<"SPV_EXT_fragment_fully_covered">;
def SPV_EXT_fragment_invocation_density  : StrEnumAttrCase<"SPV_EXT_fragment_invocation_density">;
def SPV_EXT_fragment_shader_interlock    : StrEnumAttrCase<"SPV_EXT_fragment_shader_interlock">;
def SPV_EXT_physical_storage_buffer      : StrEnumAttrCase<"SPV_EXT_physical_storage_buffer">;
def SPV_EXT_shader_stencil_export        : StrEnumAttrCase<"SPV_EXT_shader_stencil_export">;
def SPV_EXT_shader_viewport_index_layer  : StrEnumAttrCase<"SPV_EXT_shader_viewport_index_layer">;

def SPV_AMD_gpu_shader_half_float_fetch      : StrEnumAttrCase<"SPV_AMD_gpu_shader_half_float_fetch">;
def SPV_AMD_shader_ballot                    : StrEnumAttrCase<"SPV_AMD_shader_ballot">;
def SPV_AMD_shader_explicit_vertex_parameter : StrEnumAttrCase<"SPV_AMD_shader_explicit_vertex_parameter">;
def SPV_AMD_shader_fragment_mask             : StrEnumAttrCase<"SPV_AMD_shader_fragment_mask">;
def SPV_AMD_shader_image_load_store_lod      : StrEnumAttrCase<"SPV_AMD_shader_image_load_store_lod">;
def SPV_AMD_texture_gather_bias_lod          : StrEnumAttrCase<"SPV_AMD_texture_gather_bias_lod">;

def SPV_GOOGLE_decorate_string           : StrEnumAttrCase<"SPV_GOOGLE_decorate_string">;
def SPV_GOOGLE_hlsl_functionality1       : StrEnumAttrCase<"SPV_GOOGLE_hlsl_functionality1">;
def SPV_GOOGLE_user_type                 : StrEnumAttrCase<"SPV_GOOGLE_user_type">;

def SPV_INTEL_device_side_avc_motion_estimation : StrEnumAttrCase<"SPV_INTEL_device_side_avc_motion_estimation">;
def SPV_INTEL_media_block_io                    : StrEnumAttrCase<"SPV_INTEL_media_block_io">;
def SPV_INTEL_shader_integer_functions2         : StrEnumAttrCase<"SPV_INTEL_shader_integer_functions2">;
def SPV_INTEL_subgroups                         : StrEnumAttrCase<"SPV_INTEL_subgroups">;

def SPV_NV_compute_shader_derivatives    : StrEnumAttrCase<"SPV_NV_compute_shader_derivatives">;
def SPV_NV_cooperative_matrix            : StrEnumAttrCase<"SPV_NV_cooperative_matrix">;
def SPV_NV_fragment_shader_barycentric   : StrEnumAttrCase<"SPV_NV_fragment_shader_barycentric">;
def SPV_NV_geometry_shader_passthrough   : StrEnumAttrCase<"SPV_NV_geometry_shader_passthrough">;
def SPV_NV_mesh_shader                   : StrEnumAttrCase<"SPV_NV_mesh_shader">;
def SPV_NV_ray_tracing                   : StrEnumAttrCase<"SPV_NV_ray_tracing">;
def SPV_NV_sample_mask_override_coverage : StrEnumAttrCase<"SPV_NV_sample_mask_override_coverage">;
def SPV_NV_shader_image_footprint        : StrEnumAttrCase<"SPV_NV_shader_image_footprint">;
def SPV_NV_shader_sm_builtins            : StrEnumAttrCase<"SPV_NV_shader_sm_builtins">;
def SPV_NV_shader_subgroup_partitioned   : StrEnumAttrCase<"SPV_NV_shader_subgroup_partitioned">;
def SPV_NV_shading_rate                  : StrEnumAttrCase<"SPV_NV_shading_rate">;
def SPV_NV_stereo_view_rendering         : StrEnumAttrCase<"SPV_NV_stereo_view_rendering">;
def SPV_NV_viewport_array2               : StrEnumAttrCase<"SPV_NV_viewport_array2">;

def SPV_NVX_multiview_per_view_attributes : StrEnumAttrCase<"SPV_NVX_multiview_per_view_attributes">;

def SPV_ExtensionAttr :
    SPV_StrEnumAttr<"Extension", "supported SPIR-V extensions", [
      SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_device_group,
      SPV_KHR_float_controls, SPV_KHR_physical_storage_buffer, SPV_KHR_multiview,
      SPV_KHR_no_integer_wrap_decoration, SPV_KHR_post_depth_coverage,
      SPV_KHR_shader_atomic_counter_ops, SPV_KHR_shader_ballot,
      SPV_KHR_shader_clock, SPV_KHR_shader_draw_parameters,
      SPV_KHR_storage_buffer_storage_class, SPV_KHR_subgroup_vote,
      SPV_KHR_variable_pointers, SPV_KHR_vulkan_memory_model,
      SPV_EXT_demote_to_helper_invocation, SPV_EXT_descriptor_indexing,
      SPV_EXT_fragment_fully_covered, SPV_EXT_fragment_invocation_density,
      SPV_EXT_fragment_shader_interlock, SPV_EXT_physical_storage_buffer,
      SPV_EXT_shader_stencil_export, SPV_EXT_shader_viewport_index_layer,
      SPV_AMD_gpu_shader_half_float_fetch, SPV_AMD_shader_ballot,
      SPV_AMD_shader_explicit_vertex_parameter, SPV_AMD_shader_fragment_mask,
      SPV_AMD_shader_image_load_store_lod, SPV_AMD_texture_gather_bias_lod,
      SPV_GOOGLE_decorate_string, SPV_GOOGLE_hlsl_functionality1, SPV_GOOGLE_user_type,
      SPV_INTEL_device_side_avc_motion_estimation, SPV_INTEL_media_block_io,
      SPV_INTEL_shader_integer_functions2, SPV_INTEL_subgroups,
      SPV_NV_compute_shader_derivatives, SPV_NV_cooperative_matrix,
      SPV_NV_fragment_shader_barycentric, SPV_NV_geometry_shader_passthrough,
      SPV_NV_mesh_shader, SPV_NV_ray_tracing, SPV_NV_sample_mask_override_coverage,
      SPV_NV_shader_image_footprint, SPV_NV_shader_sm_builtins,
      SPV_NV_shader_subgroup_partitioned, SPV_NV_shading_rate,
      SPV_NV_stereo_view_rendering, SPV_NV_viewport_array2,
      SPV_NVX_multiview_per_view_attributes
    ]>;

//===----------------------------------------------------------------------===//
// SPIR-V enum definitions
//===----------------------------------------------------------------------===//

// Begin enum section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPV_C_Matrix                                    : I32EnumAttrCase<"Matrix", 0>;
def SPV_C_Addresses                                 : I32EnumAttrCase<"Addresses", 4>;
def SPV_C_Linkage                                   : I32EnumAttrCase<"Linkage", 5>;
def SPV_C_Kernel                                    : I32EnumAttrCase<"Kernel", 6>;
def SPV_C_Float16                                   : I32EnumAttrCase<"Float16", 9>;
def SPV_C_Float64                                   : I32EnumAttrCase<"Float64", 10>;
def SPV_C_Int64                                     : I32EnumAttrCase<"Int64", 11>;
def SPV_C_Groups                                    : I32EnumAttrCase<"Groups", 18> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_ballot]>
  ];
}
def SPV_C_Int16                                     : I32EnumAttrCase<"Int16", 22>;
def SPV_C_Int8                                      : I32EnumAttrCase<"Int8", 39>;
def SPV_C_Sampled1D                                 : I32EnumAttrCase<"Sampled1D", 43>;
def SPV_C_SampledBuffer                             : I32EnumAttrCase<"SampledBuffer", 46>;
def SPV_C_GroupNonUniform                           : I32EnumAttrCase<"GroupNonUniform", 61> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_ShaderLayer                               : I32EnumAttrCase<"ShaderLayer", 69> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_ShaderViewportIndex                       : I32EnumAttrCase<"ShaderViewportIndex", 70> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_SubgroupBallotKHR                         : I32EnumAttrCase<"SubgroupBallotKHR", 4423> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_ballot]>
  ];
}
def SPV_C_SubgroupVoteKHR                           : I32EnumAttrCase<"SubgroupVoteKHR", 4431> {
  list<Availability> availability = [
    Extension<[SPV_KHR_subgroup_vote]>
  ];
}
def SPV_C_StorageBuffer16BitAccess                  : I32EnumAttrCase<"StorageBuffer16BitAccess", 4433> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPV_C_StoragePushConstant16                     : I32EnumAttrCase<"StoragePushConstant16", 4435> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPV_C_StorageInputOutput16                      : I32EnumAttrCase<"StorageInputOutput16", 4436> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPV_C_DeviceGroup                               : I32EnumAttrCase<"DeviceGroup", 4437> {
  list<Availability> availability = [
    Extension<[SPV_KHR_device_group]>
  ];
}
def SPV_C_AtomicStorageOps                          : I32EnumAttrCase<"AtomicStorageOps", 4445> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_atomic_counter_ops]>
  ];
}
def SPV_C_SampleMaskPostDepthCoverage               : I32EnumAttrCase<"SampleMaskPostDepthCoverage", 4447> {
  list<Availability> availability = [
    Extension<[SPV_KHR_post_depth_coverage]>
  ];
}
def SPV_C_StorageBuffer8BitAccess                   : I32EnumAttrCase<"StorageBuffer8BitAccess", 4448> {
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPV_C_StoragePushConstant8                      : I32EnumAttrCase<"StoragePushConstant8", 4450> {
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPV_C_DenormPreserve                            : I32EnumAttrCase<"DenormPreserve", 4464> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPV_C_DenormFlushToZero                         : I32EnumAttrCase<"DenormFlushToZero", 4465> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPV_C_SignedZeroInfNanPreserve                  : I32EnumAttrCase<"SignedZeroInfNanPreserve", 4466> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPV_C_RoundingModeRTE                           : I32EnumAttrCase<"RoundingModeRTE", 4467> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPV_C_RoundingModeRTZ                           : I32EnumAttrCase<"RoundingModeRTZ", 4468> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPV_C_ImageFootprintNV                          : I32EnumAttrCase<"ImageFootprintNV", 5282> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_image_footprint]>
  ];
}
def SPV_C_FragmentBarycentricNV                     : I32EnumAttrCase<"FragmentBarycentricNV", 5284> {
  list<Availability> availability = [
    Extension<[SPV_NV_fragment_shader_barycentric]>
  ];
}
def SPV_C_ComputeDerivativeGroupQuadsNV             : I32EnumAttrCase<"ComputeDerivativeGroupQuadsNV", 5288> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>
  ];
}
def SPV_C_GroupNonUniformPartitionedNV              : I32EnumAttrCase<"GroupNonUniformPartitionedNV", 5297> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>
  ];
}
def SPV_C_VulkanMemoryModel                         : I32EnumAttrCase<"VulkanMemoryModel", 5345> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_VulkanMemoryModelDeviceScope              : I32EnumAttrCase<"VulkanMemoryModelDeviceScope", 5346> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_ComputeDerivativeGroupLinearNV            : I32EnumAttrCase<"ComputeDerivativeGroupLinearNV", 5350> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>
  ];
}
def SPV_C_SubgroupShuffleINTEL                      : I32EnumAttrCase<"SubgroupShuffleINTEL", 5568> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPV_C_SubgroupBufferBlockIOINTEL                : I32EnumAttrCase<"SubgroupBufferBlockIOINTEL", 5569> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPV_C_SubgroupImageBlockIOINTEL                 : I32EnumAttrCase<"SubgroupImageBlockIOINTEL", 5570> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPV_C_SubgroupImageMediaBlockIOINTEL            : I32EnumAttrCase<"SubgroupImageMediaBlockIOINTEL", 5579> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_media_block_io]>
  ];
}
def SPV_C_SubgroupAvcMotionEstimationINTEL          : I32EnumAttrCase<"SubgroupAvcMotionEstimationINTEL", 5696> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPV_C_SubgroupAvcMotionEstimationIntraINTEL     : I32EnumAttrCase<"SubgroupAvcMotionEstimationIntraINTEL", 5697> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPV_C_SubgroupAvcMotionEstimationChromaINTEL    : I32EnumAttrCase<"SubgroupAvcMotionEstimationChromaINTEL", 5698> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPV_C_Shader                                    : I32EnumAttrCase<"Shader", 1> {
  list<I32EnumAttrCase> implies = [SPV_C_Matrix];
}
def SPV_C_Vector16                                  : I32EnumAttrCase<"Vector16", 7> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_Float16Buffer                             : I32EnumAttrCase<"Float16Buffer", 8> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_Int64Atomics                              : I32EnumAttrCase<"Int64Atomics", 12> {
  list<I32EnumAttrCase> implies = [SPV_C_Int64];
}
def SPV_C_ImageBasic                                : I32EnumAttrCase<"ImageBasic", 13> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_Pipes                                     : I32EnumAttrCase<"Pipes", 17> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_DeviceEnqueue                             : I32EnumAttrCase<"DeviceEnqueue", 19> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_LiteralSampler                            : I32EnumAttrCase<"LiteralSampler", 20> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
}
def SPV_C_GenericPointer                            : I32EnumAttrCase<"GenericPointer", 38> {
  list<I32EnumAttrCase> implies = [SPV_C_Addresses];
}
def SPV_C_Image1D                                   : I32EnumAttrCase<"Image1D", 44> {
  list<I32EnumAttrCase> implies = [SPV_C_Sampled1D];
}
def SPV_C_ImageBuffer                               : I32EnumAttrCase<"ImageBuffer", 47> {
  list<I32EnumAttrCase> implies = [SPV_C_SampledBuffer];
}
def SPV_C_NamedBarrier                              : I32EnumAttrCase<"NamedBarrier", 59> {
  list<I32EnumAttrCase> implies = [SPV_C_Kernel];
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>
  ];
}
def SPV_C_GroupNonUniformVote                       : I32EnumAttrCase<"GroupNonUniformVote", 62> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformArithmetic                 : I32EnumAttrCase<"GroupNonUniformArithmetic", 63> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformBallot                     : I32EnumAttrCase<"GroupNonUniformBallot", 64> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformShuffle                    : I32EnumAttrCase<"GroupNonUniformShuffle", 65> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformShuffleRelative            : I32EnumAttrCase<"GroupNonUniformShuffleRelative", 66> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformClustered                  : I32EnumAttrCase<"GroupNonUniformClustered", 67> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_GroupNonUniformQuad                       : I32EnumAttrCase<"GroupNonUniformQuad", 68> {
  list<I32EnumAttrCase> implies = [SPV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>
  ];
}
def SPV_C_StorageUniform16                          : I32EnumAttrCase<"StorageUniform16", 4434> {
  list<I32EnumAttrCase> implies = [SPV_C_StorageBuffer16BitAccess];
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPV_C_UniformAndStorageBuffer8BitAccess         : I32EnumAttrCase<"UniformAndStorageBuffer8BitAccess", 4449> {
  list<I32EnumAttrCase> implies = [SPV_C_StorageBuffer8BitAccess];
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPV_C_UniformTexelBufferArrayDynamicIndexing    : I32EnumAttrCase<"UniformTexelBufferArrayDynamicIndexing", 5304> {
  list<I32EnumAttrCase> implies = [SPV_C_SampledBuffer];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_Geometry                                  : I32EnumAttrCase<"Geometry", 2> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_Tessellation                              : I32EnumAttrCase<"Tessellation", 3> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_ImageReadWrite                            : I32EnumAttrCase<"ImageReadWrite", 14> {
  list<I32EnumAttrCase> implies = [SPV_C_ImageBasic];
}
def SPV_C_ImageMipmap                               : I32EnumAttrCase<"ImageMipmap", 15> {
  list<I32EnumAttrCase> implies = [SPV_C_ImageBasic];
}
def SPV_C_AtomicStorage                             : I32EnumAttrCase<"AtomicStorage", 21> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_ImageGatherExtended                       : I32EnumAttrCase<"ImageGatherExtended", 25> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageImageMultisample                   : I32EnumAttrCase<"StorageImageMultisample", 27> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_UniformBufferArrayDynamicIndexing         : I32EnumAttrCase<"UniformBufferArrayDynamicIndexing", 28> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SampledImageArrayDynamicIndexing          : I32EnumAttrCase<"SampledImageArrayDynamicIndexing", 29> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageBufferArrayDynamicIndexing         : I32EnumAttrCase<"StorageBufferArrayDynamicIndexing", 30> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageImageArrayDynamicIndexing          : I32EnumAttrCase<"StorageImageArrayDynamicIndexing", 31> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_ClipDistance                              : I32EnumAttrCase<"ClipDistance", 32> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_CullDistance                              : I32EnumAttrCase<"CullDistance", 33> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SampleRateShading                         : I32EnumAttrCase<"SampleRateShading", 35> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SampledRect                               : I32EnumAttrCase<"SampledRect", 37> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_InputAttachment                           : I32EnumAttrCase<"InputAttachment", 40> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SparseResidency                           : I32EnumAttrCase<"SparseResidency", 41> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_MinLod                                    : I32EnumAttrCase<"MinLod", 42> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SampledCubeArray                          : I32EnumAttrCase<"SampledCubeArray", 45> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_ImageMSArray                              : I32EnumAttrCase<"ImageMSArray", 48> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageImageExtendedFormats               : I32EnumAttrCase<"StorageImageExtendedFormats", 49> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_ImageQuery                                : I32EnumAttrCase<"ImageQuery", 50> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_DerivativeControl                         : I32EnumAttrCase<"DerivativeControl", 51> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_InterpolationFunction                     : I32EnumAttrCase<"InterpolationFunction", 52> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_TransformFeedback                         : I32EnumAttrCase<"TransformFeedback", 53> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageImageReadWithoutFormat             : I32EnumAttrCase<"StorageImageReadWithoutFormat", 55> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_StorageImageWriteWithoutFormat            : I32EnumAttrCase<"StorageImageWriteWithoutFormat", 56> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
}
def SPV_C_SubgroupDispatch                          : I32EnumAttrCase<"SubgroupDispatch", 58> {
  list<I32EnumAttrCase> implies = [SPV_C_DeviceEnqueue];
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>
  ];
}
def SPV_C_PipeStorage                               : I32EnumAttrCase<"PipeStorage", 60> {
  list<I32EnumAttrCase> implies = [SPV_C_Pipes];
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>
  ];
}
def SPV_C_DrawParameters                            : I32EnumAttrCase<"DrawParameters", 4427> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>
  ];
}
def SPV_C_MultiView                                 : I32EnumAttrCase<"MultiView", 4439> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_multiview]>
  ];
}
def SPV_C_VariablePointersStorageBuffer             : I32EnumAttrCase<"VariablePointersStorageBuffer", 4441> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_variable_pointers]>
  ];
}
def SPV_C_Float16ImageAMD                           : I32EnumAttrCase<"Float16ImageAMD", 5008> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_gpu_shader_half_float_fetch]>
  ];
}
def SPV_C_ImageGatherBiasLodAMD                     : I32EnumAttrCase<"ImageGatherBiasLodAMD", 5009> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_texture_gather_bias_lod]>
  ];
}
def SPV_C_FragmentMaskAMD                           : I32EnumAttrCase<"FragmentMaskAMD", 5010> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_fragment_mask]>
  ];
}
def SPV_C_StencilExportEXT                          : I32EnumAttrCase<"StencilExportEXT", 5013> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>
  ];
}
def SPV_C_ImageReadWriteLodAMD                      : I32EnumAttrCase<"ImageReadWriteLodAMD", 5015> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_image_load_store_lod]>
  ];
}
def SPV_C_ShaderClockKHR                            : I32EnumAttrCase<"ShaderClockKHR", 5055> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_clock]>
  ];
}
def SPV_C_FragmentFullyCoveredEXT                   : I32EnumAttrCase<"FragmentFullyCoveredEXT", 5265> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_fully_covered]>
  ];
}
def SPV_C_MeshShadingNV                             : I32EnumAttrCase<"MeshShadingNV", 5266> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>
  ];
}
def SPV_C_FragmentDensityEXT                        : I32EnumAttrCase<"FragmentDensityEXT", 5291> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>
  ];
}
def SPV_C_ShaderNonUniform                          : I32EnumAttrCase<"ShaderNonUniform", 5301> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_RuntimeDescriptorArray                    : I32EnumAttrCase<"RuntimeDescriptorArray", 5302> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_StorageTexelBufferArrayDynamicIndexing    : I32EnumAttrCase<"StorageTexelBufferArrayDynamicIndexing", 5305> {
  list<I32EnumAttrCase> implies = [SPV_C_ImageBuffer];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_RayTracingNV                              : I32EnumAttrCase<"RayTracingNV", 5340> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>
  ];
}
def SPV_C_PhysicalStorageBufferAddresses            : I32EnumAttrCase<"PhysicalStorageBufferAddresses", 5347> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>
  ];
}
def SPV_C_CooperativeMatrixNV                       : I32EnumAttrCase<"CooperativeMatrixNV", 5357> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_cooperative_matrix]>
  ];
}
def SPV_C_FragmentShaderSampleInterlockEXT          : I32EnumAttrCase<"FragmentShaderSampleInterlockEXT", 5363> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPV_C_FragmentShaderShadingRateInterlockEXT     : I32EnumAttrCase<"FragmentShaderShadingRateInterlockEXT", 5372> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPV_C_ShaderSMBuiltinsNV                        : I32EnumAttrCase<"ShaderSMBuiltinsNV", 5373> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>
  ];
}
def SPV_C_FragmentShaderPixelInterlockEXT           : I32EnumAttrCase<"FragmentShaderPixelInterlockEXT", 5378> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPV_C_DemoteToHelperInvocationEXT               : I32EnumAttrCase<"DemoteToHelperInvocationEXT", 5379> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_demote_to_helper_invocation]>
  ];
}
def SPV_C_IntegerFunctions2INTEL                    : I32EnumAttrCase<"IntegerFunctions2INTEL", 5584> {
  list<I32EnumAttrCase> implies = [SPV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_INTEL_shader_integer_functions2]>
  ];
}
def SPV_C_TessellationPointSize                     : I32EnumAttrCase<"TessellationPointSize", 23> {
  list<I32EnumAttrCase> implies = [SPV_C_Tessellation];
}
def SPV_C_GeometryPointSize                         : I32EnumAttrCase<"GeometryPointSize", 24> {
  list<I32EnumAttrCase> implies = [SPV_C_Geometry];
}
def SPV_C_ImageCubeArray                            : I32EnumAttrCase<"ImageCubeArray", 34> {
  list<I32EnumAttrCase> implies = [SPV_C_SampledCubeArray];
}
def SPV_C_ImageRect                                 : I32EnumAttrCase<"ImageRect", 36> {
  list<I32EnumAttrCase> implies = [SPV_C_SampledRect];
}
def SPV_C_GeometryStreams                           : I32EnumAttrCase<"GeometryStreams", 54> {
  list<I32EnumAttrCase> implies = [SPV_C_Geometry];
}
def SPV_C_MultiViewport                             : I32EnumAttrCase<"MultiViewport", 57> {
  list<I32EnumAttrCase> implies = [SPV_C_Geometry];
}
def SPV_C_VariablePointers                          : I32EnumAttrCase<"VariablePointers", 4442> {
  list<I32EnumAttrCase> implies = [SPV_C_VariablePointersStorageBuffer];
  list<Availability> availability = [
    Extension<[SPV_KHR_variable_pointers]>
  ];
}
def SPV_C_SampleMaskOverrideCoverageNV              : I32EnumAttrCase<"SampleMaskOverrideCoverageNV", 5249> {
  list<I32EnumAttrCase> implies = [SPV_C_SampleRateShading];
  list<Availability> availability = [
    Extension<[SPV_NV_sample_mask_override_coverage]>
  ];
}
def SPV_C_GeometryShaderPassthroughNV               : I32EnumAttrCase<"GeometryShaderPassthroughNV", 5251> {
  list<I32EnumAttrCase> implies = [SPV_C_Geometry];
  list<Availability> availability = [
    Extension<[SPV_NV_geometry_shader_passthrough]>
  ];
}
def SPV_C_PerViewAttributesNV                       : I32EnumAttrCase<"PerViewAttributesNV", 5260> {
  list<I32EnumAttrCase> implies = [SPV_C_MultiView];
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes]>
  ];
}
def SPV_C_InputAttachmentArrayDynamicIndexing       : I32EnumAttrCase<"InputAttachmentArrayDynamicIndexing", 5303> {
  list<I32EnumAttrCase> implies = [SPV_C_InputAttachment];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_UniformBufferArrayNonUniformIndexing      : I32EnumAttrCase<"UniformBufferArrayNonUniformIndexing", 5306> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_SampledImageArrayNonUniformIndexing       : I32EnumAttrCase<"SampledImageArrayNonUniformIndexing", 5307> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_StorageBufferArrayNonUniformIndexing      : I32EnumAttrCase<"StorageBufferArrayNonUniformIndexing", 5308> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_StorageImageArrayNonUniformIndexing       : I32EnumAttrCase<"StorageImageArrayNonUniformIndexing", 5309> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_InputAttachmentArrayNonUniformIndexing    : I32EnumAttrCase<"InputAttachmentArrayNonUniformIndexing", 5310> {
  list<I32EnumAttrCase> implies = [SPV_C_InputAttachment, SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_UniformTexelBufferArrayNonUniformIndexing : I32EnumAttrCase<"UniformTexelBufferArrayNonUniformIndexing", 5311> {
  list<I32EnumAttrCase> implies = [SPV_C_SampledBuffer, SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_StorageTexelBufferArrayNonUniformIndexing : I32EnumAttrCase<"StorageTexelBufferArrayNonUniformIndexing", 5312> {
  list<I32EnumAttrCase> implies = [SPV_C_ImageBuffer, SPV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>
  ];
}
def SPV_C_ShaderViewportIndexLayerEXT               : I32EnumAttrCase<"ShaderViewportIndexLayerEXT", 5254> {
  list<I32EnumAttrCase> implies = [SPV_C_MultiViewport];
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_viewport_index_layer]>
  ];
}
def SPV_C_ShaderViewportMaskNV                      : I32EnumAttrCase<"ShaderViewportMaskNV", 5255> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderViewportIndexLayerEXT];
  list<Availability> availability = [
    Extension<[SPV_NV_viewport_array2]>
  ];
}
def SPV_C_ShaderStereoViewNV                        : I32EnumAttrCase<"ShaderStereoViewNV", 5259> {
  list<I32EnumAttrCase> implies = [SPV_C_ShaderViewportMaskNV];
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>
  ];
}

def SPV_CapabilityAttr :
    SPV_I32EnumAttr<"Capability", "valid SPIR-V Capability", [
      SPV_C_Matrix, SPV_C_Addresses, SPV_C_Linkage, SPV_C_Kernel, SPV_C_Float16,
      SPV_C_Float64, SPV_C_Int64, SPV_C_Groups, SPV_C_Int16, SPV_C_Int8,
      SPV_C_Sampled1D, SPV_C_SampledBuffer, SPV_C_GroupNonUniform, SPV_C_ShaderLayer,
      SPV_C_ShaderViewportIndex, SPV_C_SubgroupBallotKHR, SPV_C_SubgroupVoteKHR,
      SPV_C_StorageBuffer16BitAccess, SPV_C_StoragePushConstant16,
      SPV_C_StorageInputOutput16, SPV_C_DeviceGroup, SPV_C_AtomicStorageOps,
      SPV_C_SampleMaskPostDepthCoverage, SPV_C_StorageBuffer8BitAccess,
      SPV_C_StoragePushConstant8, SPV_C_DenormPreserve, SPV_C_DenormFlushToZero,
      SPV_C_SignedZeroInfNanPreserve, SPV_C_RoundingModeRTE, SPV_C_RoundingModeRTZ,
      SPV_C_ImageFootprintNV, SPV_C_FragmentBarycentricNV,
      SPV_C_ComputeDerivativeGroupQuadsNV, SPV_C_GroupNonUniformPartitionedNV,
      SPV_C_VulkanMemoryModel, SPV_C_VulkanMemoryModelDeviceScope,
      SPV_C_ComputeDerivativeGroupLinearNV, SPV_C_SubgroupShuffleINTEL,
      SPV_C_SubgroupBufferBlockIOINTEL, SPV_C_SubgroupImageBlockIOINTEL,
      SPV_C_SubgroupImageMediaBlockIOINTEL, SPV_C_SubgroupAvcMotionEstimationINTEL,
      SPV_C_SubgroupAvcMotionEstimationIntraINTEL,
      SPV_C_SubgroupAvcMotionEstimationChromaINTEL, SPV_C_Shader, SPV_C_Vector16,
      SPV_C_Float16Buffer, SPV_C_Int64Atomics, SPV_C_ImageBasic, SPV_C_Pipes,
      SPV_C_DeviceEnqueue, SPV_C_LiteralSampler, SPV_C_GenericPointer, SPV_C_Image1D,
      SPV_C_ImageBuffer, SPV_C_NamedBarrier, SPV_C_GroupNonUniformVote,
      SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformBallot,
      SPV_C_GroupNonUniformShuffle, SPV_C_GroupNonUniformShuffleRelative,
      SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformQuad,
      SPV_C_StorageUniform16, SPV_C_UniformAndStorageBuffer8BitAccess,
      SPV_C_UniformTexelBufferArrayDynamicIndexing, SPV_C_Geometry,
      SPV_C_Tessellation, SPV_C_ImageReadWrite, SPV_C_ImageMipmap,
      SPV_C_AtomicStorage, SPV_C_ImageGatherExtended, SPV_C_StorageImageMultisample,
      SPV_C_UniformBufferArrayDynamicIndexing,
      SPV_C_SampledImageArrayDynamicIndexing,
      SPV_C_StorageBufferArrayDynamicIndexing,
      SPV_C_StorageImageArrayDynamicIndexing, SPV_C_ClipDistance, SPV_C_CullDistance,
      SPV_C_SampleRateShading, SPV_C_SampledRect, SPV_C_InputAttachment,
      SPV_C_SparseResidency, SPV_C_MinLod, SPV_C_SampledCubeArray,
      SPV_C_ImageMSArray, SPV_C_StorageImageExtendedFormats, SPV_C_ImageQuery,
      SPV_C_DerivativeControl, SPV_C_InterpolationFunction, SPV_C_TransformFeedback,
      SPV_C_StorageImageReadWithoutFormat, SPV_C_StorageImageWriteWithoutFormat,
      SPV_C_SubgroupDispatch, SPV_C_PipeStorage, SPV_C_DrawParameters,
      SPV_C_MultiView, SPV_C_VariablePointersStorageBuffer, SPV_C_Float16ImageAMD,
      SPV_C_ImageGatherBiasLodAMD, SPV_C_FragmentMaskAMD, SPV_C_StencilExportEXT,
      SPV_C_ImageReadWriteLodAMD, SPV_C_ShaderClockKHR,
      SPV_C_FragmentFullyCoveredEXT, SPV_C_MeshShadingNV, SPV_C_FragmentDensityEXT,
      SPV_C_ShaderNonUniform, SPV_C_RuntimeDescriptorArray,
      SPV_C_StorageTexelBufferArrayDynamicIndexing, SPV_C_RayTracingNV,
      SPV_C_PhysicalStorageBufferAddresses, SPV_C_CooperativeMatrixNV,
      SPV_C_FragmentShaderSampleInterlockEXT,
      SPV_C_FragmentShaderShadingRateInterlockEXT, SPV_C_ShaderSMBuiltinsNV,
      SPV_C_FragmentShaderPixelInterlockEXT, SPV_C_DemoteToHelperInvocationEXT,
      SPV_C_IntegerFunctions2INTEL, SPV_C_TessellationPointSize,
      SPV_C_GeometryPointSize, SPV_C_ImageCubeArray, SPV_C_ImageRect,
      SPV_C_GeometryStreams, SPV_C_MultiViewport, SPV_C_VariablePointers,
      SPV_C_SampleMaskOverrideCoverageNV, SPV_C_GeometryShaderPassthroughNV,
      SPV_C_PerViewAttributesNV, SPV_C_InputAttachmentArrayDynamicIndexing,
      SPV_C_UniformBufferArrayNonUniformIndexing,
      SPV_C_SampledImageArrayNonUniformIndexing,
      SPV_C_StorageBufferArrayNonUniformIndexing,
      SPV_C_StorageImageArrayNonUniformIndexing,
      SPV_C_InputAttachmentArrayNonUniformIndexing,
      SPV_C_UniformTexelBufferArrayNonUniformIndexing,
      SPV_C_StorageTexelBufferArrayNonUniformIndexing,
      SPV_C_ShaderViewportIndexLayerEXT, SPV_C_ShaderViewportMaskNV,
      SPV_C_ShaderStereoViewNV
    ]>;

def SPV_AM_Logical                 : I32EnumAttrCase<"Logical", 0>;
def SPV_AM_Physical32              : I32EnumAttrCase<"Physical32", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Addresses]>
  ];
}
def SPV_AM_Physical64              : I32EnumAttrCase<"Physical64", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Addresses]>
  ];
}
def SPV_AM_PhysicalStorageBuffer64 : I32EnumAttrCase<"PhysicalStorageBuffer64", 5348> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPV_C_PhysicalStorageBufferAddresses]>
  ];
}

def SPV_AddressingModelAttr :
    SPV_I32EnumAttr<"AddressingModel", "valid SPIR-V AddressingModel", [
      SPV_AM_Logical, SPV_AM_Physical32, SPV_AM_Physical64,
      SPV_AM_PhysicalStorageBuffer64
    ]>;

def SPV_BI_Position                    : I32EnumAttrCase<"Position", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_PointSize                   : I32EnumAttrCase<"PointSize", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_ClipDistance                : I32EnumAttrCase<"ClipDistance", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_ClipDistance]>
  ];
}
def SPV_BI_CullDistance                : I32EnumAttrCase<"CullDistance", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_CullDistance]>
  ];
}
def SPV_BI_VertexId                    : I32EnumAttrCase<"VertexId", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_InstanceId                  : I32EnumAttrCase<"InstanceId", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_PrimitiveId                 : I32EnumAttrCase<"PrimitiveId", 7> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_RayTracingNV, SPV_C_Tessellation]>
  ];
}
def SPV_BI_InvocationId                : I32EnumAttrCase<"InvocationId", 8> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_Tessellation]>
  ];
}
def SPV_BI_Layer                       : I32EnumAttrCase<"Layer", 9> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_ShaderLayer, SPV_C_ShaderViewportIndexLayerEXT]>
  ];
}
def SPV_BI_ViewportIndex               : I32EnumAttrCase<"ViewportIndex", 10> {
  list<Availability> availability = [
    Capability<[SPV_C_MultiViewport, SPV_C_ShaderViewportIndex, SPV_C_ShaderViewportIndexLayerEXT]>
  ];
}
def SPV_BI_TessLevelOuter              : I32EnumAttrCase<"TessLevelOuter", 11> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_BI_TessLevelInner              : I32EnumAttrCase<"TessLevelInner", 12> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_BI_TessCoord                   : I32EnumAttrCase<"TessCoord", 13> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_BI_PatchVertices               : I32EnumAttrCase<"PatchVertices", 14> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_BI_FragCoord                   : I32EnumAttrCase<"FragCoord", 15> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_PointCoord                  : I32EnumAttrCase<"PointCoord", 16> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_FrontFacing                 : I32EnumAttrCase<"FrontFacing", 17> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_SampleId                    : I32EnumAttrCase<"SampleId", 18> {
  list<Availability> availability = [
    Capability<[SPV_C_SampleRateShading]>
  ];
}
def SPV_BI_SamplePosition              : I32EnumAttrCase<"SamplePosition", 19> {
  list<Availability> availability = [
    Capability<[SPV_C_SampleRateShading]>
  ];
}
def SPV_BI_SampleMask                  : I32EnumAttrCase<"SampleMask", 20> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_FragDepth                   : I32EnumAttrCase<"FragDepth", 22> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_HelperInvocation            : I32EnumAttrCase<"HelperInvocation", 23> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_NumWorkgroups               : I32EnumAttrCase<"NumWorkgroups", 24>;
def SPV_BI_WorkgroupSize               : I32EnumAttrCase<"WorkgroupSize", 25>;
def SPV_BI_WorkgroupId                 : I32EnumAttrCase<"WorkgroupId", 26>;
def SPV_BI_LocalInvocationId           : I32EnumAttrCase<"LocalInvocationId", 27>;
def SPV_BI_GlobalInvocationId          : I32EnumAttrCase<"GlobalInvocationId", 28>;
def SPV_BI_LocalInvocationIndex        : I32EnumAttrCase<"LocalInvocationIndex", 29>;
def SPV_BI_WorkDim                     : I32EnumAttrCase<"WorkDim", 30> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_GlobalSize                  : I32EnumAttrCase<"GlobalSize", 31> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_EnqueuedWorkgroupSize       : I32EnumAttrCase<"EnqueuedWorkgroupSize", 32> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_GlobalOffset                : I32EnumAttrCase<"GlobalOffset", 33> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_GlobalLinearId              : I32EnumAttrCase<"GlobalLinearId", 34> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_SubgroupSize                : I32EnumAttrCase<"SubgroupSize", 36> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniform, SPV_C_Kernel, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_SubgroupMaxSize             : I32EnumAttrCase<"SubgroupMaxSize", 37> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_NumSubgroups                : I32EnumAttrCase<"NumSubgroups", 38> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniform, SPV_C_Kernel]>
  ];
}
def SPV_BI_NumEnqueuedSubgroups        : I32EnumAttrCase<"NumEnqueuedSubgroups", 39> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_BI_SubgroupId                  : I32EnumAttrCase<"SubgroupId", 40> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniform, SPV_C_Kernel]>
  ];
}
def SPV_BI_SubgroupLocalInvocationId   : I32EnumAttrCase<"SubgroupLocalInvocationId", 41> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniform, SPV_C_Kernel, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_VertexIndex                 : I32EnumAttrCase<"VertexIndex", 42> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_InstanceIndex               : I32EnumAttrCase<"InstanceIndex", 43> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_BI_SubgroupEqMask              : I32EnumAttrCase<"SubgroupEqMask", 4416> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformBallot, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_SubgroupGeMask              : I32EnumAttrCase<"SubgroupGeMask", 4417> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformBallot, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_SubgroupGtMask              : I32EnumAttrCase<"SubgroupGtMask", 4418> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformBallot, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_SubgroupLeMask              : I32EnumAttrCase<"SubgroupLeMask", 4419> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformBallot, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_SubgroupLtMask              : I32EnumAttrCase<"SubgroupLtMask", 4420> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformBallot, SPV_C_SubgroupBallotKHR]>
  ];
}
def SPV_BI_BaseVertex                  : I32EnumAttrCase<"BaseVertex", 4424> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>,
    Capability<[SPV_C_DrawParameters]>
  ];
}
def SPV_BI_BaseInstance                : I32EnumAttrCase<"BaseInstance", 4425> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>,
    Capability<[SPV_C_DrawParameters]>
  ];
}
def SPV_BI_DrawIndex                   : I32EnumAttrCase<"DrawIndex", 4426> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters, SPV_NV_mesh_shader]>,
    Capability<[SPV_C_DrawParameters, SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_DeviceIndex                 : I32EnumAttrCase<"DeviceIndex", 4438> {
  list<Availability> availability = [
    Extension<[SPV_KHR_device_group]>,
    Capability<[SPV_C_DeviceGroup]>
  ];
}
def SPV_BI_ViewIndex                   : I32EnumAttrCase<"ViewIndex", 4440> {
  list<Availability> availability = [
    Extension<[SPV_KHR_multiview]>,
    Capability<[SPV_C_MultiView]>
  ];
}
def SPV_BI_BaryCoordNoPerspAMD         : I32EnumAttrCase<"BaryCoordNoPerspAMD", 4992> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordNoPerspCentroidAMD : I32EnumAttrCase<"BaryCoordNoPerspCentroidAMD", 4993> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordNoPerspSampleAMD   : I32EnumAttrCase<"BaryCoordNoPerspSampleAMD", 4994> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordSmoothAMD          : I32EnumAttrCase<"BaryCoordSmoothAMD", 4995> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordSmoothCentroidAMD  : I32EnumAttrCase<"BaryCoordSmoothCentroidAMD", 4996> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordSmoothSampleAMD    : I32EnumAttrCase<"BaryCoordSmoothSampleAMD", 4997> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_BaryCoordPullModelAMD       : I32EnumAttrCase<"BaryCoordPullModelAMD", 4998> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_BI_FragStencilRefEXT           : I32EnumAttrCase<"FragStencilRefEXT", 5014> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>,
    Capability<[SPV_C_StencilExportEXT]>
  ];
}
def SPV_BI_ViewportMaskNV              : I32EnumAttrCase<"ViewportMaskNV", 5253> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader, SPV_NV_viewport_array2]>,
    Capability<[SPV_C_MeshShadingNV, SPV_C_ShaderViewportMaskNV]>
  ];
}
def SPV_BI_SecondaryPositionNV         : I32EnumAttrCase<"SecondaryPositionNV", 5257> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPV_C_ShaderStereoViewNV]>
  ];
}
def SPV_BI_SecondaryViewportMaskNV     : I32EnumAttrCase<"SecondaryViewportMaskNV", 5258> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPV_C_ShaderStereoViewNV]>
  ];
}
def SPV_BI_PositionPerViewNV           : I32EnumAttrCase<"PositionPerViewNV", 5261> {
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes, SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV, SPV_C_PerViewAttributesNV]>
  ];
}
def SPV_BI_ViewportMaskPerViewNV       : I32EnumAttrCase<"ViewportMaskPerViewNV", 5262> {
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes, SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV, SPV_C_PerViewAttributesNV]>
  ];
}
def SPV_BI_FullyCoveredEXT             : I32EnumAttrCase<"FullyCoveredEXT", 5264> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_fully_covered]>,
    Capability<[SPV_C_FragmentFullyCoveredEXT]>
  ];
}
def SPV_BI_TaskCountNV                 : I32EnumAttrCase<"TaskCountNV", 5274> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_PrimitiveCountNV            : I32EnumAttrCase<"PrimitiveCountNV", 5275> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_PrimitiveIndicesNV          : I32EnumAttrCase<"PrimitiveIndicesNV", 5276> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_ClipDistancePerViewNV       : I32EnumAttrCase<"ClipDistancePerViewNV", 5277> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_CullDistancePerViewNV       : I32EnumAttrCase<"CullDistancePerViewNV", 5278> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_LayerPerViewNV              : I32EnumAttrCase<"LayerPerViewNV", 5279> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_MeshViewCountNV             : I32EnumAttrCase<"MeshViewCountNV", 5280> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_MeshViewIndicesNV           : I32EnumAttrCase<"MeshViewIndicesNV", 5281> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_BI_BaryCoordNV                 : I32EnumAttrCase<"BaryCoordNV", 5286> {
  list<Availability> availability = [
    Extension<[SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPV_C_FragmentBarycentricNV]>
  ];
}
def SPV_BI_BaryCoordNoPerspNV          : I32EnumAttrCase<"BaryCoordNoPerspNV", 5287> {
  list<Availability> availability = [
    Extension<[SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPV_C_FragmentBarycentricNV]>
  ];
}
def SPV_BI_FragSizeEXT                 : I32EnumAttrCase<"FragSizeEXT", 5292> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>,
    Capability<[SPV_C_FragmentDensityEXT]>
  ];
}
def SPV_BI_FragInvocationCountEXT      : I32EnumAttrCase<"FragInvocationCountEXT", 5293> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>,
    Capability<[SPV_C_FragmentDensityEXT]>
  ];
}
def SPV_BI_LaunchIdNV                  : I32EnumAttrCase<"LaunchIdNV", 5319> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_LaunchSizeNV                : I32EnumAttrCase<"LaunchSizeNV", 5320> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_WorldRayOriginNV            : I32EnumAttrCase<"WorldRayOriginNV", 5321> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_WorldRayDirectionNV         : I32EnumAttrCase<"WorldRayDirectionNV", 5322> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_ObjectRayOriginNV           : I32EnumAttrCase<"ObjectRayOriginNV", 5323> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_ObjectRayDirectionNV        : I32EnumAttrCase<"ObjectRayDirectionNV", 5324> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_RayTminNV                   : I32EnumAttrCase<"RayTminNV", 5325> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_RayTmaxNV                   : I32EnumAttrCase<"RayTmaxNV", 5326> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_InstanceCustomIndexNV       : I32EnumAttrCase<"InstanceCustomIndexNV", 5327> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_ObjectToWorldNV             : I32EnumAttrCase<"ObjectToWorldNV", 5330> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_WorldToObjectNV             : I32EnumAttrCase<"WorldToObjectNV", 5331> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_HitTNV                      : I32EnumAttrCase<"HitTNV", 5332> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_HitKindNV                   : I32EnumAttrCase<"HitKindNV", 5333> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_IncomingRayFlagsNV          : I32EnumAttrCase<"IncomingRayFlagsNV", 5351> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_BI_WarpsPerSMNV                : I32EnumAttrCase<"WarpsPerSMNV", 5374> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPV_BI_SMCountNV                   : I32EnumAttrCase<"SMCountNV", 5375> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPV_BI_WarpIDNV                    : I32EnumAttrCase<"WarpIDNV", 5376> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPV_BI_SMIDNV                      : I32EnumAttrCase<"SMIDNV", 5377> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPV_C_ShaderSMBuiltinsNV]>
  ];
}

def SPV_BuiltInAttr :
    SPV_I32EnumAttr<"BuiltIn", "valid SPIR-V BuiltIn", [
      SPV_BI_Position, SPV_BI_PointSize, SPV_BI_ClipDistance, SPV_BI_CullDistance,
      SPV_BI_VertexId, SPV_BI_InstanceId, SPV_BI_PrimitiveId, SPV_BI_InvocationId,
      SPV_BI_Layer, SPV_BI_ViewportIndex, SPV_BI_TessLevelOuter,
      SPV_BI_TessLevelInner, SPV_BI_TessCoord, SPV_BI_PatchVertices,
      SPV_BI_FragCoord, SPV_BI_PointCoord, SPV_BI_FrontFacing, SPV_BI_SampleId,
      SPV_BI_SamplePosition, SPV_BI_SampleMask, SPV_BI_FragDepth,
      SPV_BI_HelperInvocation, SPV_BI_NumWorkgroups, SPV_BI_WorkgroupSize,
      SPV_BI_WorkgroupId, SPV_BI_LocalInvocationId, SPV_BI_GlobalInvocationId,
      SPV_BI_LocalInvocationIndex, SPV_BI_WorkDim, SPV_BI_GlobalSize,
      SPV_BI_EnqueuedWorkgroupSize, SPV_BI_GlobalOffset, SPV_BI_GlobalLinearId,
      SPV_BI_SubgroupSize, SPV_BI_SubgroupMaxSize, SPV_BI_NumSubgroups,
      SPV_BI_NumEnqueuedSubgroups, SPV_BI_SubgroupId,
      SPV_BI_SubgroupLocalInvocationId, SPV_BI_VertexIndex, SPV_BI_InstanceIndex,
      SPV_BI_SubgroupEqMask, SPV_BI_SubgroupGeMask, SPV_BI_SubgroupGtMask,
      SPV_BI_SubgroupLeMask, SPV_BI_SubgroupLtMask, SPV_BI_BaseVertex,
      SPV_BI_BaseInstance, SPV_BI_DrawIndex, SPV_BI_DeviceIndex, SPV_BI_ViewIndex,
      SPV_BI_BaryCoordNoPerspAMD, SPV_BI_BaryCoordNoPerspCentroidAMD,
      SPV_BI_BaryCoordNoPerspSampleAMD, SPV_BI_BaryCoordSmoothAMD,
      SPV_BI_BaryCoordSmoothCentroidAMD, SPV_BI_BaryCoordSmoothSampleAMD,
      SPV_BI_BaryCoordPullModelAMD, SPV_BI_FragStencilRefEXT, SPV_BI_ViewportMaskNV,
      SPV_BI_SecondaryPositionNV, SPV_BI_SecondaryViewportMaskNV,
      SPV_BI_PositionPerViewNV, SPV_BI_ViewportMaskPerViewNV, SPV_BI_FullyCoveredEXT,
      SPV_BI_TaskCountNV, SPV_BI_PrimitiveCountNV, SPV_BI_PrimitiveIndicesNV,
      SPV_BI_ClipDistancePerViewNV, SPV_BI_CullDistancePerViewNV,
      SPV_BI_LayerPerViewNV, SPV_BI_MeshViewCountNV, SPV_BI_MeshViewIndicesNV,
      SPV_BI_BaryCoordNV, SPV_BI_BaryCoordNoPerspNV, SPV_BI_FragSizeEXT,
      SPV_BI_FragInvocationCountEXT, SPV_BI_LaunchIdNV, SPV_BI_LaunchSizeNV,
      SPV_BI_WorldRayOriginNV, SPV_BI_WorldRayDirectionNV, SPV_BI_ObjectRayOriginNV,
      SPV_BI_ObjectRayDirectionNV, SPV_BI_RayTminNV, SPV_BI_RayTmaxNV,
      SPV_BI_InstanceCustomIndexNV, SPV_BI_ObjectToWorldNV, SPV_BI_WorldToObjectNV,
      SPV_BI_HitTNV, SPV_BI_HitKindNV, SPV_BI_IncomingRayFlagsNV,
      SPV_BI_WarpsPerSMNV, SPV_BI_SMCountNV, SPV_BI_WarpIDNV, SPV_BI_SMIDNV
    ]>;

def SPV_D_RelaxedPrecision            : I32EnumAttrCase<"RelaxedPrecision", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_SpecId                      : I32EnumAttrCase<"SpecId", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel, SPV_C_Shader]>
  ];
}
def SPV_D_Block                       : I32EnumAttrCase<"Block", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_BufferBlock                 : I32EnumAttrCase<"BufferBlock", 3> {
  list<Availability> availability = [
    MaxVersion<SPV_V_1_3>,
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_RowMajor                    : I32EnumAttrCase<"RowMajor", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_Matrix]>
  ];
}
def SPV_D_ColMajor                    : I32EnumAttrCase<"ColMajor", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_Matrix]>
  ];
}
def SPV_D_ArrayStride                 : I32EnumAttrCase<"ArrayStride", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_MatrixStride                : I32EnumAttrCase<"MatrixStride", 7> {
  list<Availability> availability = [
    Capability<[SPV_C_Matrix]>
  ];
}
def SPV_D_GLSLShared                  : I32EnumAttrCase<"GLSLShared", 8> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_GLSLPacked                  : I32EnumAttrCase<"GLSLPacked", 9> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_CPacked                     : I32EnumAttrCase<"CPacked", 10> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_BuiltIn                     : I32EnumAttrCase<"BuiltIn", 11>;
def SPV_D_NoPerspective               : I32EnumAttrCase<"NoPerspective", 13> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Flat                        : I32EnumAttrCase<"Flat", 14> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Patch                       : I32EnumAttrCase<"Patch", 15> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_D_Centroid                    : I32EnumAttrCase<"Centroid", 16> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Sample                      : I32EnumAttrCase<"Sample", 17> {
  list<Availability> availability = [
    Capability<[SPV_C_SampleRateShading]>
  ];
}
def SPV_D_Invariant                   : I32EnumAttrCase<"Invariant", 18> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Restrict                    : I32EnumAttrCase<"Restrict", 19>;
def SPV_D_Aliased                     : I32EnumAttrCase<"Aliased", 20>;
def SPV_D_Volatile                    : I32EnumAttrCase<"Volatile", 21>;
def SPV_D_Constant                    : I32EnumAttrCase<"Constant", 22> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_Coherent                    : I32EnumAttrCase<"Coherent", 23>;
def SPV_D_NonWritable                 : I32EnumAttrCase<"NonWritable", 24>;
def SPV_D_NonReadable                 : I32EnumAttrCase<"NonReadable", 25>;
def SPV_D_Uniform                     : I32EnumAttrCase<"Uniform", 26> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_UniformId                   : I32EnumAttrCase<"UniformId", 27> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>,
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_SaturatedConversion         : I32EnumAttrCase<"SaturatedConversion", 28> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_Stream                      : I32EnumAttrCase<"Stream", 29> {
  list<Availability> availability = [
    Capability<[SPV_C_GeometryStreams]>
  ];
}
def SPV_D_Location                    : I32EnumAttrCase<"Location", 30> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Component                   : I32EnumAttrCase<"Component", 31> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Index                       : I32EnumAttrCase<"Index", 32> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Binding                     : I32EnumAttrCase<"Binding", 33> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_DescriptorSet               : I32EnumAttrCase<"DescriptorSet", 34> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_Offset                      : I32EnumAttrCase<"Offset", 35> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_XfbBuffer                   : I32EnumAttrCase<"XfbBuffer", 36> {
  list<Availability> availability = [
    Capability<[SPV_C_TransformFeedback]>
  ];
}
def SPV_D_XfbStride                   : I32EnumAttrCase<"XfbStride", 37> {
  list<Availability> availability = [
    Capability<[SPV_C_TransformFeedback]>
  ];
}
def SPV_D_FuncParamAttr               : I32EnumAttrCase<"FuncParamAttr", 38> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_FPRoundingMode              : I32EnumAttrCase<"FPRoundingMode", 39>;
def SPV_D_FPFastMathMode              : I32EnumAttrCase<"FPFastMathMode", 40> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_LinkageAttributes           : I32EnumAttrCase<"LinkageAttributes", 41> {
  list<Availability> availability = [
    Capability<[SPV_C_Linkage]>
  ];
}
def SPV_D_NoContraction               : I32EnumAttrCase<"NoContraction", 42> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_D_InputAttachmentIndex        : I32EnumAttrCase<"InputAttachmentIndex", 43> {
  list<Availability> availability = [
    Capability<[SPV_C_InputAttachment]>
  ];
}
def SPV_D_Alignment                   : I32EnumAttrCase<"Alignment", 44> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_MaxByteOffset               : I32EnumAttrCase<"MaxByteOffset", 45> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>,
    Capability<[SPV_C_Addresses]>
  ];
}
def SPV_D_AlignmentId                 : I32EnumAttrCase<"AlignmentId", 46> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_2>,
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_D_MaxByteOffsetId             : I32EnumAttrCase<"MaxByteOffsetId", 47> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_2>,
    Capability<[SPV_C_Addresses]>
  ];
}
def SPV_D_NoSignedWrap                : I32EnumAttrCase<"NoSignedWrap", 4469> {
  list<Availability> availability = [
    Extension<[SPV_KHR_no_integer_wrap_decoration]>
  ];
}
def SPV_D_NoUnsignedWrap              : I32EnumAttrCase<"NoUnsignedWrap", 4470> {
  list<Availability> availability = [
    Extension<[SPV_KHR_no_integer_wrap_decoration]>
  ];
}
def SPV_D_ExplicitInterpAMD           : I32EnumAttrCase<"ExplicitInterpAMD", 4999> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPV_D_OverrideCoverageNV          : I32EnumAttrCase<"OverrideCoverageNV", 5248> {
  list<Availability> availability = [
    Extension<[SPV_NV_sample_mask_override_coverage]>,
    Capability<[SPV_C_SampleMaskOverrideCoverageNV]>
  ];
}
def SPV_D_PassthroughNV               : I32EnumAttrCase<"PassthroughNV", 5250> {
  list<Availability> availability = [
    Extension<[SPV_NV_geometry_shader_passthrough]>,
    Capability<[SPV_C_GeometryShaderPassthroughNV]>
  ];
}
def SPV_D_ViewportRelativeNV          : I32EnumAttrCase<"ViewportRelativeNV", 5252> {
  list<Availability> availability = [
    Capability<[SPV_C_ShaderViewportMaskNV]>
  ];
}
def SPV_D_SecondaryViewportRelativeNV : I32EnumAttrCase<"SecondaryViewportRelativeNV", 5256> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPV_C_ShaderStereoViewNV]>
  ];
}
def SPV_D_PerPrimitiveNV              : I32EnumAttrCase<"PerPrimitiveNV", 5271> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_D_PerViewNV                   : I32EnumAttrCase<"PerViewNV", 5272> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_D_PerTaskNV                   : I32EnumAttrCase<"PerTaskNV", 5273> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_D_PerVertexNV                 : I32EnumAttrCase<"PerVertexNV", 5285> {
  list<Availability> availability = [
    Extension<[SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPV_C_FragmentBarycentricNV]>
  ];
}
def SPV_D_NonUniform                  : I32EnumAttrCase<"NonUniform", 5300> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_ShaderNonUniform]>
  ];
}
def SPV_D_RestrictPointer             : I32EnumAttrCase<"RestrictPointer", 5355> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPV_C_PhysicalStorageBufferAddresses]>
  ];
}
def SPV_D_AliasedPointer              : I32EnumAttrCase<"AliasedPointer", 5356> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPV_C_PhysicalStorageBufferAddresses]>
  ];
}
def SPV_D_CounterBuffer               : I32EnumAttrCase<"CounterBuffer", 5634> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_D_UserSemantic                : I32EnumAttrCase<"UserSemantic", 5635> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_D_UserTypeGOOGLE              : I32EnumAttrCase<"UserTypeGOOGLE", 5636> {
  list<Availability> availability = [
    Extension<[SPV_GOOGLE_user_type]>
  ];
}

def SPV_DecorationAttr :
    SPV_I32EnumAttr<"Decoration", "valid SPIR-V Decoration", [
      SPV_D_RelaxedPrecision, SPV_D_SpecId, SPV_D_Block, SPV_D_BufferBlock,
      SPV_D_RowMajor, SPV_D_ColMajor, SPV_D_ArrayStride, SPV_D_MatrixStride,
      SPV_D_GLSLShared, SPV_D_GLSLPacked, SPV_D_CPacked, SPV_D_BuiltIn,
      SPV_D_NoPerspective, SPV_D_Flat, SPV_D_Patch, SPV_D_Centroid, SPV_D_Sample,
      SPV_D_Invariant, SPV_D_Restrict, SPV_D_Aliased, SPV_D_Volatile, SPV_D_Constant,
      SPV_D_Coherent, SPV_D_NonWritable, SPV_D_NonReadable, SPV_D_Uniform,
      SPV_D_UniformId, SPV_D_SaturatedConversion, SPV_D_Stream, SPV_D_Location,
      SPV_D_Component, SPV_D_Index, SPV_D_Binding, SPV_D_DescriptorSet, SPV_D_Offset,
      SPV_D_XfbBuffer, SPV_D_XfbStride, SPV_D_FuncParamAttr, SPV_D_FPRoundingMode,
      SPV_D_FPFastMathMode, SPV_D_LinkageAttributes, SPV_D_NoContraction,
      SPV_D_InputAttachmentIndex, SPV_D_Alignment, SPV_D_MaxByteOffset,
      SPV_D_AlignmentId, SPV_D_MaxByteOffsetId, SPV_D_NoSignedWrap,
      SPV_D_NoUnsignedWrap, SPV_D_ExplicitInterpAMD, SPV_D_OverrideCoverageNV,
      SPV_D_PassthroughNV, SPV_D_ViewportRelativeNV,
      SPV_D_SecondaryViewportRelativeNV, SPV_D_PerPrimitiveNV, SPV_D_PerViewNV,
      SPV_D_PerTaskNV, SPV_D_PerVertexNV, SPV_D_NonUniform, SPV_D_RestrictPointer,
      SPV_D_AliasedPointer, SPV_D_CounterBuffer, SPV_D_UserSemantic,
      SPV_D_UserTypeGOOGLE
    ]>;

def SPV_D_1D          : I32EnumAttrCase<"Dim1D", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Image1D, SPV_C_Sampled1D]>
  ];
}
def SPV_D_2D          : I32EnumAttrCase<"Dim2D", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_ImageMSArray, SPV_C_Kernel, SPV_C_Shader]>
  ];
}
def SPV_D_3D          : I32EnumAttrCase<"Dim3D", 2>;
def SPV_D_Cube        : I32EnumAttrCase<"Cube", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_ImageCubeArray, SPV_C_Shader]>
  ];
}
def SPV_D_Rect        : I32EnumAttrCase<"Rect", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_ImageRect, SPV_C_SampledRect]>
  ];
}
def SPV_D_Buffer      : I32EnumAttrCase<"Buffer", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_ImageBuffer, SPV_C_SampledBuffer]>
  ];
}
def SPV_D_SubpassData : I32EnumAttrCase<"SubpassData", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_InputAttachment]>
  ];
}

def SPV_DimAttr :
    SPV_I32EnumAttr<"Dim", "valid SPIR-V Dim", [
      SPV_D_1D, SPV_D_2D, SPV_D_3D, SPV_D_Cube, SPV_D_Rect, SPV_D_Buffer,
      SPV_D_SubpassData
    ]>;

def SPV_EM_Invocations                      : I32EnumAttrCase<"Invocations", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_SpacingEqual                     : I32EnumAttrCase<"SpacingEqual", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_SpacingFractionalEven            : I32EnumAttrCase<"SpacingFractionalEven", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_SpacingFractionalOdd             : I32EnumAttrCase<"SpacingFractionalOdd", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_VertexOrderCw                    : I32EnumAttrCase<"VertexOrderCw", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_VertexOrderCcw                   : I32EnumAttrCase<"VertexOrderCcw", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_PixelCenterInteger               : I32EnumAttrCase<"PixelCenterInteger", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_OriginUpperLeft                  : I32EnumAttrCase<"OriginUpperLeft", 7> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_OriginLowerLeft                  : I32EnumAttrCase<"OriginLowerLeft", 8> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_EarlyFragmentTests               : I32EnumAttrCase<"EarlyFragmentTests", 9> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_PointMode                        : I32EnumAttrCase<"PointMode", 10> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_Xfb                              : I32EnumAttrCase<"Xfb", 11> {
  list<Availability> availability = [
    Capability<[SPV_C_TransformFeedback]>
  ];
}
def SPV_EM_DepthReplacing                   : I32EnumAttrCase<"DepthReplacing", 12> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_DepthGreater                     : I32EnumAttrCase<"DepthGreater", 14> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_DepthLess                        : I32EnumAttrCase<"DepthLess", 15> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_DepthUnchanged                   : I32EnumAttrCase<"DepthUnchanged", 16> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_LocalSize                        : I32EnumAttrCase<"LocalSize", 17>;
def SPV_EM_LocalSizeHint                    : I32EnumAttrCase<"LocalSizeHint", 18> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_InputPoints                      : I32EnumAttrCase<"InputPoints", 19> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_InputLines                       : I32EnumAttrCase<"InputLines", 20> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_InputLinesAdjacency              : I32EnumAttrCase<"InputLinesAdjacency", 21> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_Triangles                        : I32EnumAttrCase<"Triangles", 22> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_Tessellation]>
  ];
}
def SPV_EM_InputTrianglesAdjacency          : I32EnumAttrCase<"InputTrianglesAdjacency", 23> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_Quads                            : I32EnumAttrCase<"Quads", 24> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_Isolines                         : I32EnumAttrCase<"Isolines", 25> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_OutputVertices                   : I32EnumAttrCase<"OutputVertices", 26> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_MeshShadingNV, SPV_C_Tessellation]>
  ];
}
def SPV_EM_OutputPoints                     : I32EnumAttrCase<"OutputPoints", 27> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry, SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_OutputLineStrip                  : I32EnumAttrCase<"OutputLineStrip", 28> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_OutputTriangleStrip              : I32EnumAttrCase<"OutputTriangleStrip", 29> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_VecTypeHint                      : I32EnumAttrCase<"VecTypeHint", 30> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_ContractionOff                   : I32EnumAttrCase<"ContractionOff", 31> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_Initializer                      : I32EnumAttrCase<"Initializer", 33> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>,
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_Finalizer                        : I32EnumAttrCase<"Finalizer", 34> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>,
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_SubgroupSize                     : I32EnumAttrCase<"SubgroupSize", 35> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>,
    Capability<[SPV_C_SubgroupDispatch]>
  ];
}
def SPV_EM_SubgroupsPerWorkgroup            : I32EnumAttrCase<"SubgroupsPerWorkgroup", 36> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>,
    Capability<[SPV_C_SubgroupDispatch]>
  ];
}
def SPV_EM_SubgroupsPerWorkgroupId          : I32EnumAttrCase<"SubgroupsPerWorkgroupId", 37> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_2>,
    Capability<[SPV_C_SubgroupDispatch]>
  ];
}
def SPV_EM_LocalSizeId                      : I32EnumAttrCase<"LocalSizeId", 38> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_2>
  ];
}
def SPV_EM_LocalSizeHintId                  : I32EnumAttrCase<"LocalSizeHintId", 39> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_2>,
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_PostDepthCoverage                : I32EnumAttrCase<"PostDepthCoverage", 4446> {
  list<Availability> availability = [
    Extension<[SPV_KHR_post_depth_coverage]>,
    Capability<[SPV_C_SampleMaskPostDepthCoverage]>
  ];
}
def SPV_EM_DenormPreserve                   : I32EnumAttrCase<"DenormPreserve", 4459> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPV_C_DenormPreserve]>
  ];
}
def SPV_EM_DenormFlushToZero                : I32EnumAttrCase<"DenormFlushToZero", 4460> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPV_C_DenormFlushToZero]>
  ];
}
def SPV_EM_SignedZeroInfNanPreserve         : I32EnumAttrCase<"SignedZeroInfNanPreserve", 4461> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPV_C_SignedZeroInfNanPreserve]>
  ];
}
def SPV_EM_RoundingModeRTE                  : I32EnumAttrCase<"RoundingModeRTE", 4462> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPV_C_RoundingModeRTE]>
  ];
}
def SPV_EM_RoundingModeRTZ                  : I32EnumAttrCase<"RoundingModeRTZ", 4463> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPV_C_RoundingModeRTZ]>
  ];
}
def SPV_EM_StencilRefReplacingEXT           : I32EnumAttrCase<"StencilRefReplacingEXT", 5027> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>,
    Capability<[SPV_C_StencilExportEXT]>
  ];
}
def SPV_EM_OutputLinesNV                    : I32EnumAttrCase<"OutputLinesNV", 5269> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_OutputPrimitivesNV               : I32EnumAttrCase<"OutputPrimitivesNV", 5270> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_DerivativeGroupQuadsNV           : I32EnumAttrCase<"DerivativeGroupQuadsNV", 5289> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>,
    Capability<[SPV_C_ComputeDerivativeGroupQuadsNV]>
  ];
}
def SPV_EM_DerivativeGroupLinearNV          : I32EnumAttrCase<"DerivativeGroupLinearNV", 5290> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>,
    Capability<[SPV_C_ComputeDerivativeGroupLinearNV]>
  ];
}
def SPV_EM_OutputTrianglesNV                : I32EnumAttrCase<"OutputTrianglesNV", 5298> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_PixelInterlockOrderedEXT         : I32EnumAttrCase<"PixelInterlockOrderedEXT", 5366> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderPixelInterlockEXT]>
  ];
}
def SPV_EM_PixelInterlockUnorderedEXT       : I32EnumAttrCase<"PixelInterlockUnorderedEXT", 5367> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderPixelInterlockEXT]>
  ];
}
def SPV_EM_SampleInterlockOrderedEXT        : I32EnumAttrCase<"SampleInterlockOrderedEXT", 5368> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderSampleInterlockEXT]>
  ];
}
def SPV_EM_SampleInterlockUnorderedEXT      : I32EnumAttrCase<"SampleInterlockUnorderedEXT", 5369> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderSampleInterlockEXT]>
  ];
}
def SPV_EM_ShadingRateInterlockOrderedEXT   : I32EnumAttrCase<"ShadingRateInterlockOrderedEXT", 5370> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderShadingRateInterlockEXT]>
  ];
}
def SPV_EM_ShadingRateInterlockUnorderedEXT : I32EnumAttrCase<"ShadingRateInterlockUnorderedEXT", 5371> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPV_C_FragmentShaderShadingRateInterlockEXT]>
  ];
}

def SPV_ExecutionModeAttr :
    SPV_I32EnumAttr<"ExecutionMode", "valid SPIR-V ExecutionMode", [
      SPV_EM_Invocations, SPV_EM_SpacingEqual, SPV_EM_SpacingFractionalEven,
      SPV_EM_SpacingFractionalOdd, SPV_EM_VertexOrderCw, SPV_EM_VertexOrderCcw,
      SPV_EM_PixelCenterInteger, SPV_EM_OriginUpperLeft, SPV_EM_OriginLowerLeft,
      SPV_EM_EarlyFragmentTests, SPV_EM_PointMode, SPV_EM_Xfb, SPV_EM_DepthReplacing,
      SPV_EM_DepthGreater, SPV_EM_DepthLess, SPV_EM_DepthUnchanged, SPV_EM_LocalSize,
      SPV_EM_LocalSizeHint, SPV_EM_InputPoints, SPV_EM_InputLines,
      SPV_EM_InputLinesAdjacency, SPV_EM_Triangles, SPV_EM_InputTrianglesAdjacency,
      SPV_EM_Quads, SPV_EM_Isolines, SPV_EM_OutputVertices, SPV_EM_OutputPoints,
      SPV_EM_OutputLineStrip, SPV_EM_OutputTriangleStrip, SPV_EM_VecTypeHint,
      SPV_EM_ContractionOff, SPV_EM_Initializer, SPV_EM_Finalizer,
      SPV_EM_SubgroupSize, SPV_EM_SubgroupsPerWorkgroup,
      SPV_EM_SubgroupsPerWorkgroupId, SPV_EM_LocalSizeId, SPV_EM_LocalSizeHintId,
      SPV_EM_PostDepthCoverage, SPV_EM_DenormPreserve, SPV_EM_DenormFlushToZero,
      SPV_EM_SignedZeroInfNanPreserve, SPV_EM_RoundingModeRTE,
      SPV_EM_RoundingModeRTZ, SPV_EM_StencilRefReplacingEXT, SPV_EM_OutputLinesNV,
      SPV_EM_OutputPrimitivesNV, SPV_EM_DerivativeGroupQuadsNV,
      SPV_EM_DerivativeGroupLinearNV, SPV_EM_OutputTrianglesNV,
      SPV_EM_PixelInterlockOrderedEXT, SPV_EM_PixelInterlockUnorderedEXT,
      SPV_EM_SampleInterlockOrderedEXT, SPV_EM_SampleInterlockUnorderedEXT,
      SPV_EM_ShadingRateInterlockOrderedEXT, SPV_EM_ShadingRateInterlockUnorderedEXT
    ]>;

def SPV_EM_Vertex                 : I32EnumAttrCase<"Vertex", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_TessellationControl    : I32EnumAttrCase<"TessellationControl", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_TessellationEvaluation : I32EnumAttrCase<"TessellationEvaluation", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Tessellation]>
  ];
}
def SPV_EM_Geometry               : I32EnumAttrCase<"Geometry", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_Geometry]>
  ];
}
def SPV_EM_Fragment               : I32EnumAttrCase<"Fragment", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_GLCompute              : I32EnumAttrCase<"GLCompute", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_EM_Kernel                 : I32EnumAttrCase<"Kernel", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_EM_TaskNV                 : I32EnumAttrCase<"TaskNV", 5267> {
  list<Availability> availability = [
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_MeshNV                 : I32EnumAttrCase<"MeshNV", 5268> {
  list<Availability> availability = [
    Capability<[SPV_C_MeshShadingNV]>
  ];
}
def SPV_EM_RayGenerationNV        : I32EnumAttrCase<"RayGenerationNV", 5313> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_EM_IntersectionNV         : I32EnumAttrCase<"IntersectionNV", 5314> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_EM_AnyHitNV               : I32EnumAttrCase<"AnyHitNV", 5315> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_EM_ClosestHitNV           : I32EnumAttrCase<"ClosestHitNV", 5316> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_EM_MissNV                 : I32EnumAttrCase<"MissNV", 5317> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_EM_CallableNV             : I32EnumAttrCase<"CallableNV", 5318> {
  list<Availability> availability = [
    Capability<[SPV_C_RayTracingNV]>
  ];
}

def SPV_ExecutionModelAttr :
    SPV_I32EnumAttr<"ExecutionModel", "valid SPIR-V ExecutionModel", [
      SPV_EM_Vertex, SPV_EM_TessellationControl, SPV_EM_TessellationEvaluation,
      SPV_EM_Geometry, SPV_EM_Fragment, SPV_EM_GLCompute, SPV_EM_Kernel,
      SPV_EM_TaskNV, SPV_EM_MeshNV, SPV_EM_RayGenerationNV, SPV_EM_IntersectionNV,
      SPV_EM_AnyHitNV, SPV_EM_ClosestHitNV, SPV_EM_MissNV, SPV_EM_CallableNV
    ]>;

def SPV_FC_None       : BitEnumAttrCase<"None", 0x0000>;
def SPV_FC_Inline     : BitEnumAttrCase<"Inline", 0x0001>;
def SPV_FC_DontInline : BitEnumAttrCase<"DontInline", 0x0002>;
def SPV_FC_Pure       : BitEnumAttrCase<"Pure", 0x0004>;
def SPV_FC_Const      : BitEnumAttrCase<"Const", 0x0008>;

def SPV_FunctionControlAttr :
    SPV_BitEnumAttr<"FunctionControl", "valid SPIR-V FunctionControl", [
      SPV_FC_None, SPV_FC_Inline, SPV_FC_DontInline, SPV_FC_Pure, SPV_FC_Const
    ]>;

def SPV_GO_Reduce                     : I32EnumAttrCase<"Reduce", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformBallot, SPV_C_Kernel]>
  ];
}
def SPV_GO_InclusiveScan              : I32EnumAttrCase<"InclusiveScan", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformBallot, SPV_C_Kernel]>
  ];
}
def SPV_GO_ExclusiveScan              : I32EnumAttrCase<"ExclusiveScan", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformBallot, SPV_C_Kernel]>
  ];
}
def SPV_GO_ClusteredReduce            : I32EnumAttrCase<"ClusteredReduce", 3> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_3>,
    Capability<[SPV_C_GroupNonUniformClustered]>
  ];
}
def SPV_GO_PartitionedReduceNV        : I32EnumAttrCase<"PartitionedReduceNV", 6> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPV_C_GroupNonUniformPartitionedNV]>
  ];
}
def SPV_GO_PartitionedInclusiveScanNV : I32EnumAttrCase<"PartitionedInclusiveScanNV", 7> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPV_C_GroupNonUniformPartitionedNV]>
  ];
}
def SPV_GO_PartitionedExclusiveScanNV : I32EnumAttrCase<"PartitionedExclusiveScanNV", 8> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

def SPV_GroupOperationAttr :
    SPV_I32EnumAttr<"GroupOperation", "valid SPIR-V GroupOperation", [
      SPV_GO_Reduce, SPV_GO_InclusiveScan, SPV_GO_ExclusiveScan,
      SPV_GO_ClusteredReduce, SPV_GO_PartitionedReduceNV,
      SPV_GO_PartitionedInclusiveScanNV, SPV_GO_PartitionedExclusiveScanNV
    ]>;

def SPV_IF_Unknown      : I32EnumAttrCase<"Unknown", 0>;
def SPV_IF_Rgba32f      : I32EnumAttrCase<"Rgba32f", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba16f      : I32EnumAttrCase<"Rgba16f", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_R32f         : I32EnumAttrCase<"R32f", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba8        : I32EnumAttrCase<"Rgba8", 4> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba8Snorm   : I32EnumAttrCase<"Rgba8Snorm", 5> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rg32f        : I32EnumAttrCase<"Rg32f", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg16f        : I32EnumAttrCase<"Rg16f", 7> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R11fG11fB10f : I32EnumAttrCase<"R11fG11fB10f", 8> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R16f         : I32EnumAttrCase<"R16f", 9> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rgba16       : I32EnumAttrCase<"Rgba16", 10> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rgb10A2      : I32EnumAttrCase<"Rgb10A2", 11> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg16         : I32EnumAttrCase<"Rg16", 12> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg8          : I32EnumAttrCase<"Rg8", 13> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R16          : I32EnumAttrCase<"R16", 14> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R8           : I32EnumAttrCase<"R8", 15> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rgba16Snorm  : I32EnumAttrCase<"Rgba16Snorm", 16> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg16Snorm    : I32EnumAttrCase<"Rg16Snorm", 17> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg8Snorm     : I32EnumAttrCase<"Rg8Snorm", 18> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R16Snorm     : I32EnumAttrCase<"R16Snorm", 19> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R8Snorm      : I32EnumAttrCase<"R8Snorm", 20> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rgba32i      : I32EnumAttrCase<"Rgba32i", 21> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba16i      : I32EnumAttrCase<"Rgba16i", 22> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba8i       : I32EnumAttrCase<"Rgba8i", 23> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_R32i         : I32EnumAttrCase<"R32i", 24> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rg32i        : I32EnumAttrCase<"Rg32i", 25> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg16i        : I32EnumAttrCase<"Rg16i", 26> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg8i         : I32EnumAttrCase<"Rg8i", 27> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R16i         : I32EnumAttrCase<"R16i", 28> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R8i          : I32EnumAttrCase<"R8i", 29> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rgba32ui     : I32EnumAttrCase<"Rgba32ui", 30> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba16ui     : I32EnumAttrCase<"Rgba16ui", 31> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgba8ui      : I32EnumAttrCase<"Rgba8ui", 32> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_R32ui        : I32EnumAttrCase<"R32ui", 33> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_IF_Rgb10a2ui    : I32EnumAttrCase<"Rgb10a2ui", 34> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg32ui       : I32EnumAttrCase<"Rg32ui", 35> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg16ui       : I32EnumAttrCase<"Rg16ui", 36> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_Rg8ui        : I32EnumAttrCase<"Rg8ui", 37> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R16ui        : I32EnumAttrCase<"R16ui", 38> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}
def SPV_IF_R8ui         : I32EnumAttrCase<"R8ui", 39> {
  list<Availability> availability = [
    Capability<[SPV_C_StorageImageExtendedFormats]>
  ];
}

def SPV_ImageFormatAttr :
    SPV_I32EnumAttr<"ImageFormat", "valid SPIR-V ImageFormat", [
      SPV_IF_Unknown, SPV_IF_Rgba32f, SPV_IF_Rgba16f, SPV_IF_R32f, SPV_IF_Rgba8,
      SPV_IF_Rgba8Snorm, SPV_IF_Rg32f, SPV_IF_Rg16f, SPV_IF_R11fG11fB10f,
      SPV_IF_R16f, SPV_IF_Rgba16, SPV_IF_Rgb10A2, SPV_IF_Rg16, SPV_IF_Rg8,
      SPV_IF_R16, SPV_IF_R8, SPV_IF_Rgba16Snorm, SPV_IF_Rg16Snorm, SPV_IF_Rg8Snorm,
      SPV_IF_R16Snorm, SPV_IF_R8Snorm, SPV_IF_Rgba32i, SPV_IF_Rgba16i, SPV_IF_Rgba8i,
      SPV_IF_R32i, SPV_IF_Rg32i, SPV_IF_Rg16i, SPV_IF_Rg8i, SPV_IF_R16i, SPV_IF_R8i,
      SPV_IF_Rgba32ui, SPV_IF_Rgba16ui, SPV_IF_Rgba8ui, SPV_IF_R32ui,
      SPV_IF_Rgb10a2ui, SPV_IF_Rg32ui, SPV_IF_Rg16ui, SPV_IF_Rg8ui, SPV_IF_R16ui,
      SPV_IF_R8ui
    ]>;

def SPV_LT_Export : I32EnumAttrCase<"Export", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Linkage]>
  ];
}
def SPV_LT_Import : I32EnumAttrCase<"Import", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Linkage]>
  ];
}

def SPV_LinkageTypeAttr :
    SPV_I32EnumAttr<"LinkageType", "valid SPIR-V LinkageType", [
      SPV_LT_Export, SPV_LT_Import
    ]>;

def SPV_LC_None               : BitEnumAttrCase<"None", 0x0000>;
def SPV_LC_Unroll             : BitEnumAttrCase<"Unroll", 0x0001>;
def SPV_LC_DontUnroll         : BitEnumAttrCase<"DontUnroll", 0x0002>;
def SPV_LC_DependencyInfinite : BitEnumAttrCase<"DependencyInfinite", 0x0004> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>
  ];
}
def SPV_LC_DependencyLength   : BitEnumAttrCase<"DependencyLength", 0x0008> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_1>
  ];
}
def SPV_LC_MinIterations      : BitEnumAttrCase<"MinIterations", 0x0010> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_LC_MaxIterations      : BitEnumAttrCase<"MaxIterations", 0x0020> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_LC_IterationMultiple  : BitEnumAttrCase<"IterationMultiple", 0x0040> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_LC_PeelCount          : BitEnumAttrCase<"PeelCount", 0x0080> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}
def SPV_LC_PartialCount       : BitEnumAttrCase<"PartialCount", 0x0100> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_4>
  ];
}

def SPV_LoopControlAttr :
    SPV_BitEnumAttr<"LoopControl", "valid SPIR-V LoopControl", [
      SPV_LC_None, SPV_LC_Unroll, SPV_LC_DontUnroll, SPV_LC_DependencyInfinite,
      SPV_LC_DependencyLength, SPV_LC_MinIterations, SPV_LC_MaxIterations,
      SPV_LC_IterationMultiple, SPV_LC_PeelCount, SPV_LC_PartialCount
    ]>;

def SPV_MA_None                 : BitEnumAttrCase<"None", 0x0000>;
def SPV_MA_Volatile             : BitEnumAttrCase<"Volatile", 0x0001>;
def SPV_MA_Aligned              : BitEnumAttrCase<"Aligned", 0x0002>;
def SPV_MA_Nontemporal          : BitEnumAttrCase<"Nontemporal", 0x0004>;
def SPV_MA_MakePointerAvailable : BitEnumAttrCase<"MakePointerAvailable", 0x0008> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}
def SPV_MA_MakePointerVisible   : BitEnumAttrCase<"MakePointerVisible", 0x0010> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}
def SPV_MA_NonPrivatePointer    : BitEnumAttrCase<"NonPrivatePointer", 0x0020> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}

def SPV_MemoryAccessAttr :
    SPV_BitEnumAttr<"MemoryAccess", "valid SPIR-V MemoryAccess", [
      SPV_MA_None, SPV_MA_Volatile, SPV_MA_Aligned, SPV_MA_Nontemporal,
      SPV_MA_MakePointerAvailable, SPV_MA_MakePointerVisible,
      SPV_MA_NonPrivatePointer
    ]>;

def SPV_MM_Simple  : I32EnumAttrCase<"Simple", 0> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_MM_GLSL450 : I32EnumAttrCase<"GLSL450", 1> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_MM_OpenCL  : I32EnumAttrCase<"OpenCL", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Kernel]>
  ];
}
def SPV_MM_Vulkan  : I32EnumAttrCase<"Vulkan", 3> {
  list<Availability> availability = [
    Extension<[SPV_KHR_vulkan_memory_model]>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}

def SPV_MemoryModelAttr :
    SPV_I32EnumAttr<"MemoryModel", "valid SPIR-V MemoryModel", [
      SPV_MM_Simple, SPV_MM_GLSL450, SPV_MM_OpenCL, SPV_MM_Vulkan
    ]>;

def SPV_MS_None                   : BitEnumAttrCase<"None", 0x0000>;
def SPV_MS_Acquire                : BitEnumAttrCase<"Acquire", 0x0002>;
def SPV_MS_Release                : BitEnumAttrCase<"Release", 0x0004>;
def SPV_MS_AcquireRelease         : BitEnumAttrCase<"AcquireRelease", 0x0008>;
def SPV_MS_SequentiallyConsistent : BitEnumAttrCase<"SequentiallyConsistent", 0x0010>;
def SPV_MS_UniformMemory          : BitEnumAttrCase<"UniformMemory", 0x0040> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_MS_SubgroupMemory         : BitEnumAttrCase<"SubgroupMemory", 0x0080>;
def SPV_MS_WorkgroupMemory        : BitEnumAttrCase<"WorkgroupMemory", 0x0100>;
def SPV_MS_CrossWorkgroupMemory   : BitEnumAttrCase<"CrossWorkgroupMemory", 0x0200>;
def SPV_MS_AtomicCounterMemory    : BitEnumAttrCase<"AtomicCounterMemory", 0x0400> {
  list<Availability> availability = [
    Capability<[SPV_C_AtomicStorage]>
  ];
}
def SPV_MS_ImageMemory            : BitEnumAttrCase<"ImageMemory", 0x0800>;
def SPV_MS_OutputMemory           : BitEnumAttrCase<"OutputMemory", 0x1000> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}
def SPV_MS_MakeAvailable          : BitEnumAttrCase<"MakeAvailable", 0x2000> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}
def SPV_MS_MakeVisible            : BitEnumAttrCase<"MakeVisible", 0x4000> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}
def SPV_MS_Volatile               : BitEnumAttrCase<"Volatile", 0x8000> {
  list<Availability> availability = [
    Extension<[SPV_KHR_vulkan_memory_model]>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}

def SPV_MemorySemanticsAttr :
    SPV_BitEnumAttr<"MemorySemantics", "valid SPIR-V MemorySemantics", [
      SPV_MS_None, SPV_MS_Acquire, SPV_MS_Release, SPV_MS_AcquireRelease,
      SPV_MS_SequentiallyConsistent, SPV_MS_UniformMemory, SPV_MS_SubgroupMemory,
      SPV_MS_WorkgroupMemory, SPV_MS_CrossWorkgroupMemory,
      SPV_MS_AtomicCounterMemory, SPV_MS_ImageMemory, SPV_MS_OutputMemory,
      SPV_MS_MakeAvailable, SPV_MS_MakeVisible, SPV_MS_Volatile
    ]>;

def SPV_S_CrossDevice : I32EnumAttrCase<"CrossDevice", 0>;
def SPV_S_Device      : I32EnumAttrCase<"Device", 1>;
def SPV_S_Workgroup   : I32EnumAttrCase<"Workgroup", 2>;
def SPV_S_Subgroup    : I32EnumAttrCase<"Subgroup", 3>;
def SPV_S_Invocation  : I32EnumAttrCase<"Invocation", 4>;
def SPV_S_QueueFamily : I32EnumAttrCase<"QueueFamily", 5> {
  list<Availability> availability = [
    MinVersion<SPV_V_1_5>,
    Capability<[SPV_C_VulkanMemoryModel]>
  ];
}

def SPV_ScopeAttr :
    SPV_I32EnumAttr<"Scope", "valid SPIR-V Scope", [
      SPV_S_CrossDevice, SPV_S_Device, SPV_S_Workgroup, SPV_S_Subgroup,
      SPV_S_Invocation, SPV_S_QueueFamily
    ]>;

def SPV_SC_None        : BitEnumAttrCase<"None", 0x0000>;
def SPV_SC_Flatten     : BitEnumAttrCase<"Flatten", 0x0001>;
def SPV_SC_DontFlatten : BitEnumAttrCase<"DontFlatten", 0x0002>;

def SPV_SelectionControlAttr :
    SPV_BitEnumAttr<"SelectionControl", "valid SPIR-V SelectionControl", [
      SPV_SC_None, SPV_SC_Flatten, SPV_SC_DontFlatten
    ]>;

def SPV_SC_UniformConstant        : I32EnumAttrCase<"UniformConstant", 0>;
def SPV_SC_Input                  : I32EnumAttrCase<"Input", 1>;
def SPV_SC_Uniform                : I32EnumAttrCase<"Uniform", 2> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_SC_Output                 : I32EnumAttrCase<"Output", 3> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_SC_Workgroup              : I32EnumAttrCase<"Workgroup", 4>;
def SPV_SC_CrossWorkgroup         : I32EnumAttrCase<"CrossWorkgroup", 5>;
def SPV_SC_Private                : I32EnumAttrCase<"Private", 6> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_SC_Function               : I32EnumAttrCase<"Function", 7>;
def SPV_SC_Generic                : I32EnumAttrCase<"Generic", 8> {
  list<Availability> availability = [
    Capability<[SPV_C_GenericPointer]>
  ];
}
def SPV_SC_PushConstant           : I32EnumAttrCase<"PushConstant", 9> {
  list<Availability> availability = [
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_SC_AtomicCounter          : I32EnumAttrCase<"AtomicCounter", 10> {
  list<Availability> availability = [
    Capability<[SPV_C_AtomicStorage]>
  ];
}
def SPV_SC_Image                  : I32EnumAttrCase<"Image", 11>;
def SPV_SC_StorageBuffer          : I32EnumAttrCase<"StorageBuffer", 12> {
  list<Availability> availability = [
    Extension<[SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    Capability<[SPV_C_Shader]>
  ];
}
def SPV_SC_CallableDataNV         : I32EnumAttrCase<"CallableDataNV", 5328> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_IncomingCallableDataNV : I32EnumAttrCase<"IncomingCallableDataNV", 5329> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_RayPayloadNV           : I32EnumAttrCase<"RayPayloadNV", 5338> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_HitAttributeNV         : I32EnumAttrCase<"HitAttributeNV", 5339> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_IncomingRayPayloadNV   : I32EnumAttrCase<"IncomingRayPayloadNV", 5342> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_ShaderRecordBufferNV   : I32EnumAttrCase<"ShaderRecordBufferNV", 5343> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPV_C_RayTracingNV]>
  ];
}
def SPV_SC_PhysicalStorageBuffer  : I32EnumAttrCase<"PhysicalStorageBuffer", 5349> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPV_C_PhysicalStorageBufferAddresses]>
  ];
}

def SPV_StorageClassAttr :
    SPV_I32EnumAttr<"StorageClass", "valid SPIR-V StorageClass", [
      SPV_SC_UniformConstant, SPV_SC_Input, SPV_SC_Uniform, SPV_SC_Output,
      SPV_SC_Workgroup, SPV_SC_CrossWorkgroup, SPV_SC_Private, SPV_SC_Function,
      SPV_SC_Generic, SPV_SC_PushConstant, SPV_SC_AtomicCounter, SPV_SC_Image,
      SPV_SC_StorageBuffer, SPV_SC_CallableDataNV, SPV_SC_IncomingCallableDataNV,
      SPV_SC_RayPayloadNV, SPV_SC_HitAttributeNV, SPV_SC_IncomingRayPayloadNV,
      SPV_SC_ShaderRecordBufferNV, SPV_SC_PhysicalStorageBuffer
    ]>;

// End enum section. Generated from SPIR-V spec; DO NOT MODIFY!

// Enums added manually that are not part of SPIR-V spec

def SPV_IDI_NoDepth      : I32EnumAttrCase<"NoDepth", 0>;
def SPV_IDI_IsDepth      : I32EnumAttrCase<"IsDepth", 1>;
def SPV_IDI_DepthUnknown : I32EnumAttrCase<"DepthUnknown", 2>;

def SPV_DepthAttr :
    SPV_I32EnumAttr<"ImageDepthInfo", "valid SPIR-V Image Depth specification",
      [SPV_IDI_NoDepth, SPV_IDI_IsDepth, SPV_IDI_DepthUnknown]>;

def SPV_IAI_NonArrayed : I32EnumAttrCase<"NonArrayed", 0>;
def SPV_IAI_Arrayed    : I32EnumAttrCase<"Arrayed", 1>;

def SPV_ArrayedAttr :
    SPV_I32EnumAttr<
      "ImageArrayedInfo", "valid SPIR-V Image Arrayed specification",
      [SPV_IAI_NonArrayed, SPV_IAI_Arrayed]>;

def SPV_ISI_SingleSampled : I32EnumAttrCase<"SingleSampled", 0>;
def SPV_ISI_MultiSampled  : I32EnumAttrCase<"MultiSampled", 1>;

def SPV_SamplingAttr:
    SPV_I32EnumAttr<
      "ImageSamplingInfo", "valid SPIR-V Image Sampling specification",
      [SPV_ISI_SingleSampled, SPV_ISI_MultiSampled]>;

def SPV_ISUI_SamplerUnknown : I32EnumAttrCase<"SamplerUnknown", 0>;
def SPV_ISUI_NeedSampler    : I32EnumAttrCase<"NeedSampler", 1>;
def SPV_ISUI_NoSampler      : I32EnumAttrCase<"NoSampler", 2>;

def SPV_SamplerUseAttr:
    SPV_I32EnumAttr<
      "ImageSamplerUseInfo", "valid SPIR-V Sampler Use specification",
      [SPV_ISUI_SamplerUnknown, SPV_ISUI_NeedSampler, SPV_ISUI_NoSampler]>;

//===----------------------------------------------------------------------===//
// SPIR-V attribute definitions
//===----------------------------------------------------------------------===//

def SPV_VerCapExtAttr : DialectAttr<
    SPIRV_Dialect,
    CPred<"$_self.isa<::mlir::spirv::VerCapExtAttr>()">,
    "version-capability-extension attribute"> {
  let storageType = "::mlir::spirv::VerCapExtAttr";
  let returnType = "::mlir::spirv::VerCapExtAttr";
  let convertFromStorage = "$_self";
}

//===----------------------------------------------------------------------===//
// SPIR-V type definitions
//===----------------------------------------------------------------------===//

class IOrUI<int width>
    : Type<Or<[CPred<"$_self.isSignlessInteger(" # width # ")">,
               CPred<"$_self.isUnsignedInteger(" # width # ")">]>,
           width # "-bit signless/unsigned integer"> {
  int bitwidth = width;
}

class SignlessOrUnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, IOrUI<w>),
              StrJoinInt<widths, "/">.result # "-bit signless/unsigned integer">;

def SPV_IsArrayType : CPred<"$_self.isa<::mlir::spirv::ArrayType>()">;
def SPV_IsCooperativeMatrixType :
  CPred<"$_self.isa<::mlir::spirv::CooperativeMatrixNVType>()">;
def SPV_IsMatrixType : CPred<"$_self.isa<::mlir::spirv::MatrixType>()">;
def SPV_IsPtrType : CPred<"$_self.isa<::mlir::spirv::PointerType>()">;
def SPV_IsRTArrayType : CPred<"$_self.isa<::mlir::spirv::RuntimeArrayType>()">;
def SPV_IsStructType : CPred<"$_self.isa<::mlir::spirv::StructType>()">;


// See https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_types
// for the definition of the following types and type categories.

def SPV_Void : TypeAlias<NoneType, "void">;
def SPV_Bool : TypeAlias<I1, "bool">;
def SPV_Integer : AnyIntOfWidths<[8, 16, 32, 64]>;
def SPV_Int32 : TypeAlias<I32, "Int32">;
def SPV_Float : FloatOfWidths<[16, 32, 64]>;
def SPV_Float16or32 : FloatOfWidths<[16, 32]>;
def SPV_Vector : VectorOfLengthAndType<[2, 3, 4],
                                       [SPV_Bool, SPV_Integer, SPV_Float]>;
// Component type check is done in the type parser for the following SPIR-V
// dialect-specific types so we use "Any" here.
def SPV_AnyPtr : DialectType<SPIRV_Dialect, SPV_IsPtrType,
                             "any SPIR-V pointer type">;
def SPV_AnyArray : DialectType<SPIRV_Dialect, SPV_IsArrayType,
                               "any SPIR-V array type">;
def SPV_AnyCooperativeMatrix : DialectType<SPIRV_Dialect,
                               SPV_IsCooperativeMatrixType,
                               "any SPIR-V cooperative matrix type">;
def SPV_AnyMatrix : DialectType<SPIRV_Dialect, SPV_IsMatrixType,
                                "any SPIR-V matrix type">;
def SPV_AnyRTArray : DialectType<SPIRV_Dialect, SPV_IsRTArrayType,
                                 "any SPIR-V runtime array type">;
def SPV_AnyStruct : DialectType<SPIRV_Dialect, SPV_IsStructType,
                                "any SPIR-V struct type">;

def SPV_Numerical : AnyTypeOf<[SPV_Integer, SPV_Float]>;
def SPV_Scalar : AnyTypeOf<[SPV_Numerical, SPV_Bool]>;
def SPV_Aggregate : AnyTypeOf<[SPV_AnyArray, SPV_AnyRTArray, SPV_AnyStruct]>;
def SPV_Composite :
    AnyTypeOf<[SPV_Vector, SPV_AnyArray, SPV_AnyRTArray, SPV_AnyStruct,
               SPV_AnyCooperativeMatrix, SPV_AnyMatrix]>;
def SPV_Type : AnyTypeOf<[
    SPV_Void, SPV_Bool, SPV_Integer, SPV_Float, SPV_Vector,
    SPV_AnyPtr, SPV_AnyArray, SPV_AnyRTArray, SPV_AnyStruct,
    SPV_AnyCooperativeMatrix, SPV_AnyMatrix
  ]>;

def SPV_SignlessOrUnsignedInt : SignlessOrUnsignedIntOfWidths<[8, 16, 32, 64]>;

class SPV_CoopMatrixOfType<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, SPV_IsCooperativeMatrixType,
    "$_self.cast<::mlir::spirv::CooperativeMatrixNVType>().getElementType()",
    "Cooperative Matrix">;

class SPV_ScalarOrVectorOf<Type type> :
    AnyTypeOf<[type, VectorOfLengthAndType<[2, 3, 4], [type]>]>;

class SPV_ScalarOrVectorOrCoopMatrixOf<Type type> :
    AnyTypeOf<[type, VectorOfLengthAndType<[2, 3, 4], [type]>,
               SPV_CoopMatrixOfType<[type]>]>;

def SPV_ScalarOrVector : AnyTypeOf<[SPV_Scalar, SPV_Vector]>;
def SPV_ScalarOrVectorOrPtr : AnyTypeOf<[SPV_ScalarOrVector, SPV_AnyPtr]>;

class SPV_Vec4<Type type> : VectorOfLengthAndType<[4], [type]>;
def SPV_IntVec4 : SPV_Vec4<SPV_Integer>;
def SPV_IOrUIVec4 : SPV_Vec4<SPV_SignlessOrUnsignedInt>;
def SPV_Int32Vec4 : SPV_Vec4<AnyI32>;

// TODO(ravishankarm): From 1.4, this should also include Composite type.
def SPV_SelectType : AnyTypeOf<[SPV_Scalar, SPV_Vector, SPV_AnyPtr]>;

//===----------------------------------------------------------------------===//
// SPIR-V OpTrait definitions
//===----------------------------------------------------------------------===//

// Check that an op can only be used within the scope of a function-like op.
def InFunctionScope : PredOpTrait<
  "op must appear in a function-like op's block",
  CPred<"isNestedInFunctionLikeOp($_op.getParentOp())">>;

// Check that an op can only be used within the scope of a module-like op.
def InModuleScope : PredOpTrait<
  "op must appear in a module-like op's block",
  CPred<"isDirectInModuleLikeOp($_op.getParentOp())">>;

//===----------------------------------------------------------------------===//
// SPIR-V opcode specification
//===----------------------------------------------------------------------===//

class SPV_OpCode<string name, int val> {
  // Name used as reference to retrieve the opcode
  string opname = name;

  // Opcode associated with the name
  int opcode = val;
}

// Begin opcode section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPV_OC_OpNop                       : I32EnumAttrCase<"OpNop", 0>;
def SPV_OC_OpUndef                     : I32EnumAttrCase<"OpUndef", 1>;
def SPV_OC_OpSourceContinued           : I32EnumAttrCase<"OpSourceContinued", 2>;
def SPV_OC_OpSource                    : I32EnumAttrCase<"OpSource", 3>;
def SPV_OC_OpSourceExtension           : I32EnumAttrCase<"OpSourceExtension", 4>;
def SPV_OC_OpName                      : I32EnumAttrCase<"OpName", 5>;
def SPV_OC_OpMemberName                : I32EnumAttrCase<"OpMemberName", 6>;
def SPV_OC_OpString                    : I32EnumAttrCase<"OpString", 7>;
def SPV_OC_OpLine                      : I32EnumAttrCase<"OpLine", 8>;
def SPV_OC_OpExtension                 : I32EnumAttrCase<"OpExtension", 10>;
def SPV_OC_OpExtInstImport             : I32EnumAttrCase<"OpExtInstImport", 11>;
def SPV_OC_OpExtInst                   : I32EnumAttrCase<"OpExtInst", 12>;
def SPV_OC_OpMemoryModel               : I32EnumAttrCase<"OpMemoryModel", 14>;
def SPV_OC_OpEntryPoint                : I32EnumAttrCase<"OpEntryPoint", 15>;
def SPV_OC_OpExecutionMode             : I32EnumAttrCase<"OpExecutionMode", 16>;
def SPV_OC_OpCapability                : I32EnumAttrCase<"OpCapability", 17>;
def SPV_OC_OpTypeVoid                  : I32EnumAttrCase<"OpTypeVoid", 19>;
def SPV_OC_OpTypeBool                  : I32EnumAttrCase<"OpTypeBool", 20>;
def SPV_OC_OpTypeInt                   : I32EnumAttrCase<"OpTypeInt", 21>;
def SPV_OC_OpTypeFloat                 : I32EnumAttrCase<"OpTypeFloat", 22>;
def SPV_OC_OpTypeVector                : I32EnumAttrCase<"OpTypeVector", 23>;
def SPV_OC_OpTypeMatrix                : I32EnumAttrCase<"OpTypeMatrix", 24>;
def SPV_OC_OpTypeArray                 : I32EnumAttrCase<"OpTypeArray", 28>;
def SPV_OC_OpTypeRuntimeArray          : I32EnumAttrCase<"OpTypeRuntimeArray", 29>;
def SPV_OC_OpTypeStruct                : I32EnumAttrCase<"OpTypeStruct", 30>;
def SPV_OC_OpTypePointer               : I32EnumAttrCase<"OpTypePointer", 32>;
def SPV_OC_OpTypeFunction              : I32EnumAttrCase<"OpTypeFunction", 33>;
def SPV_OC_OpConstantTrue              : I32EnumAttrCase<"OpConstantTrue", 41>;
def SPV_OC_OpConstantFalse             : I32EnumAttrCase<"OpConstantFalse", 42>;
def SPV_OC_OpConstant                  : I32EnumAttrCase<"OpConstant", 43>;
def SPV_OC_OpConstantComposite         : I32EnumAttrCase<"OpConstantComposite", 44>;
def SPV_OC_OpConstantNull              : I32EnumAttrCase<"OpConstantNull", 46>;
def SPV_OC_OpSpecConstantTrue          : I32EnumAttrCase<"OpSpecConstantTrue", 48>;
def SPV_OC_OpSpecConstantFalse         : I32EnumAttrCase<"OpSpecConstantFalse", 49>;
def SPV_OC_OpSpecConstant              : I32EnumAttrCase<"OpSpecConstant", 50>;
def SPV_OC_OpSpecConstantComposite     : I32EnumAttrCase<"OpSpecConstantComposite", 51>;
def SPV_OC_OpFunction                  : I32EnumAttrCase<"OpFunction", 54>;
def SPV_OC_OpFunctionParameter         : I32EnumAttrCase<"OpFunctionParameter", 55>;
def SPV_OC_OpFunctionEnd               : I32EnumAttrCase<"OpFunctionEnd", 56>;
def SPV_OC_OpFunctionCall              : I32EnumAttrCase<"OpFunctionCall", 57>;
def SPV_OC_OpVariable                  : I32EnumAttrCase<"OpVariable", 59>;
def SPV_OC_OpLoad                      : I32EnumAttrCase<"OpLoad", 61>;
def SPV_OC_OpStore                     : I32EnumAttrCase<"OpStore", 62>;
def SPV_OC_OpAccessChain               : I32EnumAttrCase<"OpAccessChain", 65>;
def SPV_OC_OpDecorate                  : I32EnumAttrCase<"OpDecorate", 71>;
def SPV_OC_OpMemberDecorate            : I32EnumAttrCase<"OpMemberDecorate", 72>;
def SPV_OC_OpCompositeConstruct        : I32EnumAttrCase<"OpCompositeConstruct", 80>;
def SPV_OC_OpCompositeExtract          : I32EnumAttrCase<"OpCompositeExtract", 81>;
def SPV_OC_OpCompositeInsert           : I32EnumAttrCase<"OpCompositeInsert", 82>;
def SPV_OC_OpTranspose                 : I32EnumAttrCase<"OpTranspose", 84>;
def SPV_OC_OpConvertFToU               : I32EnumAttrCase<"OpConvertFToU", 109>;
def SPV_OC_OpConvertFToS               : I32EnumAttrCase<"OpConvertFToS", 110>;
def SPV_OC_OpConvertSToF               : I32EnumAttrCase<"OpConvertSToF", 111>;
def SPV_OC_OpConvertUToF               : I32EnumAttrCase<"OpConvertUToF", 112>;
def SPV_OC_OpUConvert                  : I32EnumAttrCase<"OpUConvert", 113>;
def SPV_OC_OpSConvert                  : I32EnumAttrCase<"OpSConvert", 114>;
def SPV_OC_OpFConvert                  : I32EnumAttrCase<"OpFConvert", 115>;
def SPV_OC_OpBitcast                   : I32EnumAttrCase<"OpBitcast", 124>;
def SPV_OC_OpFNegate                   : I32EnumAttrCase<"OpFNegate", 127>;
def SPV_OC_OpIAdd                      : I32EnumAttrCase<"OpIAdd", 128>;
def SPV_OC_OpFAdd                      : I32EnumAttrCase<"OpFAdd", 129>;
def SPV_OC_OpISub                      : I32EnumAttrCase<"OpISub", 130>;
def SPV_OC_OpFSub                      : I32EnumAttrCase<"OpFSub", 131>;
def SPV_OC_OpIMul                      : I32EnumAttrCase<"OpIMul", 132>;
def SPV_OC_OpFMul                      : I32EnumAttrCase<"OpFMul", 133>;
def SPV_OC_OpUDiv                      : I32EnumAttrCase<"OpUDiv", 134>;
def SPV_OC_OpSDiv                      : I32EnumAttrCase<"OpSDiv", 135>;
def SPV_OC_OpFDiv                      : I32EnumAttrCase<"OpFDiv", 136>;
def SPV_OC_OpUMod                      : I32EnumAttrCase<"OpUMod", 137>;
def SPV_OC_OpSRem                      : I32EnumAttrCase<"OpSRem", 138>;
def SPV_OC_OpSMod                      : I32EnumAttrCase<"OpSMod", 139>;
def SPV_OC_OpFRem                      : I32EnumAttrCase<"OpFRem", 140>;
def SPV_OC_OpFMod                      : I32EnumAttrCase<"OpFMod", 141>;
def SPV_OC_OpMatrixTimesScalar         : I32EnumAttrCase<"OpMatrixTimesScalar", 143>;
def SPV_OC_OpLogicalEqual              : I32EnumAttrCase<"OpLogicalEqual", 164>;
def SPV_OC_OpLogicalNotEqual           : I32EnumAttrCase<"OpLogicalNotEqual", 165>;
def SPV_OC_OpLogicalOr                 : I32EnumAttrCase<"OpLogicalOr", 166>;
def SPV_OC_OpLogicalAnd                : I32EnumAttrCase<"OpLogicalAnd", 167>;
def SPV_OC_OpLogicalNot                : I32EnumAttrCase<"OpLogicalNot", 168>;
def SPV_OC_OpSelect                    : I32EnumAttrCase<"OpSelect", 169>;
def SPV_OC_OpIEqual                    : I32EnumAttrCase<"OpIEqual", 170>;
def SPV_OC_OpINotEqual                 : I32EnumAttrCase<"OpINotEqual", 171>;
def SPV_OC_OpUGreaterThan              : I32EnumAttrCase<"OpUGreaterThan", 172>;
def SPV_OC_OpSGreaterThan              : I32EnumAttrCase<"OpSGreaterThan", 173>;
def SPV_OC_OpUGreaterThanEqual         : I32EnumAttrCase<"OpUGreaterThanEqual", 174>;
def SPV_OC_OpSGreaterThanEqual         : I32EnumAttrCase<"OpSGreaterThanEqual", 175>;
def SPV_OC_OpULessThan                 : I32EnumAttrCase<"OpULessThan", 176>;
def SPV_OC_OpSLessThan                 : I32EnumAttrCase<"OpSLessThan", 177>;
def SPV_OC_OpULessThanEqual            : I32EnumAttrCase<"OpULessThanEqual", 178>;
def SPV_OC_OpSLessThanEqual            : I32EnumAttrCase<"OpSLessThanEqual", 179>;
def SPV_OC_OpFOrdEqual                 : I32EnumAttrCase<"OpFOrdEqual", 180>;
def SPV_OC_OpFUnordEqual               : I32EnumAttrCase<"OpFUnordEqual", 181>;
def SPV_OC_OpFOrdNotEqual              : I32EnumAttrCase<"OpFOrdNotEqual", 182>;
def SPV_OC_OpFUnordNotEqual            : I32EnumAttrCase<"OpFUnordNotEqual", 183>;
def SPV_OC_OpFOrdLessThan              : I32EnumAttrCase<"OpFOrdLessThan", 184>;
def SPV_OC_OpFUnordLessThan            : I32EnumAttrCase<"OpFUnordLessThan", 185>;
def SPV_OC_OpFOrdGreaterThan           : I32EnumAttrCase<"OpFOrdGreaterThan", 186>;
def SPV_OC_OpFUnordGreaterThan         : I32EnumAttrCase<"OpFUnordGreaterThan", 187>;
def SPV_OC_OpFOrdLessThanEqual         : I32EnumAttrCase<"OpFOrdLessThanEqual", 188>;
def SPV_OC_OpFUnordLessThanEqual       : I32EnumAttrCase<"OpFUnordLessThanEqual", 189>;
def SPV_OC_OpFOrdGreaterThanEqual      : I32EnumAttrCase<"OpFOrdGreaterThanEqual", 190>;
def SPV_OC_OpFUnordGreaterThanEqual    : I32EnumAttrCase<"OpFUnordGreaterThanEqual", 191>;
def SPV_OC_OpShiftRightLogical         : I32EnumAttrCase<"OpShiftRightLogical", 194>;
def SPV_OC_OpShiftRightArithmetic      : I32EnumAttrCase<"OpShiftRightArithmetic", 195>;
def SPV_OC_OpShiftLeftLogical          : I32EnumAttrCase<"OpShiftLeftLogical", 196>;
def SPV_OC_OpBitwiseOr                 : I32EnumAttrCase<"OpBitwiseOr", 197>;
def SPV_OC_OpBitwiseXor                : I32EnumAttrCase<"OpBitwiseXor", 198>;
def SPV_OC_OpBitwiseAnd                : I32EnumAttrCase<"OpBitwiseAnd", 199>;
def SPV_OC_OpNot                       : I32EnumAttrCase<"OpNot", 200>;
def SPV_OC_OpBitFieldInsert            : I32EnumAttrCase<"OpBitFieldInsert", 201>;
def SPV_OC_OpBitFieldSExtract          : I32EnumAttrCase<"OpBitFieldSExtract", 202>;
def SPV_OC_OpBitFieldUExtract          : I32EnumAttrCase<"OpBitFieldUExtract", 203>;
def SPV_OC_OpBitReverse                : I32EnumAttrCase<"OpBitReverse", 204>;
def SPV_OC_OpBitCount                  : I32EnumAttrCase<"OpBitCount", 205>;
def SPV_OC_OpControlBarrier            : I32EnumAttrCase<"OpControlBarrier", 224>;
def SPV_OC_OpMemoryBarrier             : I32EnumAttrCase<"OpMemoryBarrier", 225>;
def SPV_OC_OpAtomicCompareExchangeWeak : I32EnumAttrCase<"OpAtomicCompareExchangeWeak", 231>;
def SPV_OC_OpAtomicIIncrement          : I32EnumAttrCase<"OpAtomicIIncrement", 232>;
def SPV_OC_OpAtomicIDecrement          : I32EnumAttrCase<"OpAtomicIDecrement", 233>;
def SPV_OC_OpAtomicIAdd                : I32EnumAttrCase<"OpAtomicIAdd", 234>;
def SPV_OC_OpAtomicISub                : I32EnumAttrCase<"OpAtomicISub", 235>;
def SPV_OC_OpAtomicSMin                : I32EnumAttrCase<"OpAtomicSMin", 236>;
def SPV_OC_OpAtomicUMin                : I32EnumAttrCase<"OpAtomicUMin", 237>;
def SPV_OC_OpAtomicSMax                : I32EnumAttrCase<"OpAtomicSMax", 238>;
def SPV_OC_OpAtomicUMax                : I32EnumAttrCase<"OpAtomicUMax", 239>;
def SPV_OC_OpAtomicAnd                 : I32EnumAttrCase<"OpAtomicAnd", 240>;
def SPV_OC_OpAtomicOr                  : I32EnumAttrCase<"OpAtomicOr", 241>;
def SPV_OC_OpAtomicXor                 : I32EnumAttrCase<"OpAtomicXor", 242>;
def SPV_OC_OpPhi                       : I32EnumAttrCase<"OpPhi", 245>;
def SPV_OC_OpLoopMerge                 : I32EnumAttrCase<"OpLoopMerge", 246>;
def SPV_OC_OpSelectionMerge            : I32EnumAttrCase<"OpSelectionMerge", 247>;
def SPV_OC_OpLabel                     : I32EnumAttrCase<"OpLabel", 248>;
def SPV_OC_OpBranch                    : I32EnumAttrCase<"OpBranch", 249>;
def SPV_OC_OpBranchConditional         : I32EnumAttrCase<"OpBranchConditional", 250>;
def SPV_OC_OpReturn                    : I32EnumAttrCase<"OpReturn", 253>;
def SPV_OC_OpReturnValue               : I32EnumAttrCase<"OpReturnValue", 254>;
def SPV_OC_OpUnreachable               : I32EnumAttrCase<"OpUnreachable", 255>;
def SPV_OC_OpNoLine                    : I32EnumAttrCase<"OpNoLine", 317>;
def SPV_OC_OpModuleProcessed           : I32EnumAttrCase<"OpModuleProcessed", 330>;
def SPV_OC_OpGroupNonUniformElect      : I32EnumAttrCase<"OpGroupNonUniformElect", 333>;
def SPV_OC_OpGroupNonUniformBallot     : I32EnumAttrCase<"OpGroupNonUniformBallot", 339>;
def SPV_OC_OpGroupNonUniformIAdd       : I32EnumAttrCase<"OpGroupNonUniformIAdd", 349>;
def SPV_OC_OpGroupNonUniformFAdd       : I32EnumAttrCase<"OpGroupNonUniformFAdd", 350>;
def SPV_OC_OpGroupNonUniformIMul       : I32EnumAttrCase<"OpGroupNonUniformIMul", 351>;
def SPV_OC_OpGroupNonUniformFMul       : I32EnumAttrCase<"OpGroupNonUniformFMul", 352>;
def SPV_OC_OpGroupNonUniformSMin       : I32EnumAttrCase<"OpGroupNonUniformSMin", 353>;
def SPV_OC_OpGroupNonUniformUMin       : I32EnumAttrCase<"OpGroupNonUniformUMin", 354>;
def SPV_OC_OpGroupNonUniformFMin       : I32EnumAttrCase<"OpGroupNonUniformFMin", 355>;
def SPV_OC_OpGroupNonUniformSMax       : I32EnumAttrCase<"OpGroupNonUniformSMax", 356>;
def SPV_OC_OpGroupNonUniformUMax       : I32EnumAttrCase<"OpGroupNonUniformUMax", 357>;
def SPV_OC_OpGroupNonUniformFMax       : I32EnumAttrCase<"OpGroupNonUniformFMax", 358>;
def SPV_OC_OpSubgroupBallotKHR         : I32EnumAttrCase<"OpSubgroupBallotKHR", 4421>;
def SPV_OC_OpTypeCooperativeMatrixNV   : I32EnumAttrCase<"OpTypeCooperativeMatrixNV", 5358>;
def SPV_OC_OpCooperativeMatrixLoadNV   : I32EnumAttrCase<"OpCooperativeMatrixLoadNV", 5359>;
def SPV_OC_OpCooperativeMatrixStoreNV  : I32EnumAttrCase<"OpCooperativeMatrixStoreNV", 5360>;
def SPV_OC_OpCooperativeMatrixMulAddNV : I32EnumAttrCase<"OpCooperativeMatrixMulAddNV", 5361>;
def SPV_OC_OpCooperativeMatrixLengthNV : I32EnumAttrCase<"OpCooperativeMatrixLengthNV", 5362>;

def SPV_OpcodeAttr :
    SPV_I32EnumAttr<"Opcode", "valid SPIR-V instructions", [
      SPV_OC_OpNop, SPV_OC_OpUndef, SPV_OC_OpSourceContinued, SPV_OC_OpSource,
      SPV_OC_OpSourceExtension, SPV_OC_OpName, SPV_OC_OpMemberName, SPV_OC_OpString,
      SPV_OC_OpLine, SPV_OC_OpExtension, SPV_OC_OpExtInstImport, SPV_OC_OpExtInst,
      SPV_OC_OpMemoryModel, SPV_OC_OpEntryPoint, SPV_OC_OpExecutionMode,
      SPV_OC_OpCapability, SPV_OC_OpTypeVoid, SPV_OC_OpTypeBool, SPV_OC_OpTypeInt,
      SPV_OC_OpTypeFloat, SPV_OC_OpTypeVector, SPV_OC_OpTypeMatrix,
      SPV_OC_OpTypeArray, SPV_OC_OpTypeRuntimeArray, SPV_OC_OpTypeStruct,
      SPV_OC_OpTypePointer, SPV_OC_OpTypeFunction, SPV_OC_OpConstantTrue,
      SPV_OC_OpConstantFalse, SPV_OC_OpConstant, SPV_OC_OpConstantComposite,
      SPV_OC_OpConstantNull, SPV_OC_OpSpecConstantTrue, SPV_OC_OpSpecConstantFalse,
      SPV_OC_OpSpecConstant, SPV_OC_OpSpecConstantComposite, SPV_OC_OpFunction,
      SPV_OC_OpFunctionParameter, SPV_OC_OpFunctionEnd, SPV_OC_OpFunctionCall,
      SPV_OC_OpVariable, SPV_OC_OpLoad, SPV_OC_OpStore, SPV_OC_OpAccessChain,
      SPV_OC_OpDecorate, SPV_OC_OpMemberDecorate, SPV_OC_OpCompositeConstruct,
      SPV_OC_OpCompositeExtract, SPV_OC_OpCompositeInsert, SPV_OC_OpTranspose,
      SPV_OC_OpConvertFToU, SPV_OC_OpConvertFToS, SPV_OC_OpConvertSToF,
      SPV_OC_OpConvertUToF, SPV_OC_OpUConvert, SPV_OC_OpSConvert, SPV_OC_OpFConvert,
      SPV_OC_OpBitcast, SPV_OC_OpFNegate, SPV_OC_OpIAdd, SPV_OC_OpFAdd,
      SPV_OC_OpISub, SPV_OC_OpFSub, SPV_OC_OpIMul, SPV_OC_OpFMul, SPV_OC_OpUDiv,
      SPV_OC_OpSDiv, SPV_OC_OpFDiv, SPV_OC_OpUMod, SPV_OC_OpSRem, SPV_OC_OpSMod,
      SPV_OC_OpFRem, SPV_OC_OpFMod, SPV_OC_OpMatrixTimesScalar,
      SPV_OC_OpLogicalEqual, SPV_OC_OpLogicalNotEqual, SPV_OC_OpLogicalOr,
      SPV_OC_OpLogicalAnd, SPV_OC_OpLogicalNot, SPV_OC_OpSelect, SPV_OC_OpIEqual,
      SPV_OC_OpINotEqual, SPV_OC_OpUGreaterThan, SPV_OC_OpSGreaterThan,
      SPV_OC_OpUGreaterThanEqual, SPV_OC_OpSGreaterThanEqual, SPV_OC_OpULessThan,
      SPV_OC_OpSLessThan, SPV_OC_OpULessThanEqual, SPV_OC_OpSLessThanEqual,
      SPV_OC_OpFOrdEqual, SPV_OC_OpFUnordEqual, SPV_OC_OpFOrdNotEqual,
      SPV_OC_OpFUnordNotEqual, SPV_OC_OpFOrdLessThan, SPV_OC_OpFUnordLessThan,
      SPV_OC_OpFOrdGreaterThan, SPV_OC_OpFUnordGreaterThan,
      SPV_OC_OpFOrdLessThanEqual, SPV_OC_OpFUnordLessThanEqual,
      SPV_OC_OpFOrdGreaterThanEqual, SPV_OC_OpFUnordGreaterThanEqual,
      SPV_OC_OpShiftRightLogical, SPV_OC_OpShiftRightArithmetic,
      SPV_OC_OpShiftLeftLogical, SPV_OC_OpBitwiseOr, SPV_OC_OpBitwiseXor,
      SPV_OC_OpBitwiseAnd, SPV_OC_OpNot, SPV_OC_OpBitFieldInsert,
      SPV_OC_OpBitFieldSExtract, SPV_OC_OpBitFieldUExtract, SPV_OC_OpBitReverse,
      SPV_OC_OpBitCount, SPV_OC_OpControlBarrier, SPV_OC_OpMemoryBarrier,
      SPV_OC_OpAtomicCompareExchangeWeak, SPV_OC_OpAtomicIIncrement,
      SPV_OC_OpAtomicIDecrement, SPV_OC_OpAtomicIAdd, SPV_OC_OpAtomicISub,
      SPV_OC_OpAtomicSMin, SPV_OC_OpAtomicUMin, SPV_OC_OpAtomicSMax,
      SPV_OC_OpAtomicUMax, SPV_OC_OpAtomicAnd, SPV_OC_OpAtomicOr, SPV_OC_OpAtomicXor,
      SPV_OC_OpPhi, SPV_OC_OpLoopMerge, SPV_OC_OpSelectionMerge, SPV_OC_OpLabel,
      SPV_OC_OpBranch, SPV_OC_OpBranchConditional, SPV_OC_OpReturn,
      SPV_OC_OpReturnValue, SPV_OC_OpUnreachable, SPV_OC_OpNoLine,
      SPV_OC_OpModuleProcessed, SPV_OC_OpGroupNonUniformElect,
      SPV_OC_OpGroupNonUniformBallot, SPV_OC_OpGroupNonUniformIAdd,
      SPV_OC_OpGroupNonUniformFAdd, SPV_OC_OpGroupNonUniformIMul,
      SPV_OC_OpGroupNonUniformFMul, SPV_OC_OpGroupNonUniformSMin,
      SPV_OC_OpGroupNonUniformUMin, SPV_OC_OpGroupNonUniformFMin,
      SPV_OC_OpGroupNonUniformSMax, SPV_OC_OpGroupNonUniformUMax,
      SPV_OC_OpGroupNonUniformFMax, SPV_OC_OpSubgroupBallotKHR,
      SPV_OC_OpTypeCooperativeMatrixNV, SPV_OC_OpCooperativeMatrixLoadNV,
      SPV_OC_OpCooperativeMatrixStoreNV, SPV_OC_OpCooperativeMatrixMulAddNV,
      SPV_OC_OpCooperativeMatrixLengthNV
    ]>;

// End opcode section. Generated from SPIR-V spec; DO NOT MODIFY!

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPV_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SPIRV_Dialect, mnemonic, !listconcat(traits, [
         // TODO(antiagainst): We don't need all of the following traits for
         // every op; only the suitable ones should be added automatically
         // after ODS supports dialect-specific contents.
         DeclareOpInterfaceMethods<QueryMinVersionInterface>,
         DeclareOpInterfaceMethods<QueryMaxVersionInterface>,
         DeclareOpInterfaceMethods<QueryExtensionInterface>,
         DeclareOpInterfaceMethods<QueryCapabilityInterface>
       ])> {
  // Availability specification for this op itself.
  list<Availability> availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];

  // For each SPIR-V op, the following static functions need to be defined
  // in SPVOps.cpp:
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
  //                                               OperationState &result)
  // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
  // * static LogicalResult verify(<op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];

  // Specifies whether this op has a direct corresponding SPIR-V binary
  // instruction opcode. The (de)serializer use this field to determine whether
  // to auto-generate an entry in the (de)serialization dispatch table for this
  // op.
  bit hasOpcode = 1;

  // Name of the corresponding SPIR-V op. Only valid to use when hasOpcode is 1.
  string spirvOpName = "Op" # mnemonic;

  // Controls whether to auto-generate this op's (de)serialization method.
  // If set, it results in generation of the following methods:
  //
  // ```c++
  // template<typename OpTy> Serializer::processOp(OpTy op);
  // template<typename OpTy> Deserializer::processOp(ArrayRef<uint32_t>);
  // ```
  //
  // If this field is not set, then manual implementation of a specialization of
  // these methods is required.
  //
  // Note:
  // 1) If hasOpcode is set but autogenSerialization is not set, the
  //    (de)serializer dispatch method still calls the above method for
  //    (de)serializing this op.
  // 2) If hasOpcode is not set, but autogenSerialization is set, the
  //    above methods for (de)serialization are generated, but there is no
  //    entry added in the dispatch tables to invoke these methods. The
  //    dispatch needs to be handled manually. SPV_ExtInstOps are an
  //    example of this.
  bit autogenSerialization = 1;
}

class SPV_UnaryOp<string mnemonic, Type resultType, Type operandType,
                  list<OpTrait> traits = []> :
      SPV_Op<mnemonic, traits> {
  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return ::parseUnaryOp(parser, result); }];
  let printer = [{ return ::printUnaryOp(getOperation(), p); }];
  // No additional verification needed in addition to the ODS-generated ones.
  let verifier = [{ return success(); }];
}

class SPV_BinaryOp<string mnemonic, Type resultType, Type operandsType,
                   list<OpTrait> traits = []> :
      SPV_Op<mnemonic, traits> {
  let arguments = (ins
    SPV_ScalarOrVectorOf<operandsType>:$operand1,
    SPV_ScalarOrVectorOf<operandsType>:$operand2
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ return impl::printOneResultOp(getOperation(), p); }];
  // No additional verification needed in addition to the ODS-generated ones.
  let verifier = [{ return success(); }];
}

class SPV_ExtInstOp<string mnemonic, string setPrefix, string setName,
                    int opcode, list<OpTrait> traits = []> :
  SPV_Op<setPrefix # "." # mnemonic, traits> {

  // Extended instruction sets have no direct opcode (they share the
  // same `OpExtInst` instruction). So the hasOpcode field is set to
  // false. So no entry corresponding to these ops are added in the
  // dispatch functions for (de)serialization. The methods for
  // (de)serialization are still automatically generated (since
  // autogenSerialization remains 1). A separate method is generated
  // for dispatching extended instruction set ops.
  let hasOpcode = 0;

  // Opcode within extended instruction set.
  int extendedInstOpcode = opcode;

  // Name used to import the extended instruction set.
  string extendedInstSetName = setName;
}

#endif // SPIRV_BASE
