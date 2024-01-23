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

#ifndef MLIR_DIALECT_SPIRV_IR_BASE
#define MLIR_DIALECT_SPIRV_IR_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.td"

//===----------------------------------------------------------------------===//
// SPIR-V dialect definitions
//===----------------------------------------------------------------------===//

def SPIRV_Dialect : Dialect {
  let name = "spirv";

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

  let cppNamespace = "::mlir::spirv";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    //===------------------------------------------------------------------===//
    // Attribute
    //===------------------------------------------------------------------===//

    /// Returns the attribute name to use when specifying decorations on results
    /// of operations.
    static std::string getAttributeName(Decoration decoration);

    /// Dialect attribute parsing hook.
    Attribute parseAttribute(
        DialectAsmParser &parser, Type type) const override;
    /// Dialect attribute printing hook.
    void printAttribute(
        Attribute attr, DialectAsmPrinter &printer) const override;
  }];
}

//===----------------------------------------------------------------------===//
// Utility definitions
//===----------------------------------------------------------------------===//

// Wrapper over base BitEnumAttr to set common fields.
class SPIRV_BitEnum<string name, string description,
                    list<BitEnumAttrCaseBase> cases>
    : I32BitEnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::spirv";
}
class SPIRV_BitEnumAttr<string name, string description, string mnemonic,
                        list<BitEnumAttrCaseBase> cases> :
    EnumAttr<SPIRV_Dialect, SPIRV_BitEnum<name, description, cases>, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

// Wrapper over base I32EnumAttr to set common fields.
class SPIRV_I32Enum<string name, string description,
                    list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::spirv";
}
class SPIRV_I32EnumAttr<string name, string description, string mnemonic,
                        list<I32EnumAttrCase> cases> :
    EnumAttr<SPIRV_Dialect, SPIRV_I32Enum<name, description, cases>, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// SPIR-V availability definitions
//===----------------------------------------------------------------------===//

def SPIRV_V_1_0 : I32EnumAttrCase<"V_1_0", 0, "v1.0">;
def SPIRV_V_1_1 : I32EnumAttrCase<"V_1_1", 1, "v1.1">;
def SPIRV_V_1_2 : I32EnumAttrCase<"V_1_2", 2, "v1.2">;
def SPIRV_V_1_3 : I32EnumAttrCase<"V_1_3", 3, "v1.3">;
def SPIRV_V_1_4 : I32EnumAttrCase<"V_1_4", 4, "v1.4">;
def SPIRV_V_1_5 : I32EnumAttrCase<"V_1_5", 5, "v1.5">;
def SPIRV_V_1_6 : I32EnumAttrCase<"V_1_6", 6, "v1.6">;

def SPIRV_VersionAttr : SPIRV_I32EnumAttr<
  "Version", "valid SPIR-V version", "version", [
    SPIRV_V_1_0, SPIRV_V_1_1, SPIRV_V_1_2, SPIRV_V_1_3, SPIRV_V_1_4, SPIRV_V_1_5,
    SPIRV_V_1_6]>;

class MinVersion<I32EnumAttrCase min> : MinVersionBase<
    "QueryMinVersionInterface", SPIRV_VersionAttr, min> {
  let cppNamespace = "::mlir::spirv";
  let interfaceDescription = [{
    Querying interface for minimal required SPIR-V version.

    This interface provides a `getMinVersion()` method to query the minimal
    required version for the implementing SPIR-V operation. The returned value
    is a `mlir::spirv::Version` enumerant.
  }];
}

class MaxVersion<I32EnumAttrCase max> : MaxVersionBase<
    "QueryMaxVersionInterface", SPIRV_VersionAttr, max> {
  let cppNamespace = "::mlir::spirv";
  let interfaceDescription = [{
    Querying interface for maximal supported SPIR-V version.

    This interface provides a `getMaxVersion()` method to query the maximal
    supported version for the implementing SPIR-V operation. The returned value
    is a `mlir::spirv::Version` enumerant.
  }];
}

class Extension<list<I32EnumAttrCase> extensions> : Availability {
  let cppNamespace = "::mlir::spirv";
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

  // TODO: Returning SmallVector<ArrayRef<...>> is not recommended.
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
    !interleave(!foreach(ext, extensions,
                         "::mlir::spirv::Extension::" # ext.symbol), ", ") #
    "}; " #
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    "ArrayRef<::mlir::spirv::Extension> " #
      "ref(exts, std::size(exts));");
  let instance = "ref";
}

class Capability<list<I32EnumAttrCase> capabilities> : Availability {
  let cppNamespace = "::mlir::spirv";
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
    !interleave(!foreach(cap, capabilities,
                         "::mlir::spirv::Capability::" # cap.symbol), ", ") #
    "}; " #
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    "ArrayRef<::mlir::spirv::Capability> " #
      "ref(caps, std::size(caps));");
  let instance = "ref";
}

class SPIRVOpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::mlir::spirv";
}
// TODO: the following interfaces definitions are duplicating with the above.
// Remove them once we are able to support dialect-specific contents in ODS.
def QueryMinVersionInterface : SPIRVOpInterface<"QueryMinVersionInterface"> {
  let methods = [InterfaceMethod<
    "", "::std::optional<::mlir::spirv::Version>", "getMinVersion">];
}
def QueryMaxVersionInterface : SPIRVOpInterface<"QueryMaxVersionInterface"> {
  let methods = [InterfaceMethod<
    "", "::std::optional<::mlir::spirv::Version>", "getMaxVersion">];
}
def QueryExtensionInterface : SPIRVOpInterface<"QueryExtensionInterface"> {
  let methods = [InterfaceMethod<
    "",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1>",
    "getExtensions">];
}
def QueryCapabilityInterface : SPIRVOpInterface<"QueryCapabilityInterface"> {
  let methods = [InterfaceMethod<
    "",
    "::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1>",
    "getCapabilities">];
}

//===----------------------------------------------------------------------===//
// SPIR-V target GPU vendor and device definitions
//===----------------------------------------------------------------------===//

def SPIRV_DT_CPU           : I32EnumAttrCase<"CPU", 0>;
def SPIRV_DT_DiscreteGPU   : I32EnumAttrCase<"DiscreteGPU", 1>;
def SPIRV_DT_IntegratedGPU : I32EnumAttrCase<"IntegratedGPU", 2>;
// An accelerator other than GPU or CPU
def SPIRV_DT_Other         : I32EnumAttrCase<"Other", 3>;
// Information missing.
def SPIRV_DT_Unknown       : I32EnumAttrCase<"Unknown", 0xffffffff>;

def SPIRV_DeviceTypeAttr : SPIRV_I32EnumAttr<
  "DeviceType", "valid SPIR-V device types", "device_type", [
    SPIRV_DT_Other, SPIRV_DT_IntegratedGPU, SPIRV_DT_DiscreteGPU,
    SPIRV_DT_CPU, SPIRV_DT_Unknown
  ]>;

def SPIRV_V_AMD         : I32EnumAttrCase<"AMD", 0>;
def SPIRV_V_Apple       : I32EnumAttrCase<"Apple", 1>;
def SPIRV_V_ARM         : I32EnumAttrCase<"ARM", 2>;
def SPIRV_V_Imagination : I32EnumAttrCase<"Imagination", 3>;
def SPIRV_V_Intel       : I32EnumAttrCase<"Intel", 4>;
def SPIRV_V_NVIDIA      : I32EnumAttrCase<"NVIDIA", 5>;
def SPIRV_V_Qualcomm    : I32EnumAttrCase<"Qualcomm", 6>;
def SPIRV_V_SwiftShader : I32EnumAttrCase<"SwiftShader", 7>;
def SPIRV_V_Unknown     : I32EnumAttrCase<"Unknown", 0xffffffff>;

def SPIRV_VendorAttr : SPIRV_I32EnumAttr<
  "Vendor", "recognized SPIR-V vendor strings", "vendor", [
    SPIRV_V_AMD, SPIRV_V_Apple, SPIRV_V_ARM, SPIRV_V_Imagination,
    SPIRV_V_Intel, SPIRV_V_NVIDIA, SPIRV_V_Qualcomm, SPIRV_V_SwiftShader,
    SPIRV_V_Unknown
  ]>;

def SPIRV_CA_Metal   : I32EnumAttrCase<"Metal", 0>;
def SPIRV_CA_OpenCL  : I32EnumAttrCase<"OpenCL", 1>;
def SPIRV_CA_Vulkan  : I32EnumAttrCase<"Vulkan", 2>;
def SPIRV_CA_WebGPU  : I32EnumAttrCase<"WebGPU", 3>;
def SPIRV_CA_Unknown : I32EnumAttrCase<"Unknown", 0xffffffff>;

def SPIRV_ClientAPIAttr : SPIRV_I32EnumAttr<
  "ClientAPI", "recognized SPIR-V client APIs", "client_api", [
    SPIRV_CA_Metal, SPIRV_CA_OpenCL, SPIRV_CA_Vulkan, SPIRV_CA_WebGPU,
    SPIRV_CA_Unknown
  ]>;

//===----------------------------------------------------------------------===//
// SPIR-V extension definitions
//===----------------------------------------------------------------------===//

// Extensions known to the SPIR-V dialect.
// https://github.com/KhronosGroup/SPIRV-Registry has the full list.
// These start with the 'SPV' prefix to match the Khronos' naming scheme.
def SPV_KHR_16bit_storage                    : I32EnumAttrCase<"SPV_KHR_16bit_storage", 0>;
def SPV_KHR_8bit_storage                     : I32EnumAttrCase<"SPV_KHR_8bit_storage", 1>;
def SPV_KHR_device_group                     : I32EnumAttrCase<"SPV_KHR_device_group", 2>;
def SPV_KHR_float_controls                   : I32EnumAttrCase<"SPV_KHR_float_controls", 3>;
def SPV_KHR_physical_storage_buffer          : I32EnumAttrCase<"SPV_KHR_physical_storage_buffer", 4>;
def SPV_KHR_multiview                        : I32EnumAttrCase<"SPV_KHR_multiview", 5>;
def SPV_KHR_no_integer_wrap_decoration       : I32EnumAttrCase<"SPV_KHR_no_integer_wrap_decoration", 6>;
def SPV_KHR_post_depth_coverage              : I32EnumAttrCase<"SPV_KHR_post_depth_coverage", 7>;
def SPV_KHR_shader_atomic_counter_ops        : I32EnumAttrCase<"SPV_KHR_shader_atomic_counter_ops", 8>;
def SPV_KHR_shader_ballot                    : I32EnumAttrCase<"SPV_KHR_shader_ballot", 9>;
def SPV_KHR_shader_clock                     : I32EnumAttrCase<"SPV_KHR_shader_clock", 10>;
def SPV_KHR_shader_draw_parameters           : I32EnumAttrCase<"SPV_KHR_shader_draw_parameters", 11>;
def SPV_KHR_storage_buffer_storage_class     : I32EnumAttrCase<"SPV_KHR_storage_buffer_storage_class", 12>;
def SPV_KHR_subgroup_vote                    : I32EnumAttrCase<"SPV_KHR_subgroup_vote", 13>;
def SPV_KHR_variable_pointers                : I32EnumAttrCase<"SPV_KHR_variable_pointers", 14>;
def SPV_KHR_vulkan_memory_model              : I32EnumAttrCase<"SPV_KHR_vulkan_memory_model", 15>;
def SPV_KHR_expect_assume                    : I32EnumAttrCase<"SPV_KHR_expect_assume", 16>;
def SPV_KHR_integer_dot_product              : I32EnumAttrCase<"SPV_KHR_integer_dot_product", 17>;
def SPV_KHR_bit_instructions                 : I32EnumAttrCase<"SPV_KHR_bit_instructions", 18>;
def SPV_KHR_fragment_shading_rate            : I32EnumAttrCase<"SPV_KHR_fragment_shading_rate", 19>;
def SPV_KHR_workgroup_memory_explicit_layout : I32EnumAttrCase<"SPV_KHR_workgroup_memory_explicit_layout", 20>;
def SPV_KHR_ray_query                        : I32EnumAttrCase<"SPV_KHR_ray_query", 21>;
def SPV_KHR_ray_tracing                      : I32EnumAttrCase<"SPV_KHR_ray_tracing", 22>;
def SPV_KHR_subgroup_uniform_control_flow    : I32EnumAttrCase<"SPV_KHR_subgroup_uniform_control_flow", 23>;
def SPV_KHR_linkonce_odr                     : I32EnumAttrCase<"SPV_KHR_linkonce_odr", 24>;
def SPV_KHR_fragment_shader_barycentric      : I32EnumAttrCase<"SPV_KHR_fragment_shader_barycentric", 25>;
def SPV_KHR_ray_cull_mask                    : I32EnumAttrCase<"SPV_KHR_ray_cull_mask", 26>;
def SPV_KHR_uniform_group_instructions       : I32EnumAttrCase<"SPV_KHR_uniform_group_instructions", 27>;
def SPV_KHR_subgroup_rotate                  : I32EnumAttrCase<"SPV_KHR_subgroup_rotate", 28>;
def SPV_KHR_non_semantic_info                : I32EnumAttrCase<"SPV_KHR_non_semantic_info", 29>;
def SPV_KHR_terminate_invocation             : I32EnumAttrCase<"SPV_KHR_terminate_invocation", 30>;
def SPV_KHR_cooperative_matrix               : I32EnumAttrCase<"SPV_KHR_cooperative_matrix", 31>;

def SPV_EXT_demote_to_helper_invocation  : I32EnumAttrCase<"SPV_EXT_demote_to_helper_invocation", 1000>;
def SPV_EXT_descriptor_indexing          : I32EnumAttrCase<"SPV_EXT_descriptor_indexing", 1001>;
def SPV_EXT_fragment_fully_covered       : I32EnumAttrCase<"SPV_EXT_fragment_fully_covered", 1002>;
def SPV_EXT_fragment_invocation_density  : I32EnumAttrCase<"SPV_EXT_fragment_invocation_density", 1003>;
def SPV_EXT_fragment_shader_interlock    : I32EnumAttrCase<"SPV_EXT_fragment_shader_interlock", 1004>;
def SPV_EXT_physical_storage_buffer      : I32EnumAttrCase<"SPV_EXT_physical_storage_buffer", 1005>;
def SPV_EXT_shader_stencil_export        : I32EnumAttrCase<"SPV_EXT_shader_stencil_export", 1006>;
def SPV_EXT_shader_viewport_index_layer  : I32EnumAttrCase<"SPV_EXT_shader_viewport_index_layer", 1007>;
def SPV_EXT_shader_atomic_float_add      : I32EnumAttrCase<"SPV_EXT_shader_atomic_float_add", 1008>;
def SPV_EXT_shader_atomic_float_min_max  : I32EnumAttrCase<"SPV_EXT_shader_atomic_float_min_max", 1009>;
def SPV_EXT_shader_image_int64           : I32EnumAttrCase<"SPV_EXT_shader_image_int64", 1010>;
def SPV_EXT_shader_atomic_float16_add    : I32EnumAttrCase<"SPV_EXT_shader_atomic_float16_add", 1011>;

def SPV_AMD_gpu_shader_half_float_fetch          : I32EnumAttrCase<"SPV_AMD_gpu_shader_half_float_fetch", 2000>;
def SPV_AMD_shader_ballot                        : I32EnumAttrCase<"SPV_AMD_shader_ballot", 2001>;
def SPV_AMD_shader_explicit_vertex_parameter     : I32EnumAttrCase<"SPV_AMD_shader_explicit_vertex_parameter", 2002>;
def SPV_AMD_shader_fragment_mask                 : I32EnumAttrCase<"SPV_AMD_shader_fragment_mask", 2003>;
def SPV_AMD_shader_image_load_store_lod          : I32EnumAttrCase<"SPV_AMD_shader_image_load_store_lod", 2004>;
def SPV_AMD_texture_gather_bias_lod              : I32EnumAttrCase<"SPV_AMD_texture_gather_bias_lod", 2005>;
def SPV_AMD_shader_early_and_late_fragment_tests : I32EnumAttrCase<"SPV_AMD_shader_early_and_late_fragment_tests", 2006>;

def SPV_GOOGLE_decorate_string           : I32EnumAttrCase<"SPV_GOOGLE_decorate_string", 3000>;
def SPV_GOOGLE_hlsl_functionality1       : I32EnumAttrCase<"SPV_GOOGLE_hlsl_functionality1", 3001>;
def SPV_GOOGLE_user_type                 : I32EnumAttrCase<"SPV_GOOGLE_user_type", 3002>;

def SPV_INTEL_device_side_avc_motion_estimation  : I32EnumAttrCase<"SPV_INTEL_device_side_avc_motion_estimation", 4000>;
def SPV_INTEL_media_block_io                     : I32EnumAttrCase<"SPV_INTEL_media_block_io", 4001>;
def SPV_INTEL_shader_integer_functions2          : I32EnumAttrCase<"SPV_INTEL_shader_integer_functions2", 4002>;
def SPV_INTEL_subgroups                          : I32EnumAttrCase<"SPV_INTEL_subgroups", 4003>;
def SPV_INTEL_float_controls2                    : I32EnumAttrCase<"SPV_INTEL_float_controls2", 4004>;
def SPV_INTEL_function_pointers                  : I32EnumAttrCase<"SPV_INTEL_function_pointers", 4005>;
def SPV_INTEL_inline_assembly                    : I32EnumAttrCase<"SPV_INTEL_inline_assembly", 4006>;
def SPV_INTEL_vector_compute                     : I32EnumAttrCase<"SPV_INTEL_vector_compute", 4007>;
def SPV_INTEL_variable_length_array              : I32EnumAttrCase<"SPV_INTEL_variable_length_array", 4008>;
def SPV_INTEL_fpga_memory_attributes             : I32EnumAttrCase<"SPV_INTEL_fpga_memory_attributes", 4009>;
def SPV_INTEL_arbitrary_precision_integers       : I32EnumAttrCase<"SPV_INTEL_arbitrary_precision_integers", 4010>;
def SPV_INTEL_arbitrary_precision_floating_point : I32EnumAttrCase<"SPV_INTEL_arbitrary_precision_floating_point", 4011>;
def SPV_INTEL_unstructured_loop_controls         : I32EnumAttrCase<"SPV_INTEL_unstructured_loop_controls", 4012>;
def SPV_INTEL_fpga_loop_controls                 : I32EnumAttrCase<"SPV_INTEL_fpga_loop_controls", 4013>;
def SPV_INTEL_kernel_attributes                  : I32EnumAttrCase<"SPV_INTEL_kernel_attributes", 4014>;
def SPV_INTEL_fpga_memory_accesses               : I32EnumAttrCase<"SPV_INTEL_fpga_memory_accesses", 4015>;
def SPV_INTEL_fpga_cluster_attributes            : I32EnumAttrCase<"SPV_INTEL_fpga_cluster_attributes", 4016>;
def SPV_INTEL_loop_fuse                          : I32EnumAttrCase<"SPV_INTEL_loop_fuse", 4017>;
def SPV_INTEL_fpga_buffer_location               : I32EnumAttrCase<"SPV_INTEL_fpga_buffer_location", 4018>;
def SPV_INTEL_arbitrary_precision_fixed_point    : I32EnumAttrCase<"SPV_INTEL_arbitrary_precision_fixed_point", 4019>;
def SPV_INTEL_usm_storage_classes                : I32EnumAttrCase<"SPV_INTEL_usm_storage_classes", 4020>;
def SPV_INTEL_io_pipes                           : I32EnumAttrCase<"SPV_INTEL_io_pipes", 4021>;
def SPV_INTEL_blocking_pipes                     : I32EnumAttrCase<"SPV_INTEL_blocking_pipes", 4022>;
def SPV_INTEL_fpga_reg                           : I32EnumAttrCase<"SPV_INTEL_fpga_reg", 4023>;
def SPV_INTEL_long_constant_composite            : I32EnumAttrCase<"SPV_INTEL_long_constant_composite", 4024>;
def SPV_INTEL_optnone                            : I32EnumAttrCase<"SPV_INTEL_optnone", 4025>;
def SPV_INTEL_debug_module                       : I32EnumAttrCase<"SPV_INTEL_debug_module", 4026>;
def SPV_INTEL_fp_fast_math_mode                  : I32EnumAttrCase<"SPV_INTEL_fp_fast_math_mode", 4027>;
def SPV_INTEL_memory_access_aliasing             : I32EnumAttrCase<"SPV_INTEL_memory_access_aliasing", 4028>;
def SPV_INTEL_split_barrier                      : I32EnumAttrCase<"SPV_INTEL_split_barrier", 4029>;
def SPV_INTEL_joint_matrix                       : I32EnumAttrCase<"SPV_INTEL_joint_matrix", 4030>;
def SPV_INTEL_bfloat16_conversion                : I32EnumAttrCase<"SPV_INTEL_bfloat16_conversion", 4031>;

def SPV_NV_compute_shader_derivatives    : I32EnumAttrCase<"SPV_NV_compute_shader_derivatives", 5000>;
def SPV_NV_cooperative_matrix            : I32EnumAttrCase<"SPV_NV_cooperative_matrix", 5001>;
def SPV_NV_fragment_shader_barycentric   : I32EnumAttrCase<"SPV_NV_fragment_shader_barycentric", 5002>;
def SPV_NV_geometry_shader_passthrough   : I32EnumAttrCase<"SPV_NV_geometry_shader_passthrough", 5003>;
def SPV_NV_mesh_shader                   : I32EnumAttrCase<"SPV_NV_mesh_shader", 5004>;
def SPV_NV_ray_tracing                   : I32EnumAttrCase<"SPV_NV_ray_tracing", 5005>;
def SPV_NV_sample_mask_override_coverage : I32EnumAttrCase<"SPV_NV_sample_mask_override_coverage", 5006>;
def SPV_NV_shader_image_footprint        : I32EnumAttrCase<"SPV_NV_shader_image_footprint", 5007>;
def SPV_NV_shader_sm_builtins            : I32EnumAttrCase<"SPV_NV_shader_sm_builtins", 5008>;
def SPV_NV_shader_subgroup_partitioned   : I32EnumAttrCase<"SPV_NV_shader_subgroup_partitioned", 5009>;
def SPV_NV_shading_rate                  : I32EnumAttrCase<"SPV_NV_shading_rate", 5010>;
def SPV_NV_stereo_view_rendering         : I32EnumAttrCase<"SPV_NV_stereo_view_rendering", 5011>;
def SPV_NV_viewport_array2               : I32EnumAttrCase<"SPV_NV_viewport_array2", 5012>;
def SPV_NV_bindless_texture              : I32EnumAttrCase<"SPV_NV_bindless_texture", 5013>;
def SPV_NV_ray_tracing_motion_blur       : I32EnumAttrCase<"SPV_NV_ray_tracing_motion_blur", 5014>;

def SPV_NVX_multiview_per_view_attributes : I32EnumAttrCase<"SPV_NVX_multiview_per_view_attributes", 5015>;

def SPIRV_ExtensionAttr :
    SPIRV_I32EnumAttr<"Extension", "supported SPIR-V extensions", "ext", [
      SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_device_group,
      SPV_KHR_float_controls, SPV_KHR_physical_storage_buffer, SPV_KHR_multiview,
      SPV_KHR_no_integer_wrap_decoration, SPV_KHR_post_depth_coverage,
      SPV_KHR_shader_atomic_counter_ops, SPV_KHR_shader_ballot,
      SPV_KHR_shader_clock, SPV_KHR_shader_draw_parameters,
      SPV_KHR_storage_buffer_storage_class, SPV_KHR_subgroup_vote,
      SPV_KHR_variable_pointers, SPV_KHR_vulkan_memory_model, SPV_KHR_expect_assume,
      SPV_KHR_integer_dot_product, SPV_KHR_bit_instructions, SPV_KHR_fragment_shading_rate,
      SPV_KHR_workgroup_memory_explicit_layout, SPV_KHR_ray_query,
      SPV_KHR_ray_tracing, SPV_KHR_subgroup_uniform_control_flow, SPV_KHR_linkonce_odr,
      SPV_KHR_fragment_shader_barycentric, SPV_KHR_ray_cull_mask,
      SPV_KHR_uniform_group_instructions, SPV_KHR_subgroup_rotate,
      SPV_KHR_non_semantic_info, SPV_KHR_terminate_invocation,
      SPV_KHR_cooperative_matrix,
      SPV_EXT_demote_to_helper_invocation, SPV_EXT_descriptor_indexing,
      SPV_EXT_fragment_fully_covered, SPV_EXT_fragment_invocation_density,
      SPV_EXT_fragment_shader_interlock, SPV_EXT_physical_storage_buffer,
      SPV_EXT_shader_stencil_export, SPV_EXT_shader_viewport_index_layer,
      SPV_EXT_shader_atomic_float_add, SPV_EXT_shader_atomic_float_min_max,
      SPV_EXT_shader_image_int64, SPV_EXT_shader_atomic_float16_add,
      SPV_AMD_gpu_shader_half_float_fetch, SPV_AMD_shader_ballot,
      SPV_AMD_shader_explicit_vertex_parameter, SPV_AMD_shader_fragment_mask,
      SPV_AMD_shader_image_load_store_lod, SPV_AMD_texture_gather_bias_lod,
      SPV_AMD_shader_early_and_late_fragment_tests,
      SPV_GOOGLE_decorate_string, SPV_GOOGLE_hlsl_functionality1, SPV_GOOGLE_user_type,
      SPV_INTEL_device_side_avc_motion_estimation, SPV_INTEL_media_block_io,
      SPV_INTEL_shader_integer_functions2, SPV_INTEL_subgroups, SPV_INTEL_vector_compute,
      SPV_INTEL_float_controls2, SPV_INTEL_function_pointers, SPV_INTEL_inline_assembly,
      SPV_INTEL_variable_length_array, SPV_INTEL_fpga_memory_attributes,
      SPV_INTEL_unstructured_loop_controls, SPV_INTEL_fpga_loop_controls,
      SPV_INTEL_arbitrary_precision_integers, SPV_INTEL_arbitrary_precision_floating_point,
      SPV_INTEL_kernel_attributes, SPV_INTEL_fpga_memory_accesses,
      SPV_INTEL_fpga_cluster_attributes, SPV_INTEL_loop_fuse,
      SPV_INTEL_fpga_buffer_location, SPV_INTEL_arbitrary_precision_fixed_point,
      SPV_INTEL_usm_storage_classes, SPV_INTEL_io_pipes, SPV_INTEL_blocking_pipes,
      SPV_INTEL_fpga_reg, SPV_INTEL_long_constant_composite, SPV_INTEL_optnone,
      SPV_INTEL_debug_module, SPV_INTEL_fp_fast_math_mode,
      SPV_INTEL_memory_access_aliasing, SPV_INTEL_split_barrier, SPV_INTEL_joint_matrix,
      SPV_INTEL_bfloat16_conversion, SPV_NV_compute_shader_derivatives, SPV_NV_cooperative_matrix,
      SPV_NV_fragment_shader_barycentric, SPV_NV_geometry_shader_passthrough,
      SPV_NV_mesh_shader, SPV_NV_ray_tracing, SPV_NV_sample_mask_override_coverage,
      SPV_NV_shader_image_footprint, SPV_NV_shader_sm_builtins,
      SPV_NV_shader_subgroup_partitioned, SPV_NV_shading_rate,
      SPV_NV_stereo_view_rendering, SPV_NV_viewport_array2, SPV_NV_bindless_texture,
      SPV_NV_ray_tracing_motion_blur, SPV_NVX_multiview_per_view_attributes
    ]>;

//===----------------------------------------------------------------------===//
// SPIR-V enum definitions
//===----------------------------------------------------------------------===//

// Begin enum section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPIRV_C_Matrix                                      : I32EnumAttrCase<"Matrix", 0>;
def SPIRV_C_Addresses                                   : I32EnumAttrCase<"Addresses", 4>;
def SPIRV_C_Linkage                                     : I32EnumAttrCase<"Linkage", 5>;
def SPIRV_C_Kernel                                      : I32EnumAttrCase<"Kernel", 6>;
def SPIRV_C_Float16                                     : I32EnumAttrCase<"Float16", 9>;
def SPIRV_C_Float64                                     : I32EnumAttrCase<"Float64", 10>;
def SPIRV_C_Int64                                       : I32EnumAttrCase<"Int64", 11>;
def SPIRV_C_Groups                                      : I32EnumAttrCase<"Groups", 18> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_ballot]>
  ];
}
def SPIRV_C_Int16                                       : I32EnumAttrCase<"Int16", 22>;
def SPIRV_C_Int8                                        : I32EnumAttrCase<"Int8", 39>;
def SPIRV_C_Sampled1D                                   : I32EnumAttrCase<"Sampled1D", 43>;
def SPIRV_C_SampledBuffer                               : I32EnumAttrCase<"SampledBuffer", 46>;
def SPIRV_C_GroupNonUniform                             : I32EnumAttrCase<"GroupNonUniform", 61> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_ShaderLayer                                 : I32EnumAttrCase<"ShaderLayer", 69> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_ShaderViewportIndex                         : I32EnumAttrCase<"ShaderViewportIndex", 70> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_UniformDecoration                           : I32EnumAttrCase<"UniformDecoration", 71> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_SubgroupBallotKHR                           : I32EnumAttrCase<"SubgroupBallotKHR", 4423> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_ballot]>
  ];
}
def SPIRV_C_SubgroupVoteKHR                             : I32EnumAttrCase<"SubgroupVoteKHR", 4431> {
  list<Availability> availability = [
    Extension<[SPV_KHR_subgroup_vote]>
  ];
}
def SPIRV_C_StorageBuffer16BitAccess                    : I32EnumAttrCase<"StorageBuffer16BitAccess", 4433> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPIRV_C_StoragePushConstant16                       : I32EnumAttrCase<"StoragePushConstant16", 4435> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPIRV_C_StorageInputOutput16                        : I32EnumAttrCase<"StorageInputOutput16", 4436> {
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPIRV_C_DeviceGroup                                 : I32EnumAttrCase<"DeviceGroup", 4437> {
  list<Availability> availability = [
    Extension<[SPV_KHR_device_group]>
  ];
}
def SPIRV_C_AtomicStorageOps                            : I32EnumAttrCase<"AtomicStorageOps", 4445> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_atomic_counter_ops]>
  ];
}
def SPIRV_C_SampleMaskPostDepthCoverage                 : I32EnumAttrCase<"SampleMaskPostDepthCoverage", 4447> {
  list<Availability> availability = [
    Extension<[SPV_KHR_post_depth_coverage]>
  ];
}
def SPIRV_C_StorageBuffer8BitAccess                     : I32EnumAttrCase<"StorageBuffer8BitAccess", 4448> {
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPIRV_C_StoragePushConstant8                        : I32EnumAttrCase<"StoragePushConstant8", 4450> {
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPIRV_C_DenormPreserve                              : I32EnumAttrCase<"DenormPreserve", 4464> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPIRV_C_DenormFlushToZero                           : I32EnumAttrCase<"DenormFlushToZero", 4465> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPIRV_C_SignedZeroInfNanPreserve                    : I32EnumAttrCase<"SignedZeroInfNanPreserve", 4466> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPIRV_C_RoundingModeRTE                             : I32EnumAttrCase<"RoundingModeRTE", 4467> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPIRV_C_RoundingModeRTZ                             : I32EnumAttrCase<"RoundingModeRTZ", 4468> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>
  ];
}
def SPIRV_C_ImageFootprintNV                            : I32EnumAttrCase<"ImageFootprintNV", 5282> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_image_footprint]>
  ];
}
def SPIRV_C_FragmentBarycentricKHR                      : I32EnumAttrCase<"FragmentBarycentricKHR", 5284> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shader_barycentric, SPV_NV_fragment_shader_barycentric]>
  ];
}
def SPIRV_C_ComputeDerivativeGroupQuadsNV               : I32EnumAttrCase<"ComputeDerivativeGroupQuadsNV", 5288> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>
  ];
}
def SPIRV_C_GroupNonUniformPartitionedNV                : I32EnumAttrCase<"GroupNonUniformPartitionedNV", 5297> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>
  ];
}
def SPIRV_C_VulkanMemoryModel                           : I32EnumAttrCase<"VulkanMemoryModel", 5345> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_VulkanMemoryModelDeviceScope                : I32EnumAttrCase<"VulkanMemoryModelDeviceScope", 5346> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_ComputeDerivativeGroupLinearNV              : I32EnumAttrCase<"ComputeDerivativeGroupLinearNV", 5350> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>
  ];
}
def SPIRV_C_BindlessTextureNV                           : I32EnumAttrCase<"BindlessTextureNV", 5390> {
  list<Availability> availability = [
    Extension<[SPV_NV_bindless_texture]>
  ];
}
def SPIRV_C_SubgroupShuffleINTEL                        : I32EnumAttrCase<"SubgroupShuffleINTEL", 5568> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPIRV_C_SubgroupBufferBlockIOINTEL                  : I32EnumAttrCase<"SubgroupBufferBlockIOINTEL", 5569> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPIRV_C_SubgroupImageBlockIOINTEL                   : I32EnumAttrCase<"SubgroupImageBlockIOINTEL", 5570> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_subgroups]>
  ];
}
def SPIRV_C_SubgroupImageMediaBlockIOINTEL              : I32EnumAttrCase<"SubgroupImageMediaBlockIOINTEL", 5579> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_media_block_io]>
  ];
}
def SPIRV_C_RoundToInfinityINTEL                        : I32EnumAttrCase<"RoundToInfinityINTEL", 5582> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_float_controls2]>
  ];
}
def SPIRV_C_FloatingPointModeINTEL                      : I32EnumAttrCase<"FloatingPointModeINTEL", 5583> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_float_controls2]>
  ];
}
def SPIRV_C_FunctionPointersINTEL                       : I32EnumAttrCase<"FunctionPointersINTEL", 5603> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_function_pointers]>
  ];
}
def SPIRV_C_IndirectReferencesINTEL                     : I32EnumAttrCase<"IndirectReferencesINTEL", 5604> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_function_pointers]>
  ];
}
def SPIRV_C_AsmINTEL                                    : I32EnumAttrCase<"AsmINTEL", 5606> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_inline_assembly]>
  ];
}
def SPIRV_C_AtomicFloat32MinMaxEXT                      : I32EnumAttrCase<"AtomicFloat32MinMaxEXT", 5612> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float_min_max]>
  ];
}
def SPIRV_C_AtomicFloat64MinMaxEXT                      : I32EnumAttrCase<"AtomicFloat64MinMaxEXT", 5613> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float_min_max]>
  ];
}
def SPIRV_C_AtomicFloat16MinMaxEXT                      : I32EnumAttrCase<"AtomicFloat16MinMaxEXT", 5616> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float_min_max]>
  ];
}
def SPIRV_C_VectorAnyINTEL                              : I32EnumAttrCase<"VectorAnyINTEL", 5619> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_vector_compute]>
  ];
}
def SPIRV_C_ExpectAssumeKHR                             : I32EnumAttrCase<"ExpectAssumeKHR", 5629> {
  list<Availability> availability = [
    Extension<[SPV_KHR_expect_assume]>
  ];
}
def SPIRV_C_SubgroupAvcMotionEstimationINTEL            : I32EnumAttrCase<"SubgroupAvcMotionEstimationINTEL", 5696> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPIRV_C_SubgroupAvcMotionEstimationIntraINTEL       : I32EnumAttrCase<"SubgroupAvcMotionEstimationIntraINTEL", 5697> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPIRV_C_SubgroupAvcMotionEstimationChromaINTEL      : I32EnumAttrCase<"SubgroupAvcMotionEstimationChromaINTEL", 5698> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_device_side_avc_motion_estimation]>
  ];
}
def SPIRV_C_VariableLengthArrayINTEL                    : I32EnumAttrCase<"VariableLengthArrayINTEL", 5817> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_variable_length_array]>
  ];
}
def SPIRV_C_FunctionFloatControlINTEL                   : I32EnumAttrCase<"FunctionFloatControlINTEL", 5821> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_float_controls2]>
  ];
}
def SPIRV_C_FPGAMemoryAttributesINTEL                   : I32EnumAttrCase<"FPGAMemoryAttributesINTEL", 5824> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>
  ];
}
def SPIRV_C_ArbitraryPrecisionIntegersINTEL             : I32EnumAttrCase<"ArbitraryPrecisionIntegersINTEL", 5844> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_arbitrary_precision_integers]>
  ];
}
def SPIRV_C_ArbitraryPrecisionFloatingPointINTEL        : I32EnumAttrCase<"ArbitraryPrecisionFloatingPointINTEL", 5845> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_arbitrary_precision_floating_point]>
  ];
}
def SPIRV_C_UnstructuredLoopControlsINTEL               : I32EnumAttrCase<"UnstructuredLoopControlsINTEL", 5886> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_unstructured_loop_controls]>
  ];
}
def SPIRV_C_FPGALoopControlsINTEL                       : I32EnumAttrCase<"FPGALoopControlsINTEL", 5888> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>
  ];
}
def SPIRV_C_KernelAttributesINTEL                       : I32EnumAttrCase<"KernelAttributesINTEL", 5892> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>
  ];
}
def SPIRV_C_FPGAKernelAttributesINTEL                   : I32EnumAttrCase<"FPGAKernelAttributesINTEL", 5897> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>
  ];
}
def SPIRV_C_FPGAMemoryAccessesINTEL                     : I32EnumAttrCase<"FPGAMemoryAccessesINTEL", 5898> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_accesses]>
  ];
}
def SPIRV_C_FPGAClusterAttributesINTEL                  : I32EnumAttrCase<"FPGAClusterAttributesINTEL", 5904> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_cluster_attributes]>
  ];
}
def SPIRV_C_LoopFuseINTEL                               : I32EnumAttrCase<"LoopFuseINTEL", 5906> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_loop_fuse]>
  ];
}
def SPIRV_C_MemoryAccessAliasingINTEL                   : I32EnumAttrCase<"MemoryAccessAliasingINTEL", 5910> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_memory_access_aliasing]>
  ];
}
def SPIRV_C_FPGABufferLocationINTEL                     : I32EnumAttrCase<"FPGABufferLocationINTEL", 5920> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_buffer_location]>
  ];
}
def SPIRV_C_ArbitraryPrecisionFixedPointINTEL           : I32EnumAttrCase<"ArbitraryPrecisionFixedPointINTEL", 5922> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_arbitrary_precision_fixed_point]>
  ];
}
def SPIRV_C_USMStorageClassesINTEL                      : I32EnumAttrCase<"USMStorageClassesINTEL", 5935> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_usm_storage_classes]>
  ];
}
def SPIRV_C_IOPipesINTEL                                : I32EnumAttrCase<"IOPipesINTEL", 5943> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_io_pipes]>
  ];
}
def SPIRV_C_BlockingPipesINTEL                          : I32EnumAttrCase<"BlockingPipesINTEL", 5945> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_blocking_pipes]>
  ];
}
def SPIRV_C_FPGARegINTEL                                : I32EnumAttrCase<"FPGARegINTEL", 5948> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_reg]>
  ];
}
def SPIRV_C_DotProductInputAll                          : I32EnumAttrCase<"DotProductInputAll", 6016> {
  list<Availability> availability = [
    Extension<[SPV_KHR_integer_dot_product]>,
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_DotProductInput4x8Bit                       : I32EnumAttrCase<"DotProductInput4x8Bit", 6017> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Int8];
  list<Availability> availability = [
    Extension<[SPV_KHR_integer_dot_product]>,
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_DotProductInput4x8BitPacked                 : I32EnumAttrCase<"DotProductInput4x8BitPacked", 6018> {
  list<Availability> availability = [
    Extension<[SPV_KHR_integer_dot_product]>,
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_DotProduct                                  : I32EnumAttrCase<"DotProduct", 6019> {
  list<Availability> availability = [
    Extension<[SPV_KHR_integer_dot_product]>,
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_RayCullMaskKHR                              : I32EnumAttrCase<"RayCullMaskKHR", 6020> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_cull_mask]>
  ];
}
def SPIRV_C_CooperativeMatrixKHR                        : I32EnumAttrCase<"CooperativeMatrixKHR", 6022> {
  list<Availability> availability = [
    Extension<[SPV_KHR_cooperative_matrix]>,
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_BitInstructions                             : I32EnumAttrCase<"BitInstructions", 6025> {
  list<Availability> availability = [
    Extension<[SPV_KHR_bit_instructions]>
  ];
}
def SPIRV_C_AtomicFloat32AddEXT                         : I32EnumAttrCase<"AtomicFloat32AddEXT", 6033> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float_add]>
  ];
}
def SPIRV_C_AtomicFloat64AddEXT                         : I32EnumAttrCase<"AtomicFloat64AddEXT", 6034> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float_add]>
  ];
}
def SPIRV_C_LongConstantCompositeINTEL                  : I32EnumAttrCase<"LongConstantCompositeINTEL", 6089> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_long_constant_composite]>
  ];
}
def SPIRV_C_OptNoneINTEL                                : I32EnumAttrCase<"OptNoneINTEL", 6094> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_optnone]>
  ];
}
def SPIRV_C_AtomicFloat16AddEXT                         : I32EnumAttrCase<"AtomicFloat16AddEXT", 6095> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_atomic_float16_add]>
  ];
}
def SPIRV_C_DebugInfoModuleINTEL                        : I32EnumAttrCase<"DebugInfoModuleINTEL", 6114> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_debug_module]>
  ];
}
def SPIRV_C_SplitBarrierINTEL                           : I32EnumAttrCase<"SplitBarrierINTEL", 6141> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_split_barrier]>
  ];
}
def SPIRV_C_GroupUniformArithmeticKHR                   : I32EnumAttrCase<"GroupUniformArithmeticKHR", 6400> {
  list<Availability> availability = [
    Extension<[SPV_KHR_uniform_group_instructions]>
  ];
}
def SPIRV_C_Shader                                      : I32EnumAttrCase<"Shader", 1> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Matrix];
}
def SPIRV_C_Vector16                                    : I32EnumAttrCase<"Vector16", 7> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_Float16Buffer                               : I32EnumAttrCase<"Float16Buffer", 8> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_Int64Atomics                                : I32EnumAttrCase<"Int64Atomics", 12> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Int64];
}
def SPIRV_C_ImageBasic                                  : I32EnumAttrCase<"ImageBasic", 13> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_Pipes                                       : I32EnumAttrCase<"Pipes", 17> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_DeviceEnqueue                               : I32EnumAttrCase<"DeviceEnqueue", 19> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_LiteralSampler                              : I32EnumAttrCase<"LiteralSampler", 20> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
}
def SPIRV_C_GenericPointer                              : I32EnumAttrCase<"GenericPointer", 38> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Addresses];
}
def SPIRV_C_Image1D                                     : I32EnumAttrCase<"Image1D", 44> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Sampled1D];
}
def SPIRV_C_ImageBuffer                                 : I32EnumAttrCase<"ImageBuffer", 47> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampledBuffer];
}
def SPIRV_C_NamedBarrier                                : I32EnumAttrCase<"NamedBarrier", 59> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>
  ];
}
def SPIRV_C_GroupNonUniformVote                         : I32EnumAttrCase<"GroupNonUniformVote", 62> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformArithmetic                   : I32EnumAttrCase<"GroupNonUniformArithmetic", 63> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformBallot                       : I32EnumAttrCase<"GroupNonUniformBallot", 64> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformShuffle                      : I32EnumAttrCase<"GroupNonUniformShuffle", 65> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformShuffleRelative              : I32EnumAttrCase<"GroupNonUniformShuffleRelative", 66> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformClustered                    : I32EnumAttrCase<"GroupNonUniformClustered", 67> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_GroupNonUniformQuad                         : I32EnumAttrCase<"GroupNonUniformQuad", 68> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>
  ];
}
def SPIRV_C_StorageUniform16                            : I32EnumAttrCase<"StorageUniform16", 4434> {
  list<I32EnumAttrCase> implies = [SPIRV_C_StorageBuffer16BitAccess];
  list<Availability> availability = [
    Extension<[SPV_KHR_16bit_storage]>
  ];
}
def SPIRV_C_UniformAndStorageBuffer8BitAccess           : I32EnumAttrCase<"UniformAndStorageBuffer8BitAccess", 4449> {
  list<I32EnumAttrCase> implies = [SPIRV_C_StorageBuffer8BitAccess];
  list<Availability> availability = [
    Extension<[SPV_KHR_8bit_storage]>
  ];
}
def SPIRV_C_UniformTexelBufferArrayDynamicIndexing      : I32EnumAttrCase<"UniformTexelBufferArrayDynamicIndexing", 5304> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampledBuffer];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_VectorComputeINTEL                          : I32EnumAttrCase<"VectorComputeINTEL", 5617> {
  list<I32EnumAttrCase> implies = [SPIRV_C_VectorAnyINTEL];
  list<Availability> availability = [
    Extension<[SPV_INTEL_vector_compute]>
  ];
}
def SPIRV_C_FPFastMathModeINTEL                         : I32EnumAttrCase<"FPFastMathModeINTEL", 5837> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Kernel];
  list<Availability> availability = [
    Extension<[SPV_INTEL_fp_fast_math_mode]>
  ];
}
def SPIRV_C_GroupNonUniformRotateKHR                    : I32EnumAttrCase<"GroupNonUniformRotateKHR", 6026> {
  list<I32EnumAttrCase> implies = [SPIRV_C_GroupNonUniform];
  list<Availability> availability = [
    Extension<[SPV_KHR_subgroup_rotate]>
  ];
}
def SPIRV_C_Geometry                                    : I32EnumAttrCase<"Geometry", 2> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_Tessellation                                : I32EnumAttrCase<"Tessellation", 3> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_ImageReadWrite                              : I32EnumAttrCase<"ImageReadWrite", 14> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ImageBasic];
}
def SPIRV_C_ImageMipmap                                 : I32EnumAttrCase<"ImageMipmap", 15> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ImageBasic];
}
def SPIRV_C_AtomicStorage                               : I32EnumAttrCase<"AtomicStorage", 21> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_ImageGatherExtended                         : I32EnumAttrCase<"ImageGatherExtended", 25> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageImageMultisample                     : I32EnumAttrCase<"StorageImageMultisample", 27> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_UniformBufferArrayDynamicIndexing           : I32EnumAttrCase<"UniformBufferArrayDynamicIndexing", 28> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SampledImageArrayDynamicIndexing            : I32EnumAttrCase<"SampledImageArrayDynamicIndexing", 29> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageBufferArrayDynamicIndexing           : I32EnumAttrCase<"StorageBufferArrayDynamicIndexing", 30> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageImageArrayDynamicIndexing            : I32EnumAttrCase<"StorageImageArrayDynamicIndexing", 31> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_ClipDistance                                : I32EnumAttrCase<"ClipDistance", 32> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_CullDistance                                : I32EnumAttrCase<"CullDistance", 33> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SampleRateShading                           : I32EnumAttrCase<"SampleRateShading", 35> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SampledRect                                 : I32EnumAttrCase<"SampledRect", 37> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_InputAttachment                             : I32EnumAttrCase<"InputAttachment", 40> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SparseResidency                             : I32EnumAttrCase<"SparseResidency", 41> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_MinLod                                      : I32EnumAttrCase<"MinLod", 42> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SampledCubeArray                            : I32EnumAttrCase<"SampledCubeArray", 45> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_ImageMSArray                                : I32EnumAttrCase<"ImageMSArray", 48> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageImageExtendedFormats                 : I32EnumAttrCase<"StorageImageExtendedFormats", 49> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_ImageQuery                                  : I32EnumAttrCase<"ImageQuery", 50> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_DerivativeControl                           : I32EnumAttrCase<"DerivativeControl", 51> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_InterpolationFunction                       : I32EnumAttrCase<"InterpolationFunction", 52> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_TransformFeedback                           : I32EnumAttrCase<"TransformFeedback", 53> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageImageReadWithoutFormat               : I32EnumAttrCase<"StorageImageReadWithoutFormat", 55> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_StorageImageWriteWithoutFormat              : I32EnumAttrCase<"StorageImageWriteWithoutFormat", 56> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
}
def SPIRV_C_SubgroupDispatch                            : I32EnumAttrCase<"SubgroupDispatch", 58> {
  list<I32EnumAttrCase> implies = [SPIRV_C_DeviceEnqueue];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>
  ];
}
def SPIRV_C_PipeStorage                                 : I32EnumAttrCase<"PipeStorage", 60> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Pipes];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>
  ];
}
def SPIRV_C_FragmentShadingRateKHR                      : I32EnumAttrCase<"FragmentShadingRateKHR", 4422> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shading_rate]>
  ];
}
def SPIRV_C_DrawParameters                              : I32EnumAttrCase<"DrawParameters", 4427> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>
  ];
}
def SPIRV_C_WorkgroupMemoryExplicitLayoutKHR            : I32EnumAttrCase<"WorkgroupMemoryExplicitLayoutKHR", 4428> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_workgroup_memory_explicit_layout]>
  ];
}
def SPIRV_C_WorkgroupMemoryExplicitLayout16BitAccessKHR : I32EnumAttrCase<"WorkgroupMemoryExplicitLayout16BitAccessKHR", 4430> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_workgroup_memory_explicit_layout]>
  ];
}
def SPIRV_C_MultiView                                   : I32EnumAttrCase<"MultiView", 4439> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_multiview]>
  ];
}
def SPIRV_C_VariablePointersStorageBuffer               : I32EnumAttrCase<"VariablePointersStorageBuffer", 4441> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_variable_pointers]>
  ];
}
def SPIRV_C_RayQueryProvisionalKHR                      : I32EnumAttrCase<"RayQueryProvisionalKHR", 4471> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_query]>
  ];
}
def SPIRV_C_RayQueryKHR                                 : I32EnumAttrCase<"RayQueryKHR", 4472> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_query]>
  ];
}
def SPIRV_C_RayTracingKHR                               : I32EnumAttrCase<"RayTracingKHR", 4479> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing]>
  ];
}
def SPIRV_C_Float16ImageAMD                             : I32EnumAttrCase<"Float16ImageAMD", 5008> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_gpu_shader_half_float_fetch]>
  ];
}
def SPIRV_C_ImageGatherBiasLodAMD                       : I32EnumAttrCase<"ImageGatherBiasLodAMD", 5009> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_texture_gather_bias_lod]>
  ];
}
def SPIRV_C_FragmentMaskAMD                             : I32EnumAttrCase<"FragmentMaskAMD", 5010> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_fragment_mask]>
  ];
}
def SPIRV_C_StencilExportEXT                            : I32EnumAttrCase<"StencilExportEXT", 5013> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>
  ];
}
def SPIRV_C_ImageReadWriteLodAMD                        : I32EnumAttrCase<"ImageReadWriteLodAMD", 5015> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_image_load_store_lod]>
  ];
}
def SPIRV_C_Int64ImageEXT                               : I32EnumAttrCase<"Int64ImageEXT", 5016> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_image_int64]>
  ];
}
def SPIRV_C_ShaderClockKHR                              : I32EnumAttrCase<"ShaderClockKHR", 5055> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_clock]>
  ];
}
def SPIRV_C_FragmentFullyCoveredEXT                     : I32EnumAttrCase<"FragmentFullyCoveredEXT", 5265> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_fully_covered]>
  ];
}
def SPIRV_C_MeshShadingNV                               : I32EnumAttrCase<"MeshShadingNV", 5266> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>
  ];
}
def SPIRV_C_FragmentDensityEXT                          : I32EnumAttrCase<"FragmentDensityEXT", 5291> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>
  ];
}
def SPIRV_C_ShaderNonUniform                            : I32EnumAttrCase<"ShaderNonUniform", 5301> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_RuntimeDescriptorArray                      : I32EnumAttrCase<"RuntimeDescriptorArray", 5302> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_StorageTexelBufferArrayDynamicIndexing      : I32EnumAttrCase<"StorageTexelBufferArrayDynamicIndexing", 5305> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ImageBuffer];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_RayTracingNV                                : I32EnumAttrCase<"RayTracingNV", 5340> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>
  ];
}
def SPIRV_C_RayTracingMotionBlurNV                      : I32EnumAttrCase<"RayTracingMotionBlurNV", 5341> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing_motion_blur]>
  ];
}
def SPIRV_C_PhysicalStorageBufferAddresses              : I32EnumAttrCase<"PhysicalStorageBufferAddresses", 5347> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>
  ];
}
def SPIRV_C_RayTracingProvisionalKHR                    : I32EnumAttrCase<"RayTracingProvisionalKHR", 5353> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing]>
  ];
}
def SPIRV_C_FragmentShaderSampleInterlockEXT            : I32EnumAttrCase<"FragmentShaderSampleInterlockEXT", 5363> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPIRV_C_FragmentShaderShadingRateInterlockEXT       : I32EnumAttrCase<"FragmentShaderShadingRateInterlockEXT", 5372> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPIRV_C_ShaderSMBuiltinsNV                          : I32EnumAttrCase<"ShaderSMBuiltinsNV", 5373> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>
  ];
}
def SPIRV_C_FragmentShaderPixelInterlockEXT             : I32EnumAttrCase<"FragmentShaderPixelInterlockEXT", 5378> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>
  ];
}
def SPIRV_C_DemoteToHelperInvocation                    : I32EnumAttrCase<"DemoteToHelperInvocation", 5379> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_6>
  ];
}
def SPIRV_C_IntegerFunctions2INTEL                      : I32EnumAttrCase<"IntegerFunctions2INTEL", 5584> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Shader];
  list<Availability> availability = [
    Extension<[SPV_INTEL_shader_integer_functions2]>
  ];
}
def SPIRV_C_TessellationPointSize                       : I32EnumAttrCase<"TessellationPointSize", 23> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Tessellation];
}
def SPIRV_C_GeometryPointSize                           : I32EnumAttrCase<"GeometryPointSize", 24> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Geometry];
}
def SPIRV_C_ImageCubeArray                              : I32EnumAttrCase<"ImageCubeArray", 34> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampledCubeArray];
}
def SPIRV_C_ImageRect                                   : I32EnumAttrCase<"ImageRect", 36> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampledRect];
}
def SPIRV_C_GeometryStreams                             : I32EnumAttrCase<"GeometryStreams", 54> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Geometry];
}
def SPIRV_C_MultiViewport                               : I32EnumAttrCase<"MultiViewport", 57> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Geometry];
}
def SPIRV_C_WorkgroupMemoryExplicitLayout8BitAccessKHR  : I32EnumAttrCase<"WorkgroupMemoryExplicitLayout8BitAccessKHR", 4429> {
  list<I32EnumAttrCase> implies = [SPIRV_C_WorkgroupMemoryExplicitLayoutKHR];
  list<Availability> availability = [
    Extension<[SPV_KHR_workgroup_memory_explicit_layout]>
  ];
}
def SPIRV_C_VariablePointers                            : I32EnumAttrCase<"VariablePointers", 4442> {
  list<I32EnumAttrCase> implies = [SPIRV_C_VariablePointersStorageBuffer];
  list<Availability> availability = [
    Extension<[SPV_KHR_variable_pointers]>
  ];
}
def SPIRV_C_RayTraversalPrimitiveCullingKHR             : I32EnumAttrCase<"RayTraversalPrimitiveCullingKHR", 4478> {
  list<I32EnumAttrCase> implies = [SPIRV_C_RayQueryKHR, SPIRV_C_RayTracingKHR];
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_query, SPV_KHR_ray_tracing]>
  ];
}
def SPIRV_C_SampleMaskOverrideCoverageNV                : I32EnumAttrCase<"SampleMaskOverrideCoverageNV", 5249> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampleRateShading];
  list<Availability> availability = [
    Extension<[SPV_NV_sample_mask_override_coverage]>
  ];
}
def SPIRV_C_GeometryShaderPassthroughNV                 : I32EnumAttrCase<"GeometryShaderPassthroughNV", 5251> {
  list<I32EnumAttrCase> implies = [SPIRV_C_Geometry];
  list<Availability> availability = [
    Extension<[SPV_NV_geometry_shader_passthrough]>
  ];
}
def SPIRV_C_PerViewAttributesNV                         : I32EnumAttrCase<"PerViewAttributesNV", 5260> {
  list<I32EnumAttrCase> implies = [SPIRV_C_MultiView];
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes]>
  ];
}
def SPIRV_C_InputAttachmentArrayDynamicIndexing         : I32EnumAttrCase<"InputAttachmentArrayDynamicIndexing", 5303> {
  list<I32EnumAttrCase> implies = [SPIRV_C_InputAttachment];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_UniformBufferArrayNonUniformIndexing        : I32EnumAttrCase<"UniformBufferArrayNonUniformIndexing", 5306> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_SampledImageArrayNonUniformIndexing         : I32EnumAttrCase<"SampledImageArrayNonUniformIndexing", 5307> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_StorageBufferArrayNonUniformIndexing        : I32EnumAttrCase<"StorageBufferArrayNonUniformIndexing", 5308> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_StorageImageArrayNonUniformIndexing         : I32EnumAttrCase<"StorageImageArrayNonUniformIndexing", 5309> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_InputAttachmentArrayNonUniformIndexing      : I32EnumAttrCase<"InputAttachmentArrayNonUniformIndexing", 5310> {
  list<I32EnumAttrCase> implies = [SPIRV_C_InputAttachment, SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_UniformTexelBufferArrayNonUniformIndexing   : I32EnumAttrCase<"UniformTexelBufferArrayNonUniformIndexing", 5311> {
  list<I32EnumAttrCase> implies = [SPIRV_C_SampledBuffer, SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_StorageTexelBufferArrayNonUniformIndexing   : I32EnumAttrCase<"StorageTexelBufferArrayNonUniformIndexing", 5312> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ImageBuffer, SPIRV_C_ShaderNonUniform];
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>
  ];
}
def SPIRV_C_ShaderViewportIndexLayerEXT                 : I32EnumAttrCase<"ShaderViewportIndexLayerEXT", 5254> {
  list<I32EnumAttrCase> implies = [SPIRV_C_MultiViewport];
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_viewport_index_layer]>
  ];
}
def SPIRV_C_ShaderViewportMaskNV                        : I32EnumAttrCase<"ShaderViewportMaskNV", 5255> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderViewportIndexLayerEXT];
  list<Availability> availability = [
    Extension<[SPV_NV_viewport_array2]>
  ];
}
def SPIRV_C_ShaderStereoViewNV                          : I32EnumAttrCase<"ShaderStereoViewNV", 5259> {
  list<I32EnumAttrCase> implies = [SPIRV_C_ShaderViewportMaskNV];
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>
  ];
}

def SPIRV_C_JointMatrixINTEL                         : I32EnumAttrCase<"JointMatrixINTEL", 6118> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_joint_matrix]>
  ];
}

def SPIRV_C_Bfloat16ConversionINTEL                         : I32EnumAttrCase<"Bfloat16ConversionINTEL", 6115> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_bfloat16_conversion]>
  ];
}

def SPIRV_CapabilityAttr :
    SPIRV_I32EnumAttr<"Capability", "valid SPIR-V Capability", "capability", [
      SPIRV_C_Matrix, SPIRV_C_Addresses, SPIRV_C_Linkage, SPIRV_C_Kernel, SPIRV_C_Float16,
      SPIRV_C_Float64, SPIRV_C_Int64, SPIRV_C_Groups, SPIRV_C_Int16, SPIRV_C_Int8,
      SPIRV_C_Sampled1D, SPIRV_C_SampledBuffer, SPIRV_C_GroupNonUniform, SPIRV_C_ShaderLayer,
      SPIRV_C_ShaderViewportIndex, SPIRV_C_UniformDecoration, SPIRV_C_SubgroupBallotKHR,
      SPIRV_C_SubgroupVoteKHR, SPIRV_C_StorageBuffer16BitAccess,
      SPIRV_C_StoragePushConstant16, SPIRV_C_StorageInputOutput16, SPIRV_C_DeviceGroup,
      SPIRV_C_AtomicStorageOps, SPIRV_C_SampleMaskPostDepthCoverage,
      SPIRV_C_StorageBuffer8BitAccess, SPIRV_C_StoragePushConstant8,
      SPIRV_C_DenormPreserve, SPIRV_C_DenormFlushToZero, SPIRV_C_SignedZeroInfNanPreserve,
      SPIRV_C_RoundingModeRTE, SPIRV_C_RoundingModeRTZ, SPIRV_C_ImageFootprintNV,
      SPIRV_C_FragmentBarycentricKHR, SPIRV_C_ComputeDerivativeGroupQuadsNV,
      SPIRV_C_GroupNonUniformPartitionedNV, SPIRV_C_VulkanMemoryModel,
      SPIRV_C_VulkanMemoryModelDeviceScope, SPIRV_C_ComputeDerivativeGroupLinearNV,
      SPIRV_C_BindlessTextureNV, SPIRV_C_SubgroupShuffleINTEL,
      SPIRV_C_SubgroupBufferBlockIOINTEL, SPIRV_C_SubgroupImageBlockIOINTEL,
      SPIRV_C_SubgroupImageMediaBlockIOINTEL, SPIRV_C_RoundToInfinityINTEL,
      SPIRV_C_FloatingPointModeINTEL, SPIRV_C_FunctionPointersINTEL,
      SPIRV_C_IndirectReferencesINTEL, SPIRV_C_AsmINTEL, SPIRV_C_AtomicFloat32MinMaxEXT,
      SPIRV_C_AtomicFloat64MinMaxEXT, SPIRV_C_AtomicFloat16MinMaxEXT,
      SPIRV_C_VectorAnyINTEL, SPIRV_C_ExpectAssumeKHR,
      SPIRV_C_SubgroupAvcMotionEstimationINTEL,
      SPIRV_C_SubgroupAvcMotionEstimationIntraINTEL,
      SPIRV_C_SubgroupAvcMotionEstimationChromaINTEL, SPIRV_C_VariableLengthArrayINTEL,
      SPIRV_C_FunctionFloatControlINTEL, SPIRV_C_FPGAMemoryAttributesINTEL,
      SPIRV_C_ArbitraryPrecisionIntegersINTEL,
      SPIRV_C_ArbitraryPrecisionFloatingPointINTEL,
      SPIRV_C_UnstructuredLoopControlsINTEL, SPIRV_C_FPGALoopControlsINTEL,
      SPIRV_C_KernelAttributesINTEL, SPIRV_C_FPGAKernelAttributesINTEL,
      SPIRV_C_FPGAMemoryAccessesINTEL, SPIRV_C_FPGAClusterAttributesINTEL,
      SPIRV_C_LoopFuseINTEL, SPIRV_C_MemoryAccessAliasingINTEL,
      SPIRV_C_FPGABufferLocationINTEL, SPIRV_C_ArbitraryPrecisionFixedPointINTEL,
      SPIRV_C_USMStorageClassesINTEL, SPIRV_C_IOPipesINTEL, SPIRV_C_BlockingPipesINTEL,
      SPIRV_C_FPGARegINTEL, SPIRV_C_DotProductInputAll,
      SPIRV_C_DotProductInput4x8BitPacked, SPIRV_C_DotProduct, SPIRV_C_RayCullMaskKHR,
      SPIRV_C_CooperativeMatrixKHR,
      SPIRV_C_BitInstructions, SPIRV_C_AtomicFloat32AddEXT, SPIRV_C_AtomicFloat64AddEXT,
      SPIRV_C_LongConstantCompositeINTEL, SPIRV_C_OptNoneINTEL,
      SPIRV_C_AtomicFloat16AddEXT, SPIRV_C_DebugInfoModuleINTEL, SPIRV_C_SplitBarrierINTEL,
      SPIRV_C_GroupUniformArithmeticKHR, SPIRV_C_Shader, SPIRV_C_Vector16,
      SPIRV_C_Float16Buffer, SPIRV_C_Int64Atomics, SPIRV_C_ImageBasic, SPIRV_C_Pipes,
      SPIRV_C_DeviceEnqueue, SPIRV_C_LiteralSampler, SPIRV_C_GenericPointer, SPIRV_C_Image1D,
      SPIRV_C_ImageBuffer, SPIRV_C_NamedBarrier, SPIRV_C_GroupNonUniformVote,
      SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformBallot,
      SPIRV_C_GroupNonUniformShuffle, SPIRV_C_GroupNonUniformShuffleRelative,
      SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformQuad,
      SPIRV_C_StorageUniform16, SPIRV_C_UniformAndStorageBuffer8BitAccess,
      SPIRV_C_UniformTexelBufferArrayDynamicIndexing, SPIRV_C_VectorComputeINTEL,
      SPIRV_C_FPFastMathModeINTEL, SPIRV_C_DotProductInput4x8Bit,
      SPIRV_C_GroupNonUniformRotateKHR, SPIRV_C_Geometry, SPIRV_C_Tessellation,
      SPIRV_C_ImageReadWrite, SPIRV_C_ImageMipmap, SPIRV_C_AtomicStorage,
      SPIRV_C_ImageGatherExtended, SPIRV_C_StorageImageMultisample,
      SPIRV_C_UniformBufferArrayDynamicIndexing,
      SPIRV_C_SampledImageArrayDynamicIndexing,
      SPIRV_C_StorageBufferArrayDynamicIndexing,
      SPIRV_C_StorageImageArrayDynamicIndexing, SPIRV_C_ClipDistance, SPIRV_C_CullDistance,
      SPIRV_C_SampleRateShading, SPIRV_C_SampledRect, SPIRV_C_InputAttachment,
      SPIRV_C_SparseResidency, SPIRV_C_MinLod, SPIRV_C_SampledCubeArray,
      SPIRV_C_ImageMSArray, SPIRV_C_StorageImageExtendedFormats, SPIRV_C_ImageQuery,
      SPIRV_C_DerivativeControl, SPIRV_C_InterpolationFunction, SPIRV_C_TransformFeedback,
      SPIRV_C_StorageImageReadWithoutFormat, SPIRV_C_StorageImageWriteWithoutFormat,
      SPIRV_C_SubgroupDispatch, SPIRV_C_PipeStorage, SPIRV_C_FragmentShadingRateKHR,
      SPIRV_C_DrawParameters, SPIRV_C_WorkgroupMemoryExplicitLayoutKHR,
      SPIRV_C_WorkgroupMemoryExplicitLayout16BitAccessKHR, SPIRV_C_MultiView,
      SPIRV_C_VariablePointersStorageBuffer, SPIRV_C_RayQueryProvisionalKHR,
      SPIRV_C_RayQueryKHR, SPIRV_C_RayTracingKHR, SPIRV_C_Float16ImageAMD,
      SPIRV_C_ImageGatherBiasLodAMD, SPIRV_C_FragmentMaskAMD, SPIRV_C_StencilExportEXT,
      SPIRV_C_ImageReadWriteLodAMD, SPIRV_C_Int64ImageEXT, SPIRV_C_ShaderClockKHR,
      SPIRV_C_FragmentFullyCoveredEXT, SPIRV_C_MeshShadingNV, SPIRV_C_FragmentDensityEXT,
      SPIRV_C_ShaderNonUniform, SPIRV_C_RuntimeDescriptorArray,
      SPIRV_C_StorageTexelBufferArrayDynamicIndexing, SPIRV_C_RayTracingNV,
      SPIRV_C_RayTracingMotionBlurNV, SPIRV_C_PhysicalStorageBufferAddresses,
      SPIRV_C_RayTracingProvisionalKHR,
      SPIRV_C_FragmentShaderSampleInterlockEXT,
      SPIRV_C_FragmentShaderShadingRateInterlockEXT, SPIRV_C_ShaderSMBuiltinsNV,
      SPIRV_C_FragmentShaderPixelInterlockEXT, SPIRV_C_DemoteToHelperInvocation,
      SPIRV_C_IntegerFunctions2INTEL, SPIRV_C_TessellationPointSize,
      SPIRV_C_GeometryPointSize, SPIRV_C_ImageCubeArray, SPIRV_C_ImageRect,
      SPIRV_C_GeometryStreams, SPIRV_C_MultiViewport,
      SPIRV_C_WorkgroupMemoryExplicitLayout8BitAccessKHR, SPIRV_C_VariablePointers,
      SPIRV_C_RayTraversalPrimitiveCullingKHR, SPIRV_C_SampleMaskOverrideCoverageNV,
      SPIRV_C_GeometryShaderPassthroughNV, SPIRV_C_PerViewAttributesNV,
      SPIRV_C_InputAttachmentArrayDynamicIndexing,
      SPIRV_C_UniformBufferArrayNonUniformIndexing,
      SPIRV_C_SampledImageArrayNonUniformIndexing,
      SPIRV_C_StorageBufferArrayNonUniformIndexing,
      SPIRV_C_StorageImageArrayNonUniformIndexing,
      SPIRV_C_InputAttachmentArrayNonUniformIndexing,
      SPIRV_C_UniformTexelBufferArrayNonUniformIndexing,
      SPIRV_C_StorageTexelBufferArrayNonUniformIndexing,
      SPIRV_C_ShaderViewportIndexLayerEXT, SPIRV_C_ShaderViewportMaskNV,
      SPIRV_C_ShaderStereoViewNV, SPIRV_C_JointMatrixINTEL, SPIRV_C_Bfloat16ConversionINTEL
    ]>;

def SPIRV_AM_Logical                 : I32EnumAttrCase<"Logical", 0>;
def SPIRV_AM_Physical32              : I32EnumAttrCase<"Physical32", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Addresses]>
  ];
}
def SPIRV_AM_Physical64              : I32EnumAttrCase<"Physical64", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Addresses]>
  ];
}
def SPIRV_AM_PhysicalStorageBuffer64 : I32EnumAttrCase<"PhysicalStorageBuffer64", 5348> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPIRV_C_PhysicalStorageBufferAddresses]>
  ];
}

def SPIRV_AddressingModelAttr :
    SPIRV_I32EnumAttr<"AddressingModel", "valid SPIR-V AddressingModel", "addressing_model", [
      SPIRV_AM_Logical, SPIRV_AM_Physical32, SPIRV_AM_Physical64,
      SPIRV_AM_PhysicalStorageBuffer64
    ]>;

def SPIRV_BI_Position                    : I32EnumAttrCase<"Position", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_PointSize                   : I32EnumAttrCase<"PointSize", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_ClipDistance                : I32EnumAttrCase<"ClipDistance", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ClipDistance]>
  ];
}
def SPIRV_BI_CullDistance                : I32EnumAttrCase<"CullDistance", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_CullDistance]>
  ];
}
def SPIRV_BI_VertexId                    : I32EnumAttrCase<"VertexId", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_InstanceId                  : I32EnumAttrCase<"InstanceId", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_PrimitiveId                 : I32EnumAttrCase<"PrimitiveId", 7> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_MeshShadingNV, SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV, SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_InvocationId                : I32EnumAttrCase<"InvocationId", 8> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_Layer                       : I32EnumAttrCase<"Layer", 9> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_MeshShadingNV, SPIRV_C_ShaderLayer, SPIRV_C_ShaderViewportIndexLayerEXT]>
  ];
}
def SPIRV_BI_ViewportIndex               : I32EnumAttrCase<"ViewportIndex", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MeshShadingNV, SPIRV_C_MultiViewport, SPIRV_C_ShaderViewportIndex, SPIRV_C_ShaderViewportIndexLayerEXT]>
  ];
}
def SPIRV_BI_TessLevelOuter              : I32EnumAttrCase<"TessLevelOuter", 11> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_TessLevelInner              : I32EnumAttrCase<"TessLevelInner", 12> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_TessCoord                   : I32EnumAttrCase<"TessCoord", 13> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_PatchVertices               : I32EnumAttrCase<"PatchVertices", 14> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_BI_FragCoord                   : I32EnumAttrCase<"FragCoord", 15> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_PointCoord                  : I32EnumAttrCase<"PointCoord", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_FrontFacing                 : I32EnumAttrCase<"FrontFacing", 17> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_SampleId                    : I32EnumAttrCase<"SampleId", 18> {
  list<Availability> availability = [
    Capability<[SPIRV_C_SampleRateShading]>
  ];
}
def SPIRV_BI_SamplePosition              : I32EnumAttrCase<"SamplePosition", 19> {
  list<Availability> availability = [
    Capability<[SPIRV_C_SampleRateShading]>
  ];
}
def SPIRV_BI_SampleMask                  : I32EnumAttrCase<"SampleMask", 20> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_FragDepth                   : I32EnumAttrCase<"FragDepth", 22> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_HelperInvocation            : I32EnumAttrCase<"HelperInvocation", 23> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_NumWorkgroups               : I32EnumAttrCase<"NumWorkgroups", 24>;
def SPIRV_BI_WorkgroupSize               : I32EnumAttrCase<"WorkgroupSize", 25>;
def SPIRV_BI_WorkgroupId                 : I32EnumAttrCase<"WorkgroupId", 26>;
def SPIRV_BI_LocalInvocationId           : I32EnumAttrCase<"LocalInvocationId", 27>;
def SPIRV_BI_GlobalInvocationId          : I32EnumAttrCase<"GlobalInvocationId", 28>;
def SPIRV_BI_LocalInvocationIndex        : I32EnumAttrCase<"LocalInvocationIndex", 29>;
def SPIRV_BI_WorkDim                     : I32EnumAttrCase<"WorkDim", 30> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_GlobalSize                  : I32EnumAttrCase<"GlobalSize", 31> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_EnqueuedWorkgroupSize       : I32EnumAttrCase<"EnqueuedWorkgroupSize", 32> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_GlobalOffset                : I32EnumAttrCase<"GlobalOffset", 33> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_GlobalLinearId              : I32EnumAttrCase<"GlobalLinearId", 34> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_SubgroupSize                : I32EnumAttrCase<"SubgroupSize", 36> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniform, SPIRV_C_Kernel, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_SubgroupMaxSize             : I32EnumAttrCase<"SubgroupMaxSize", 37> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_NumSubgroups                : I32EnumAttrCase<"NumSubgroups", 38> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniform, SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_NumEnqueuedSubgroups        : I32EnumAttrCase<"NumEnqueuedSubgroups", 39> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_SubgroupId                  : I32EnumAttrCase<"SubgroupId", 40> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniform, SPIRV_C_Kernel]>
  ];
}
def SPIRV_BI_SubgroupLocalInvocationId   : I32EnumAttrCase<"SubgroupLocalInvocationId", 41> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniform, SPIRV_C_Kernel, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_VertexIndex                 : I32EnumAttrCase<"VertexIndex", 42> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_InstanceIndex               : I32EnumAttrCase<"InstanceIndex", 43> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_BI_SubgroupEqMask              : I32EnumAttrCase<"SubgroupEqMask", 4416> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformBallot, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_SubgroupGeMask              : I32EnumAttrCase<"SubgroupGeMask", 4417> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformBallot, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_SubgroupGtMask              : I32EnumAttrCase<"SubgroupGtMask", 4418> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformBallot, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_SubgroupLeMask              : I32EnumAttrCase<"SubgroupLeMask", 4419> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformBallot, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_SubgroupLtMask              : I32EnumAttrCase<"SubgroupLtMask", 4420> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformBallot, SPIRV_C_SubgroupBallotKHR]>
  ];
}
def SPIRV_BI_BaseVertex                  : I32EnumAttrCase<"BaseVertex", 4424> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>,
    Capability<[SPIRV_C_DrawParameters]>
  ];
}
def SPIRV_BI_BaseInstance                : I32EnumAttrCase<"BaseInstance", 4425> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters]>,
    Capability<[SPIRV_C_DrawParameters]>
  ];
}
def SPIRV_BI_DrawIndex                   : I32EnumAttrCase<"DrawIndex", 4426> {
  list<Availability> availability = [
    Extension<[SPV_KHR_shader_draw_parameters, SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_DrawParameters, SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_PrimitiveShadingRateKHR     : I32EnumAttrCase<"PrimitiveShadingRateKHR", 4432> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shading_rate]>,
    Capability<[SPIRV_C_FragmentShadingRateKHR]>
  ];
}
def SPIRV_BI_DeviceIndex                 : I32EnumAttrCase<"DeviceIndex", 4438> {
  list<Availability> availability = [
    Extension<[SPV_KHR_device_group]>,
    Capability<[SPIRV_C_DeviceGroup]>
  ];
}
def SPIRV_BI_ViewIndex                   : I32EnumAttrCase<"ViewIndex", 4440> {
  list<Availability> availability = [
    Extension<[SPV_KHR_multiview]>,
    Capability<[SPIRV_C_MultiView]>
  ];
}
def SPIRV_BI_ShadingRateKHR              : I32EnumAttrCase<"ShadingRateKHR", 4444> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shading_rate]>,
    Capability<[SPIRV_C_FragmentShadingRateKHR]>
  ];
}
def SPIRV_BI_BaryCoordNoPerspAMD         : I32EnumAttrCase<"BaryCoordNoPerspAMD", 4992> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordNoPerspCentroidAMD : I32EnumAttrCase<"BaryCoordNoPerspCentroidAMD", 4993> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordNoPerspSampleAMD   : I32EnumAttrCase<"BaryCoordNoPerspSampleAMD", 4994> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordSmoothAMD          : I32EnumAttrCase<"BaryCoordSmoothAMD", 4995> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordSmoothCentroidAMD  : I32EnumAttrCase<"BaryCoordSmoothCentroidAMD", 4996> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordSmoothSampleAMD    : I32EnumAttrCase<"BaryCoordSmoothSampleAMD", 4997> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_BaryCoordPullModelAMD       : I32EnumAttrCase<"BaryCoordPullModelAMD", 4998> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_BI_FragStencilRefEXT           : I32EnumAttrCase<"FragStencilRefEXT", 5014> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_BI_ViewportMaskNV              : I32EnumAttrCase<"ViewportMaskNV", 5253> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader, SPV_NV_viewport_array2]>,
    Capability<[SPIRV_C_MeshShadingNV, SPIRV_C_ShaderViewportMaskNV]>
  ];
}
def SPIRV_BI_SecondaryPositionNV         : I32EnumAttrCase<"SecondaryPositionNV", 5257> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPIRV_C_ShaderStereoViewNV]>
  ];
}
def SPIRV_BI_SecondaryViewportMaskNV     : I32EnumAttrCase<"SecondaryViewportMaskNV", 5258> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPIRV_C_ShaderStereoViewNV]>
  ];
}
def SPIRV_BI_PositionPerViewNV           : I32EnumAttrCase<"PositionPerViewNV", 5261> {
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes, SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV, SPIRV_C_PerViewAttributesNV]>
  ];
}
def SPIRV_BI_ViewportMaskPerViewNV       : I32EnumAttrCase<"ViewportMaskPerViewNV", 5262> {
  list<Availability> availability = [
    Extension<[SPV_NVX_multiview_per_view_attributes, SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV, SPIRV_C_PerViewAttributesNV]>
  ];
}
def SPIRV_BI_FullyCoveredEXT             : I32EnumAttrCase<"FullyCoveredEXT", 5264> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_fully_covered]>,
    Capability<[SPIRV_C_FragmentFullyCoveredEXT]>
  ];
}
def SPIRV_BI_TaskCountNV                 : I32EnumAttrCase<"TaskCountNV", 5274> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_PrimitiveCountNV            : I32EnumAttrCase<"PrimitiveCountNV", 5275> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_PrimitiveIndicesNV          : I32EnumAttrCase<"PrimitiveIndicesNV", 5276> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_ClipDistancePerViewNV       : I32EnumAttrCase<"ClipDistancePerViewNV", 5277> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_CullDistancePerViewNV       : I32EnumAttrCase<"CullDistancePerViewNV", 5278> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_LayerPerViewNV              : I32EnumAttrCase<"LayerPerViewNV", 5279> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_MeshViewCountNV             : I32EnumAttrCase<"MeshViewCountNV", 5280> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_MeshViewIndicesNV           : I32EnumAttrCase<"MeshViewIndicesNV", 5281> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_BI_BaryCoordKHR                : I32EnumAttrCase<"BaryCoordKHR", 5286> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shader_barycentric, SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPIRV_C_FragmentBarycentricKHR]>
  ];
}
def SPIRV_BI_BaryCoordNoPerspKHR         : I32EnumAttrCase<"BaryCoordNoPerspKHR", 5287> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shader_barycentric, SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPIRV_C_FragmentBarycentricKHR]>
  ];
}
def SPIRV_BI_FragSizeEXT                 : I32EnumAttrCase<"FragSizeEXT", 5292> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>,
    Capability<[SPIRV_C_FragmentDensityEXT]>
  ];
}
def SPIRV_BI_FragInvocationCountEXT      : I32EnumAttrCase<"FragInvocationCountEXT", 5293> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_invocation_density, SPV_NV_shading_rate]>,
    Capability<[SPIRV_C_FragmentDensityEXT]>
  ];
}
def SPIRV_BI_LaunchIdKHR                 : I32EnumAttrCase<"LaunchIdKHR", 5319> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_LaunchSizeKHR               : I32EnumAttrCase<"LaunchSizeKHR", 5320> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_WorldRayOriginKHR           : I32EnumAttrCase<"WorldRayOriginKHR", 5321> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_WorldRayDirectionKHR        : I32EnumAttrCase<"WorldRayDirectionKHR", 5322> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_ObjectRayOriginKHR          : I32EnumAttrCase<"ObjectRayOriginKHR", 5323> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_ObjectRayDirectionKHR       : I32EnumAttrCase<"ObjectRayDirectionKHR", 5324> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_RayTminKHR                  : I32EnumAttrCase<"RayTminKHR", 5325> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_RayTmaxKHR                  : I32EnumAttrCase<"RayTmaxKHR", 5326> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_InstanceCustomIndexKHR      : I32EnumAttrCase<"InstanceCustomIndexKHR", 5327> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_ObjectToWorldKHR            : I32EnumAttrCase<"ObjectToWorldKHR", 5330> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_WorldToObjectKHR            : I32EnumAttrCase<"WorldToObjectKHR", 5331> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_HitTNV                      : I32EnumAttrCase<"HitTNV", 5332> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_HitKindKHR                  : I32EnumAttrCase<"HitKindKHR", 5333> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_CurrentRayTimeNV            : I32EnumAttrCase<"CurrentRayTimeNV", 5334> {
  list<Availability> availability = [
    Extension<[SPV_NV_ray_tracing_motion_blur]>,
    Capability<[SPIRV_C_RayTracingMotionBlurNV]>
  ];
}
def SPIRV_BI_IncomingRayFlagsKHR         : I32EnumAttrCase<"IncomingRayFlagsKHR", 5351> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_BI_RayGeometryIndexKHR         : I32EnumAttrCase<"RayGeometryIndexKHR", 5352> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR]>
  ];
}
def SPIRV_BI_WarpsPerSMNV                : I32EnumAttrCase<"WarpsPerSMNV", 5374> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPIRV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPIRV_BI_SMCountNV                   : I32EnumAttrCase<"SMCountNV", 5375> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPIRV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPIRV_BI_WarpIDNV                    : I32EnumAttrCase<"WarpIDNV", 5376> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPIRV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPIRV_BI_SMIDNV                      : I32EnumAttrCase<"SMIDNV", 5377> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_sm_builtins]>,
    Capability<[SPIRV_C_ShaderSMBuiltinsNV]>
  ];
}
def SPIRV_BI_CullMaskKHR                 : I32EnumAttrCase<"CullMaskKHR", 6021> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_cull_mask]>,
    Capability<[SPIRV_C_RayCullMaskKHR]>
  ];
}

def SPIRV_BuiltInAttr :
    SPIRV_I32EnumAttr<"BuiltIn", "valid SPIR-V BuiltIn", "built_in", [
      SPIRV_BI_Position, SPIRV_BI_PointSize, SPIRV_BI_ClipDistance, SPIRV_BI_CullDistance,
      SPIRV_BI_VertexId, SPIRV_BI_InstanceId, SPIRV_BI_PrimitiveId, SPIRV_BI_InvocationId,
      SPIRV_BI_Layer, SPIRV_BI_ViewportIndex, SPIRV_BI_TessLevelOuter,
      SPIRV_BI_TessLevelInner, SPIRV_BI_TessCoord, SPIRV_BI_PatchVertices,
      SPIRV_BI_FragCoord, SPIRV_BI_PointCoord, SPIRV_BI_FrontFacing, SPIRV_BI_SampleId,
      SPIRV_BI_SamplePosition, SPIRV_BI_SampleMask, SPIRV_BI_FragDepth,
      SPIRV_BI_HelperInvocation, SPIRV_BI_NumWorkgroups, SPIRV_BI_WorkgroupSize,
      SPIRV_BI_WorkgroupId, SPIRV_BI_LocalInvocationId, SPIRV_BI_GlobalInvocationId,
      SPIRV_BI_LocalInvocationIndex, SPIRV_BI_WorkDim, SPIRV_BI_GlobalSize,
      SPIRV_BI_EnqueuedWorkgroupSize, SPIRV_BI_GlobalOffset, SPIRV_BI_GlobalLinearId,
      SPIRV_BI_SubgroupSize, SPIRV_BI_SubgroupMaxSize, SPIRV_BI_NumSubgroups,
      SPIRV_BI_NumEnqueuedSubgroups, SPIRV_BI_SubgroupId,
      SPIRV_BI_SubgroupLocalInvocationId, SPIRV_BI_VertexIndex, SPIRV_BI_InstanceIndex,
      SPIRV_BI_SubgroupEqMask, SPIRV_BI_SubgroupGeMask, SPIRV_BI_SubgroupGtMask,
      SPIRV_BI_SubgroupLeMask, SPIRV_BI_SubgroupLtMask, SPIRV_BI_BaseVertex,
      SPIRV_BI_BaseInstance, SPIRV_BI_DrawIndex, SPIRV_BI_PrimitiveShadingRateKHR,
      SPIRV_BI_DeviceIndex, SPIRV_BI_ViewIndex, SPIRV_BI_ShadingRateKHR,
      SPIRV_BI_BaryCoordNoPerspAMD, SPIRV_BI_BaryCoordNoPerspCentroidAMD,
      SPIRV_BI_BaryCoordNoPerspSampleAMD, SPIRV_BI_BaryCoordSmoothAMD,
      SPIRV_BI_BaryCoordSmoothCentroidAMD, SPIRV_BI_BaryCoordSmoothSampleAMD,
      SPIRV_BI_BaryCoordPullModelAMD, SPIRV_BI_FragStencilRefEXT, SPIRV_BI_ViewportMaskNV,
      SPIRV_BI_SecondaryPositionNV, SPIRV_BI_SecondaryViewportMaskNV,
      SPIRV_BI_PositionPerViewNV, SPIRV_BI_ViewportMaskPerViewNV, SPIRV_BI_FullyCoveredEXT,
      SPIRV_BI_TaskCountNV, SPIRV_BI_PrimitiveCountNV, SPIRV_BI_PrimitiveIndicesNV,
      SPIRV_BI_ClipDistancePerViewNV, SPIRV_BI_CullDistancePerViewNV,
      SPIRV_BI_LayerPerViewNV, SPIRV_BI_MeshViewCountNV, SPIRV_BI_MeshViewIndicesNV,
      SPIRV_BI_BaryCoordKHR, SPIRV_BI_BaryCoordNoPerspKHR, SPIRV_BI_FragSizeEXT,
      SPIRV_BI_FragInvocationCountEXT, SPIRV_BI_LaunchIdKHR, SPIRV_BI_LaunchSizeKHR,
      SPIRV_BI_WorldRayOriginKHR, SPIRV_BI_WorldRayDirectionKHR,
      SPIRV_BI_ObjectRayOriginKHR, SPIRV_BI_ObjectRayDirectionKHR, SPIRV_BI_RayTminKHR,
      SPIRV_BI_RayTmaxKHR, SPIRV_BI_InstanceCustomIndexKHR, SPIRV_BI_ObjectToWorldKHR,
      SPIRV_BI_WorldToObjectKHR, SPIRV_BI_HitTNV, SPIRV_BI_HitKindKHR,
      SPIRV_BI_CurrentRayTimeNV, SPIRV_BI_IncomingRayFlagsKHR,
      SPIRV_BI_RayGeometryIndexKHR, SPIRV_BI_WarpsPerSMNV, SPIRV_BI_SMCountNV,
      SPIRV_BI_WarpIDNV, SPIRV_BI_SMIDNV, SPIRV_BI_CullMaskKHR
    ]>;

def SPIRV_D_RelaxedPrecision                   : I32EnumAttrCase<"RelaxedPrecision", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_SpecId                             : I32EnumAttrCase<"SpecId", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel, SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Block                              : I32EnumAttrCase<"Block", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_BufferBlock                        : I32EnumAttrCase<"BufferBlock", 3> {
  list<Availability> availability = [
    MaxVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_RowMajor                           : I32EnumAttrCase<"RowMajor", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Matrix]>
  ];
}
def SPIRV_D_ColMajor                           : I32EnumAttrCase<"ColMajor", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Matrix]>
  ];
}
def SPIRV_D_ArrayStride                        : I32EnumAttrCase<"ArrayStride", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_MatrixStride                       : I32EnumAttrCase<"MatrixStride", 7> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Matrix]>
  ];
}
def SPIRV_D_GLSLShared                         : I32EnumAttrCase<"GLSLShared", 8> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_GLSLPacked                         : I32EnumAttrCase<"GLSLPacked", 9> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_CPacked                            : I32EnumAttrCase<"CPacked", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_BuiltIn                            : I32EnumAttrCase<"BuiltIn", 11>;
def SPIRV_D_NoPerspective                      : I32EnumAttrCase<"NoPerspective", 13> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Flat                               : I32EnumAttrCase<"Flat", 14> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Patch                              : I32EnumAttrCase<"Patch", 15> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_D_Centroid                           : I32EnumAttrCase<"Centroid", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Sample                             : I32EnumAttrCase<"Sample", 17> {
  list<Availability> availability = [
    Capability<[SPIRV_C_SampleRateShading]>
  ];
}
def SPIRV_D_Invariant                          : I32EnumAttrCase<"Invariant", 18> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Restrict                           : I32EnumAttrCase<"Restrict", 19>;
def SPIRV_D_Aliased                            : I32EnumAttrCase<"Aliased", 20>;
def SPIRV_D_Volatile                           : I32EnumAttrCase<"Volatile", 21>;
def SPIRV_D_Constant                           : I32EnumAttrCase<"Constant", 22> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_Coherent                           : I32EnumAttrCase<"Coherent", 23>;
def SPIRV_D_NonWritable                        : I32EnumAttrCase<"NonWritable", 24>;
def SPIRV_D_NonReadable                        : I32EnumAttrCase<"NonReadable", 25>;
def SPIRV_D_Uniform                            : I32EnumAttrCase<"Uniform", 26> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader, SPIRV_C_UniformDecoration]>
  ];
}
def SPIRV_D_UniformId                          : I32EnumAttrCase<"UniformId", 27> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>,
    Capability<[SPIRV_C_Shader, SPIRV_C_UniformDecoration]>
  ];
}
def SPIRV_D_SaturatedConversion                : I32EnumAttrCase<"SaturatedConversion", 28> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_Stream                             : I32EnumAttrCase<"Stream", 29> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GeometryStreams]>
  ];
}
def SPIRV_D_Location                           : I32EnumAttrCase<"Location", 30> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Component                          : I32EnumAttrCase<"Component", 31> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Index                              : I32EnumAttrCase<"Index", 32> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Binding                            : I32EnumAttrCase<"Binding", 33> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_DescriptorSet                      : I32EnumAttrCase<"DescriptorSet", 34> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Offset                             : I32EnumAttrCase<"Offset", 35> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_XfbBuffer                          : I32EnumAttrCase<"XfbBuffer", 36> {
  list<Availability> availability = [
    Capability<[SPIRV_C_TransformFeedback]>
  ];
}
def SPIRV_D_XfbStride                          : I32EnumAttrCase<"XfbStride", 37> {
  list<Availability> availability = [
    Capability<[SPIRV_C_TransformFeedback]>
  ];
}
def SPIRV_D_FuncParamAttr                      : I32EnumAttrCase<"FuncParamAttr", 38> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_FPRoundingMode                     : I32EnumAttrCase<"FPRoundingMode", 39>;
def SPIRV_D_FPFastMathMode                     : I32EnumAttrCase<"FPFastMathMode", 40> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_LinkageAttributes                  : I32EnumAttrCase<"LinkageAttributes", 41> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Linkage]>
  ];
}
def SPIRV_D_NoContraction                      : I32EnumAttrCase<"NoContraction", 42> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_D_InputAttachmentIndex               : I32EnumAttrCase<"InputAttachmentIndex", 43> {
  list<Availability> availability = [
    Capability<[SPIRV_C_InputAttachment]>
  ];
}
def SPIRV_D_Alignment                          : I32EnumAttrCase<"Alignment", 44> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_MaxByteOffset                      : I32EnumAttrCase<"MaxByteOffset", 45> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>,
    Capability<[SPIRV_C_Addresses]>
  ];
}
def SPIRV_D_AlignmentId                        : I32EnumAttrCase<"AlignmentId", 46> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_2>,
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_D_MaxByteOffsetId                    : I32EnumAttrCase<"MaxByteOffsetId", 47> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_2>,
    Capability<[SPIRV_C_Addresses]>
  ];
}
def SPIRV_D_NoSignedWrap                       : I32EnumAttrCase<"NoSignedWrap", 4469> {
  list<Availability> availability = [
    Extension<[SPV_KHR_no_integer_wrap_decoration]>
  ];
}
def SPIRV_D_NoUnsignedWrap                     : I32EnumAttrCase<"NoUnsignedWrap", 4470> {
  list<Availability> availability = [
    Extension<[SPV_KHR_no_integer_wrap_decoration]>
  ];
}
def SPIRV_D_ExplicitInterpAMD                  : I32EnumAttrCase<"ExplicitInterpAMD", 4999> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_explicit_vertex_parameter]>
  ];
}
def SPIRV_D_OverrideCoverageNV                 : I32EnumAttrCase<"OverrideCoverageNV", 5248> {
  list<Availability> availability = [
    Extension<[SPV_NV_sample_mask_override_coverage]>,
    Capability<[SPIRV_C_SampleMaskOverrideCoverageNV]>
  ];
}
def SPIRV_D_PassthroughNV                      : I32EnumAttrCase<"PassthroughNV", 5250> {
  list<Availability> availability = [
    Extension<[SPV_NV_geometry_shader_passthrough]>,
    Capability<[SPIRV_C_GeometryShaderPassthroughNV]>
  ];
}
def SPIRV_D_ViewportRelativeNV                 : I32EnumAttrCase<"ViewportRelativeNV", 5252> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ShaderViewportMaskNV]>
  ];
}
def SPIRV_D_SecondaryViewportRelativeNV        : I32EnumAttrCase<"SecondaryViewportRelativeNV", 5256> {
  list<Availability> availability = [
    Extension<[SPV_NV_stereo_view_rendering]>,
    Capability<[SPIRV_C_ShaderStereoViewNV]>
  ];
}
def SPIRV_D_PerPrimitiveNV                     : I32EnumAttrCase<"PerPrimitiveNV", 5271> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_D_PerViewNV                          : I32EnumAttrCase<"PerViewNV", 5272> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_D_PerTaskNV                          : I32EnumAttrCase<"PerTaskNV", 5273> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_D_PerVertexKHR                       : I32EnumAttrCase<"PerVertexKHR", 5285> {
  list<Availability> availability = [
    Extension<[SPV_KHR_fragment_shader_barycentric, SPV_NV_fragment_shader_barycentric]>,
    Capability<[SPIRV_C_FragmentBarycentricKHR]>
  ];
}
def SPIRV_D_NonUniform                         : I32EnumAttrCase<"NonUniform", 5300> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_ShaderNonUniform]>
  ];
}
def SPIRV_D_RestrictPointer                    : I32EnumAttrCase<"RestrictPointer", 5355> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPIRV_C_PhysicalStorageBufferAddresses]>
  ];
}
def SPIRV_D_AliasedPointer                     : I32EnumAttrCase<"AliasedPointer", 5356> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPIRV_C_PhysicalStorageBufferAddresses]>
  ];
}
def SPIRV_D_BindlessSamplerNV                  : I32EnumAttrCase<"BindlessSamplerNV", 5398> {
  list<Availability> availability = [
    Capability<[SPIRV_C_BindlessTextureNV]>
  ];
}
def SPIRV_D_BindlessImageNV                    : I32EnumAttrCase<"BindlessImageNV", 5399> {
  list<Availability> availability = [
    Capability<[SPIRV_C_BindlessTextureNV]>
  ];
}
def SPIRV_D_BoundSamplerNV                     : I32EnumAttrCase<"BoundSamplerNV", 5400> {
  list<Availability> availability = [
    Capability<[SPIRV_C_BindlessTextureNV]>
  ];
}
def SPIRV_D_BoundImageNV                       : I32EnumAttrCase<"BoundImageNV", 5401> {
  list<Availability> availability = [
    Capability<[SPIRV_C_BindlessTextureNV]>
  ];
}
def SPIRV_D_SIMTCallINTEL                      : I32EnumAttrCase<"SIMTCallINTEL", 5599> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_ReferencedIndirectlyINTEL          : I32EnumAttrCase<"ReferencedIndirectlyINTEL", 5602> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_function_pointers]>,
    Capability<[SPIRV_C_IndirectReferencesINTEL]>
  ];
}
def SPIRV_D_ClobberINTEL                       : I32EnumAttrCase<"ClobberINTEL", 5607> {
  list<Availability> availability = [
    Capability<[SPIRV_C_AsmINTEL]>
  ];
}
def SPIRV_D_SideEffectsINTEL                   : I32EnumAttrCase<"SideEffectsINTEL", 5608> {
  list<Availability> availability = [
    Capability<[SPIRV_C_AsmINTEL]>
  ];
}
def SPIRV_D_VectorComputeVariableINTEL         : I32EnumAttrCase<"VectorComputeVariableINTEL", 5624> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_FuncParamIOKindINTEL               : I32EnumAttrCase<"FuncParamIOKindINTEL", 5625> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_VectorComputeFunctionINTEL         : I32EnumAttrCase<"VectorComputeFunctionINTEL", 5626> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_StackCallINTEL                     : I32EnumAttrCase<"StackCallINTEL", 5627> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_GlobalVariableOffsetINTEL          : I32EnumAttrCase<"GlobalVariableOffsetINTEL", 5628> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_CounterBuffer                      : I32EnumAttrCase<"CounterBuffer", 5634> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_D_UserSemantic                       : I32EnumAttrCase<"UserSemantic", 5635> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_D_UserTypeGOOGLE                     : I32EnumAttrCase<"UserTypeGOOGLE", 5636> {
  list<Availability> availability = [
    Extension<[SPV_GOOGLE_user_type]>
  ];
}
def SPIRV_D_FunctionRoundingModeINTEL          : I32EnumAttrCase<"FunctionRoundingModeINTEL", 5822> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FunctionFloatControlINTEL]>
  ];
}
def SPIRV_D_FunctionDenormModeINTEL            : I32EnumAttrCase<"FunctionDenormModeINTEL", 5823> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FunctionFloatControlINTEL]>
  ];
}
def SPIRV_D_RegisterINTEL                      : I32EnumAttrCase<"RegisterINTEL", 5825> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_MemoryINTEL                        : I32EnumAttrCase<"MemoryINTEL", 5826> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_NumbanksINTEL                      : I32EnumAttrCase<"NumbanksINTEL", 5827> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_BankwidthINTEL                     : I32EnumAttrCase<"BankwidthINTEL", 5828> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_MaxPrivateCopiesINTEL              : I32EnumAttrCase<"MaxPrivateCopiesINTEL", 5829> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_SinglepumpINTEL                    : I32EnumAttrCase<"SinglepumpINTEL", 5830> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_DoublepumpINTEL                    : I32EnumAttrCase<"DoublepumpINTEL", 5831> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_MaxReplicatesINTEL                 : I32EnumAttrCase<"MaxReplicatesINTEL", 5832> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_SimpleDualPortINTEL                : I32EnumAttrCase<"SimpleDualPortINTEL", 5833> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_MergeINTEL                         : I32EnumAttrCase<"MergeINTEL", 5834> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_BankBitsINTEL                      : I32EnumAttrCase<"BankBitsINTEL", 5835> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_ForcePow2DepthINTEL                : I32EnumAttrCase<"ForcePow2DepthINTEL", 5836> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_memory_attributes]>,
    Capability<[SPIRV_C_FPGAMemoryAttributesINTEL]>
  ];
}
def SPIRV_D_BurstCoalesceINTEL                 : I32EnumAttrCase<"BurstCoalesceINTEL", 5899> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAMemoryAccessesINTEL]>
  ];
}
def SPIRV_D_CacheSizeINTEL                     : I32EnumAttrCase<"CacheSizeINTEL", 5900> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAMemoryAccessesINTEL]>
  ];
}
def SPIRV_D_DontStaticallyCoalesceINTEL        : I32EnumAttrCase<"DontStaticallyCoalesceINTEL", 5901> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAMemoryAccessesINTEL]>
  ];
}
def SPIRV_D_PrefetchINTEL                      : I32EnumAttrCase<"PrefetchINTEL", 5902> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAMemoryAccessesINTEL]>
  ];
}
def SPIRV_D_StallEnableINTEL                   : I32EnumAttrCase<"StallEnableINTEL", 5905> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAClusterAttributesINTEL]>
  ];
}
def SPIRV_D_FuseLoopsInFunctionINTEL           : I32EnumAttrCase<"FuseLoopsInFunctionINTEL", 5907> {
  list<Availability> availability = [
    Capability<[SPIRV_C_LoopFuseINTEL]>
  ];
}
def SPIRV_D_AliasScopeINTEL                    : I32EnumAttrCase<"AliasScopeINTEL", 5914> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MemoryAccessAliasingINTEL]>
  ];
}
def SPIRV_D_NoAliasINTEL                       : I32EnumAttrCase<"NoAliasINTEL", 5915> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MemoryAccessAliasingINTEL]>
  ];
}
def SPIRV_D_BufferLocationINTEL                : I32EnumAttrCase<"BufferLocationINTEL", 5921> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGABufferLocationINTEL]>
  ];
}
def SPIRV_D_IOPipeStorageINTEL                 : I32EnumAttrCase<"IOPipeStorageINTEL", 5944> {
  list<Availability> availability = [
    Capability<[SPIRV_C_IOPipesINTEL]>
  ];
}
def SPIRV_D_FunctionFloatingPointModeINTEL     : I32EnumAttrCase<"FunctionFloatingPointModeINTEL", 6080> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FunctionFloatControlINTEL]>
  ];
}
def SPIRV_D_SingleElementVectorINTEL           : I32EnumAttrCase<"SingleElementVectorINTEL", 6085> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_VectorComputeCallableFunctionINTEL : I32EnumAttrCase<"VectorComputeCallableFunctionINTEL", 6087> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_D_MediaBlockIOINTEL                  : I32EnumAttrCase<"MediaBlockIOINTEL", 6140> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}

def SPIRV_DecorationAttr :
    SPIRV_I32EnumAttr<"Decoration", "valid SPIR-V Decoration", "decoration", [
      SPIRV_D_RelaxedPrecision, SPIRV_D_SpecId, SPIRV_D_Block, SPIRV_D_BufferBlock,
      SPIRV_D_RowMajor, SPIRV_D_ColMajor, SPIRV_D_ArrayStride, SPIRV_D_MatrixStride,
      SPIRV_D_GLSLShared, SPIRV_D_GLSLPacked, SPIRV_D_CPacked, SPIRV_D_BuiltIn,
      SPIRV_D_NoPerspective, SPIRV_D_Flat, SPIRV_D_Patch, SPIRV_D_Centroid, SPIRV_D_Sample,
      SPIRV_D_Invariant, SPIRV_D_Restrict, SPIRV_D_Aliased, SPIRV_D_Volatile, SPIRV_D_Constant,
      SPIRV_D_Coherent, SPIRV_D_NonWritable, SPIRV_D_NonReadable, SPIRV_D_Uniform,
      SPIRV_D_UniformId, SPIRV_D_SaturatedConversion, SPIRV_D_Stream, SPIRV_D_Location,
      SPIRV_D_Component, SPIRV_D_Index, SPIRV_D_Binding, SPIRV_D_DescriptorSet, SPIRV_D_Offset,
      SPIRV_D_XfbBuffer, SPIRV_D_XfbStride, SPIRV_D_FuncParamAttr, SPIRV_D_FPRoundingMode,
      SPIRV_D_FPFastMathMode, SPIRV_D_LinkageAttributes, SPIRV_D_NoContraction,
      SPIRV_D_InputAttachmentIndex, SPIRV_D_Alignment, SPIRV_D_MaxByteOffset,
      SPIRV_D_AlignmentId, SPIRV_D_MaxByteOffsetId, SPIRV_D_NoSignedWrap,
      SPIRV_D_NoUnsignedWrap, SPIRV_D_ExplicitInterpAMD, SPIRV_D_OverrideCoverageNV,
      SPIRV_D_PassthroughNV, SPIRV_D_ViewportRelativeNV,
      SPIRV_D_SecondaryViewportRelativeNV, SPIRV_D_PerPrimitiveNV, SPIRV_D_PerViewNV,
      SPIRV_D_PerTaskNV, SPIRV_D_PerVertexKHR, SPIRV_D_NonUniform, SPIRV_D_RestrictPointer,
      SPIRV_D_AliasedPointer, SPIRV_D_BindlessSamplerNV, SPIRV_D_BindlessImageNV,
      SPIRV_D_BoundSamplerNV, SPIRV_D_BoundImageNV, SPIRV_D_SIMTCallINTEL,
      SPIRV_D_ReferencedIndirectlyINTEL, SPIRV_D_ClobberINTEL, SPIRV_D_SideEffectsINTEL,
      SPIRV_D_VectorComputeVariableINTEL, SPIRV_D_FuncParamIOKindINTEL,
      SPIRV_D_VectorComputeFunctionINTEL, SPIRV_D_StackCallINTEL,
      SPIRV_D_GlobalVariableOffsetINTEL, SPIRV_D_CounterBuffer, SPIRV_D_UserSemantic,
      SPIRV_D_UserTypeGOOGLE, SPIRV_D_FunctionRoundingModeINTEL,
      SPIRV_D_FunctionDenormModeINTEL, SPIRV_D_RegisterINTEL, SPIRV_D_MemoryINTEL,
      SPIRV_D_NumbanksINTEL, SPIRV_D_BankwidthINTEL, SPIRV_D_MaxPrivateCopiesINTEL,
      SPIRV_D_SinglepumpINTEL, SPIRV_D_DoublepumpINTEL, SPIRV_D_MaxReplicatesINTEL,
      SPIRV_D_SimpleDualPortINTEL, SPIRV_D_MergeINTEL, SPIRV_D_BankBitsINTEL,
      SPIRV_D_ForcePow2DepthINTEL, SPIRV_D_BurstCoalesceINTEL, SPIRV_D_CacheSizeINTEL,
      SPIRV_D_DontStaticallyCoalesceINTEL, SPIRV_D_PrefetchINTEL, SPIRV_D_StallEnableINTEL,
      SPIRV_D_FuseLoopsInFunctionINTEL, SPIRV_D_AliasScopeINTEL, SPIRV_D_NoAliasINTEL,
      SPIRV_D_BufferLocationINTEL, SPIRV_D_IOPipeStorageINTEL,
      SPIRV_D_FunctionFloatingPointModeINTEL, SPIRV_D_SingleElementVectorINTEL,
      SPIRV_D_VectorComputeCallableFunctionINTEL, SPIRV_D_MediaBlockIOINTEL
    ]>;

def SPIRV_D_1D          : I32EnumAttrCase<"Dim1D", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Image1D, SPIRV_C_Sampled1D]>
  ];
}
def SPIRV_D_2D          : I32EnumAttrCase<"Dim2D", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageMSArray, SPIRV_C_Kernel, SPIRV_C_Shader]>
  ];
}
def SPIRV_D_3D          : I32EnumAttrCase<"Dim3D", 2>;
def SPIRV_D_Cube        : I32EnumAttrCase<"Cube", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageCubeArray, SPIRV_C_Shader]>
  ];
}
def SPIRV_D_Rect        : I32EnumAttrCase<"Rect", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageRect, SPIRV_C_SampledRect]>
  ];
}
def SPIRV_D_Buffer      : I32EnumAttrCase<"Buffer", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageBuffer, SPIRV_C_SampledBuffer]>
  ];
}
def SPIRV_D_SubpassData : I32EnumAttrCase<"SubpassData", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_InputAttachment]>
  ];
}

def SPIRV_DimAttr :
    SPIRV_I32EnumAttr<"Dim", "valid SPIR-V Dim", "dim", [
      SPIRV_D_1D, SPIRV_D_2D, SPIRV_D_3D, SPIRV_D_Cube, SPIRV_D_Rect, SPIRV_D_Buffer,
      SPIRV_D_SubpassData
    ]>;

def SPIRV_EM_Invocations                      : I32EnumAttrCase<"Invocations", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_SpacingEqual                     : I32EnumAttrCase<"SpacingEqual", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_SpacingFractionalEven            : I32EnumAttrCase<"SpacingFractionalEven", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_SpacingFractionalOdd             : I32EnumAttrCase<"SpacingFractionalOdd", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_VertexOrderCw                    : I32EnumAttrCase<"VertexOrderCw", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_VertexOrderCcw                   : I32EnumAttrCase<"VertexOrderCcw", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_PixelCenterInteger               : I32EnumAttrCase<"PixelCenterInteger", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_OriginUpperLeft                  : I32EnumAttrCase<"OriginUpperLeft", 7> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_OriginLowerLeft                  : I32EnumAttrCase<"OriginLowerLeft", 8> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_EarlyFragmentTests               : I32EnumAttrCase<"EarlyFragmentTests", 9> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_PointMode                        : I32EnumAttrCase<"PointMode", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_Xfb                              : I32EnumAttrCase<"Xfb", 11> {
  list<Availability> availability = [
    Capability<[SPIRV_C_TransformFeedback]>
  ];
}
def SPIRV_EM_DepthReplacing                   : I32EnumAttrCase<"DepthReplacing", 12> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_DepthGreater                     : I32EnumAttrCase<"DepthGreater", 14> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_DepthLess                        : I32EnumAttrCase<"DepthLess", 15> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_DepthUnchanged                   : I32EnumAttrCase<"DepthUnchanged", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_LocalSize                        : I32EnumAttrCase<"LocalSize", 17>;
def SPIRV_EM_LocalSizeHint                    : I32EnumAttrCase<"LocalSizeHint", 18> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_InputPoints                      : I32EnumAttrCase<"InputPoints", 19> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_InputLines                       : I32EnumAttrCase<"InputLines", 20> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_InputLinesAdjacency              : I32EnumAttrCase<"InputLinesAdjacency", 21> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_Triangles                        : I32EnumAttrCase<"Triangles", 22> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_InputTrianglesAdjacency          : I32EnumAttrCase<"InputTrianglesAdjacency", 23> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_Quads                            : I32EnumAttrCase<"Quads", 24> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_Isolines                         : I32EnumAttrCase<"Isolines", 25> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_OutputVertices                   : I32EnumAttrCase<"OutputVertices", 26> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_MeshShadingNV, SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_OutputPoints                     : I32EnumAttrCase<"OutputPoints", 27> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry, SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_OutputLineStrip                  : I32EnumAttrCase<"OutputLineStrip", 28> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_OutputTriangleStrip              : I32EnumAttrCase<"OutputTriangleStrip", 29> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_VecTypeHint                      : I32EnumAttrCase<"VecTypeHint", 30> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_ContractionOff                   : I32EnumAttrCase<"ContractionOff", 31> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_Initializer                      : I32EnumAttrCase<"Initializer", 33> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>,
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_Finalizer                        : I32EnumAttrCase<"Finalizer", 34> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>,
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_SubgroupSize                     : I32EnumAttrCase<"SubgroupSize", 35> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>,
    Capability<[SPIRV_C_SubgroupDispatch]>
  ];
}
def SPIRV_EM_SubgroupsPerWorkgroup            : I32EnumAttrCase<"SubgroupsPerWorkgroup", 36> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>,
    Capability<[SPIRV_C_SubgroupDispatch]>
  ];
}
def SPIRV_EM_SubgroupsPerWorkgroupId          : I32EnumAttrCase<"SubgroupsPerWorkgroupId", 37> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_2>,
    Capability<[SPIRV_C_SubgroupDispatch]>
  ];
}
def SPIRV_EM_LocalSizeId                      : I32EnumAttrCase<"LocalSizeId", 38> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_2>
  ];
}
def SPIRV_EM_LocalSizeHintId                  : I32EnumAttrCase<"LocalSizeHintId", 39> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_2>,
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_SubgroupUniformControlFlowKHR    : I32EnumAttrCase<"SubgroupUniformControlFlowKHR", 4421> {
  list<Availability> availability = [
    Extension<[SPV_KHR_subgroup_uniform_control_flow]>,
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_PostDepthCoverage                : I32EnumAttrCase<"PostDepthCoverage", 4446> {
  list<Availability> availability = [
    Extension<[SPV_KHR_post_depth_coverage]>,
    Capability<[SPIRV_C_SampleMaskPostDepthCoverage]>
  ];
}
def SPIRV_EM_DenormPreserve                   : I32EnumAttrCase<"DenormPreserve", 4459> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPIRV_C_DenormPreserve]>
  ];
}
def SPIRV_EM_DenormFlushToZero                : I32EnumAttrCase<"DenormFlushToZero", 4460> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPIRV_C_DenormFlushToZero]>
  ];
}
def SPIRV_EM_SignedZeroInfNanPreserve         : I32EnumAttrCase<"SignedZeroInfNanPreserve", 4461> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPIRV_C_SignedZeroInfNanPreserve]>
  ];
}
def SPIRV_EM_RoundingModeRTE                  : I32EnumAttrCase<"RoundingModeRTE", 4462> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPIRV_C_RoundingModeRTE]>
  ];
}
def SPIRV_EM_RoundingModeRTZ                  : I32EnumAttrCase<"RoundingModeRTZ", 4463> {
  list<Availability> availability = [
    Extension<[SPV_KHR_float_controls]>,
    Capability<[SPIRV_C_RoundingModeRTZ]>
  ];
}
def SPIRV_EM_EarlyAndLateFragmentTestsAMD     : I32EnumAttrCase<"EarlyAndLateFragmentTestsAMD", 5017> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests]>,
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_StencilRefReplacingEXT           : I32EnumAttrCase<"StencilRefReplacingEXT", 5027> {
  list<Availability> availability = [
    Extension<[SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefUnchangedFrontAMD      : I32EnumAttrCase<"StencilRefUnchangedFrontAMD", 5079> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefGreaterFrontAMD        : I32EnumAttrCase<"StencilRefGreaterFrontAMD", 5080> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefLessFrontAMD           : I32EnumAttrCase<"StencilRefLessFrontAMD", 5081> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefUnchangedBackAMD       : I32EnumAttrCase<"StencilRefUnchangedBackAMD", 5082> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefGreaterBackAMD         : I32EnumAttrCase<"StencilRefGreaterBackAMD", 5083> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_StencilRefLessBackAMD            : I32EnumAttrCase<"StencilRefLessBackAMD", 5084> {
  list<Availability> availability = [
    Extension<[SPV_AMD_shader_early_and_late_fragment_tests, SPV_EXT_shader_stencil_export]>,
    Capability<[SPIRV_C_StencilExportEXT]>
  ];
}
def SPIRV_EM_OutputLinesNV                    : I32EnumAttrCase<"OutputLinesNV", 5269> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_OutputPrimitivesNV               : I32EnumAttrCase<"OutputPrimitivesNV", 5270> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_DerivativeGroupQuadsNV           : I32EnumAttrCase<"DerivativeGroupQuadsNV", 5289> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>,
    Capability<[SPIRV_C_ComputeDerivativeGroupQuadsNV]>
  ];
}
def SPIRV_EM_DerivativeGroupLinearNV          : I32EnumAttrCase<"DerivativeGroupLinearNV", 5290> {
  list<Availability> availability = [
    Extension<[SPV_NV_compute_shader_derivatives]>,
    Capability<[SPIRV_C_ComputeDerivativeGroupLinearNV]>
  ];
}
def SPIRV_EM_OutputTrianglesNV                : I32EnumAttrCase<"OutputTrianglesNV", 5298> {
  list<Availability> availability = [
    Extension<[SPV_NV_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_PixelInterlockOrderedEXT         : I32EnumAttrCase<"PixelInterlockOrderedEXT", 5366> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderPixelInterlockEXT]>
  ];
}
def SPIRV_EM_PixelInterlockUnorderedEXT       : I32EnumAttrCase<"PixelInterlockUnorderedEXT", 5367> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderPixelInterlockEXT]>
  ];
}
def SPIRV_EM_SampleInterlockOrderedEXT        : I32EnumAttrCase<"SampleInterlockOrderedEXT", 5368> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderSampleInterlockEXT]>
  ];
}
def SPIRV_EM_SampleInterlockUnorderedEXT      : I32EnumAttrCase<"SampleInterlockUnorderedEXT", 5369> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderSampleInterlockEXT]>
  ];
}
def SPIRV_EM_ShadingRateInterlockOrderedEXT   : I32EnumAttrCase<"ShadingRateInterlockOrderedEXT", 5370> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderShadingRateInterlockEXT]>
  ];
}
def SPIRV_EM_ShadingRateInterlockUnorderedEXT : I32EnumAttrCase<"ShadingRateInterlockUnorderedEXT", 5371> {
  list<Availability> availability = [
    Extension<[SPV_EXT_fragment_shader_interlock]>,
    Capability<[SPIRV_C_FragmentShaderShadingRateInterlockEXT]>
  ];
}
def SPIRV_EM_SharedLocalMemorySizeINTEL       : I32EnumAttrCase<"SharedLocalMemorySizeINTEL", 5618> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_EM_RoundingModeRTPINTEL             : I32EnumAttrCase<"RoundingModeRTPINTEL", 5620> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RoundToInfinityINTEL]>
  ];
}
def SPIRV_EM_RoundingModeRTNINTEL             : I32EnumAttrCase<"RoundingModeRTNINTEL", 5621> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RoundToInfinityINTEL]>
  ];
}
def SPIRV_EM_FloatingPointModeALTINTEL        : I32EnumAttrCase<"FloatingPointModeALTINTEL", 5622> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RoundToInfinityINTEL]>
  ];
}
def SPIRV_EM_FloatingPointModeIEEEINTEL       : I32EnumAttrCase<"FloatingPointModeIEEEINTEL", 5623> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RoundToInfinityINTEL]>
  ];
}
def SPIRV_EM_MaxWorkgroupSizeINTEL            : I32EnumAttrCase<"MaxWorkgroupSizeINTEL", 5893> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>,
    Capability<[SPIRV_C_KernelAttributesINTEL]>
  ];
}
def SPIRV_EM_MaxWorkDimINTEL                  : I32EnumAttrCase<"MaxWorkDimINTEL", 5894> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>,
    Capability<[SPIRV_C_KernelAttributesINTEL]>
  ];
}
def SPIRV_EM_NoGlobalOffsetINTEL              : I32EnumAttrCase<"NoGlobalOffsetINTEL", 5895> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>,
    Capability<[SPIRV_C_KernelAttributesINTEL]>
  ];
}
def SPIRV_EM_NumSIMDWorkitemsINTEL            : I32EnumAttrCase<"NumSIMDWorkitemsINTEL", 5896> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_kernel_attributes]>,
    Capability<[SPIRV_C_FPGAKernelAttributesINTEL]>
  ];
}
def SPIRV_EM_SchedulerTargetFmaxMhzINTEL      : I32EnumAttrCase<"SchedulerTargetFmaxMhzINTEL", 5903> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAKernelAttributesINTEL]>
  ];
}
def SPIRV_EM_StreamingInterfaceINTEL          : I32EnumAttrCase<"StreamingInterfaceINTEL", 6154> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPGAKernelAttributesINTEL]>
  ];
}
def SPIRV_EM_NamedBarrierCountINTEL           : I32EnumAttrCase<"NamedBarrierCountINTEL", 6417> {
  list<Availability> availability = [
    Capability<[SPIRV_C_VectorComputeINTEL]>
  ];
}

def SPIRV_ExecutionModeAttr :
    SPIRV_I32EnumAttr<"ExecutionMode", "valid SPIR-V ExecutionMode", "execution_mode", [
      SPIRV_EM_Invocations, SPIRV_EM_SpacingEqual, SPIRV_EM_SpacingFractionalEven,
      SPIRV_EM_SpacingFractionalOdd, SPIRV_EM_VertexOrderCw, SPIRV_EM_VertexOrderCcw,
      SPIRV_EM_PixelCenterInteger, SPIRV_EM_OriginUpperLeft, SPIRV_EM_OriginLowerLeft,
      SPIRV_EM_EarlyFragmentTests, SPIRV_EM_PointMode, SPIRV_EM_Xfb, SPIRV_EM_DepthReplacing,
      SPIRV_EM_DepthGreater, SPIRV_EM_DepthLess, SPIRV_EM_DepthUnchanged, SPIRV_EM_LocalSize,
      SPIRV_EM_LocalSizeHint, SPIRV_EM_InputPoints, SPIRV_EM_InputLines,
      SPIRV_EM_InputLinesAdjacency, SPIRV_EM_Triangles, SPIRV_EM_InputTrianglesAdjacency,
      SPIRV_EM_Quads, SPIRV_EM_Isolines, SPIRV_EM_OutputVertices, SPIRV_EM_OutputPoints,
      SPIRV_EM_OutputLineStrip, SPIRV_EM_OutputTriangleStrip, SPIRV_EM_VecTypeHint,
      SPIRV_EM_ContractionOff, SPIRV_EM_Initializer, SPIRV_EM_Finalizer,
      SPIRV_EM_SubgroupSize, SPIRV_EM_SubgroupsPerWorkgroup,
      SPIRV_EM_SubgroupsPerWorkgroupId, SPIRV_EM_LocalSizeId, SPIRV_EM_LocalSizeHintId,
      SPIRV_EM_SubgroupUniformControlFlowKHR, SPIRV_EM_PostDepthCoverage,
      SPIRV_EM_DenormPreserve, SPIRV_EM_DenormFlushToZero,
      SPIRV_EM_SignedZeroInfNanPreserve, SPIRV_EM_RoundingModeRTE,
      SPIRV_EM_RoundingModeRTZ, SPIRV_EM_EarlyAndLateFragmentTestsAMD,
      SPIRV_EM_StencilRefReplacingEXT, SPIRV_EM_StencilRefUnchangedFrontAMD,
      SPIRV_EM_StencilRefGreaterFrontAMD, SPIRV_EM_StencilRefLessFrontAMD,
      SPIRV_EM_StencilRefUnchangedBackAMD, SPIRV_EM_StencilRefGreaterBackAMD,
      SPIRV_EM_StencilRefLessBackAMD, SPIRV_EM_OutputLinesNV, SPIRV_EM_OutputPrimitivesNV,
      SPIRV_EM_DerivativeGroupQuadsNV, SPIRV_EM_DerivativeGroupLinearNV,
      SPIRV_EM_OutputTrianglesNV, SPIRV_EM_PixelInterlockOrderedEXT,
      SPIRV_EM_PixelInterlockUnorderedEXT, SPIRV_EM_SampleInterlockOrderedEXT,
      SPIRV_EM_SampleInterlockUnorderedEXT, SPIRV_EM_ShadingRateInterlockOrderedEXT,
      SPIRV_EM_ShadingRateInterlockUnorderedEXT, SPIRV_EM_SharedLocalMemorySizeINTEL,
      SPIRV_EM_RoundingModeRTPINTEL, SPIRV_EM_RoundingModeRTNINTEL,
      SPIRV_EM_FloatingPointModeALTINTEL, SPIRV_EM_FloatingPointModeIEEEINTEL,
      SPIRV_EM_MaxWorkgroupSizeINTEL, SPIRV_EM_MaxWorkDimINTEL,
      SPIRV_EM_NoGlobalOffsetINTEL, SPIRV_EM_NumSIMDWorkitemsINTEL,
      SPIRV_EM_SchedulerTargetFmaxMhzINTEL, SPIRV_EM_StreamingInterfaceINTEL,
      SPIRV_EM_NamedBarrierCountINTEL
    ]>;

def SPIRV_EM_Vertex                 : I32EnumAttrCase<"Vertex", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_TessellationControl    : I32EnumAttrCase<"TessellationControl", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_TessellationEvaluation : I32EnumAttrCase<"TessellationEvaluation", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Tessellation]>
  ];
}
def SPIRV_EM_Geometry               : I32EnumAttrCase<"Geometry", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Geometry]>
  ];
}
def SPIRV_EM_Fragment               : I32EnumAttrCase<"Fragment", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_GLCompute              : I32EnumAttrCase<"GLCompute", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_EM_Kernel                 : I32EnumAttrCase<"Kernel", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_EM_TaskNV                 : I32EnumAttrCase<"TaskNV", 5267> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_MeshNV                 : I32EnumAttrCase<"MeshNV", 5268> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MeshShadingNV]>
  ];
}
def SPIRV_EM_RayGenerationKHR       : I32EnumAttrCase<"RayGenerationKHR", 5313> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_EM_IntersectionKHR        : I32EnumAttrCase<"IntersectionKHR", 5314> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_EM_AnyHitKHR              : I32EnumAttrCase<"AnyHitKHR", 5315> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_EM_ClosestHitKHR          : I32EnumAttrCase<"ClosestHitKHR", 5316> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_EM_MissKHR                : I32EnumAttrCase<"MissKHR", 5317> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_EM_CallableKHR            : I32EnumAttrCase<"CallableKHR", 5318> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}

def SPIRV_ExecutionModelAttr :
    SPIRV_I32EnumAttr<"ExecutionModel", "valid SPIR-V ExecutionModel", "execution_model", [
      SPIRV_EM_Vertex, SPIRV_EM_TessellationControl, SPIRV_EM_TessellationEvaluation,
      SPIRV_EM_Geometry, SPIRV_EM_Fragment, SPIRV_EM_GLCompute, SPIRV_EM_Kernel,
      SPIRV_EM_TaskNV, SPIRV_EM_MeshNV, SPIRV_EM_RayGenerationKHR, SPIRV_EM_IntersectionKHR,
      SPIRV_EM_AnyHitKHR, SPIRV_EM_ClosestHitKHR, SPIRV_EM_MissKHR, SPIRV_EM_CallableKHR
    ]>;

def SPIRV_FC_None         : I32BitEnumAttrCaseNone<"None">;
def SPIRV_FC_Inline       : I32BitEnumAttrCaseBit<"Inline", 0>;
def SPIRV_FC_DontInline   : I32BitEnumAttrCaseBit<"DontInline", 1>;
def SPIRV_FC_Pure         : I32BitEnumAttrCaseBit<"Pure", 2>;
def SPIRV_FC_Const        : I32BitEnumAttrCaseBit<"Const", 3>;
def SPIRV_FC_OptNoneINTEL : I32BitEnumAttrCaseBit<"OptNoneINTEL", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_OptNoneINTEL]>
  ];
}

def SPIRV_FunctionControlAttr :
    SPIRV_BitEnumAttr<"FunctionControl", "valid SPIR-V FunctionControl", "function_control", [
      SPIRV_FC_None, SPIRV_FC_Inline, SPIRV_FC_DontInline, SPIRV_FC_Pure, SPIRV_FC_Const,
      SPIRV_FC_OptNoneINTEL
    ]>;

def SPIRV_GO_Reduce                     : I32EnumAttrCase<"Reduce", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformBallot, SPIRV_C_Kernel]>
  ];
}
def SPIRV_GO_InclusiveScan              : I32EnumAttrCase<"InclusiveScan", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformBallot, SPIRV_C_Kernel]>
  ];
}
def SPIRV_GO_ExclusiveScan              : I32EnumAttrCase<"ExclusiveScan", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformBallot, SPIRV_C_Kernel]>
  ];
}
def SPIRV_GO_ClusteredReduce            : I32EnumAttrCase<"ClusteredReduce", 3> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_3>,
    Capability<[SPIRV_C_GroupNonUniformClustered]>
  ];
}
def SPIRV_GO_PartitionedReduceNV        : I32EnumAttrCase<"PartitionedReduceNV", 6> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}
def SPIRV_GO_PartitionedInclusiveScanNV : I32EnumAttrCase<"PartitionedInclusiveScanNV", 7> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}
def SPIRV_GO_PartitionedExclusiveScanNV : I32EnumAttrCase<"PartitionedExclusiveScanNV", 8> {
  list<Availability> availability = [
    Extension<[SPV_NV_shader_subgroup_partitioned]>,
    Capability<[SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

def SPIRV_GroupOperationAttr :
    SPIRV_I32EnumAttr<"GroupOperation", "valid SPIR-V GroupOperation", "group_operation", [
      SPIRV_GO_Reduce, SPIRV_GO_InclusiveScan, SPIRV_GO_ExclusiveScan,
      SPIRV_GO_ClusteredReduce, SPIRV_GO_PartitionedReduceNV,
      SPIRV_GO_PartitionedInclusiveScanNV, SPIRV_GO_PartitionedExclusiveScanNV
    ]>;

def SPIRV_IF_Unknown      : I32EnumAttrCase<"Unknown", 0>;
def SPIRV_IF_Rgba32f      : I32EnumAttrCase<"Rgba32f", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba16f      : I32EnumAttrCase<"Rgba16f", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_R32f         : I32EnumAttrCase<"R32f", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba8        : I32EnumAttrCase<"Rgba8", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba8Snorm   : I32EnumAttrCase<"Rgba8Snorm", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rg32f        : I32EnumAttrCase<"Rg32f", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg16f        : I32EnumAttrCase<"Rg16f", 7> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R11fG11fB10f : I32EnumAttrCase<"R11fG11fB10f", 8> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R16f         : I32EnumAttrCase<"R16f", 9> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rgba16       : I32EnumAttrCase<"Rgba16", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rgb10A2      : I32EnumAttrCase<"Rgb10A2", 11> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg16         : I32EnumAttrCase<"Rg16", 12> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg8          : I32EnumAttrCase<"Rg8", 13> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R16          : I32EnumAttrCase<"R16", 14> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R8           : I32EnumAttrCase<"R8", 15> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rgba16Snorm  : I32EnumAttrCase<"Rgba16Snorm", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg16Snorm    : I32EnumAttrCase<"Rg16Snorm", 17> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg8Snorm     : I32EnumAttrCase<"Rg8Snorm", 18> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R16Snorm     : I32EnumAttrCase<"R16Snorm", 19> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R8Snorm      : I32EnumAttrCase<"R8Snorm", 20> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rgba32i      : I32EnumAttrCase<"Rgba32i", 21> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba16i      : I32EnumAttrCase<"Rgba16i", 22> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba8i       : I32EnumAttrCase<"Rgba8i", 23> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_R32i         : I32EnumAttrCase<"R32i", 24> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rg32i        : I32EnumAttrCase<"Rg32i", 25> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg16i        : I32EnumAttrCase<"Rg16i", 26> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg8i         : I32EnumAttrCase<"Rg8i", 27> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R16i         : I32EnumAttrCase<"R16i", 28> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R8i          : I32EnumAttrCase<"R8i", 29> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rgba32ui     : I32EnumAttrCase<"Rgba32ui", 30> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba16ui     : I32EnumAttrCase<"Rgba16ui", 31> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgba8ui      : I32EnumAttrCase<"Rgba8ui", 32> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_R32ui        : I32EnumAttrCase<"R32ui", 33> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IF_Rgb10a2ui    : I32EnumAttrCase<"Rgb10a2ui", 34> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg32ui       : I32EnumAttrCase<"Rg32ui", 35> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg16ui       : I32EnumAttrCase<"Rg16ui", 36> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_Rg8ui        : I32EnumAttrCase<"Rg8ui", 37> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R16ui        : I32EnumAttrCase<"R16ui", 38> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R8ui         : I32EnumAttrCase<"R8ui", 39> {
  list<Availability> availability = [
    Capability<[SPIRV_C_StorageImageExtendedFormats]>
  ];
}
def SPIRV_IF_R64ui        : I32EnumAttrCase<"R64ui", 40> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Int64ImageEXT]>
  ];
}
def SPIRV_IF_R64i         : I32EnumAttrCase<"R64i", 41> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Int64ImageEXT]>
  ];
}

def SPIRV_ImageFormatAttr :
    SPIRV_I32EnumAttr<"ImageFormat", "valid SPIR-V ImageFormat", "image_format", [
      SPIRV_IF_Unknown, SPIRV_IF_Rgba32f, SPIRV_IF_Rgba16f, SPIRV_IF_R32f, SPIRV_IF_Rgba8,
      SPIRV_IF_Rgba8Snorm, SPIRV_IF_Rg32f, SPIRV_IF_Rg16f, SPIRV_IF_R11fG11fB10f,
      SPIRV_IF_R16f, SPIRV_IF_Rgba16, SPIRV_IF_Rgb10A2, SPIRV_IF_Rg16, SPIRV_IF_Rg8,
      SPIRV_IF_R16, SPIRV_IF_R8, SPIRV_IF_Rgba16Snorm, SPIRV_IF_Rg16Snorm, SPIRV_IF_Rg8Snorm,
      SPIRV_IF_R16Snorm, SPIRV_IF_R8Snorm, SPIRV_IF_Rgba32i, SPIRV_IF_Rgba16i, SPIRV_IF_Rgba8i,
      SPIRV_IF_R32i, SPIRV_IF_Rg32i, SPIRV_IF_Rg16i, SPIRV_IF_Rg8i, SPIRV_IF_R16i, SPIRV_IF_R8i,
      SPIRV_IF_Rgba32ui, SPIRV_IF_Rgba16ui, SPIRV_IF_Rgba8ui, SPIRV_IF_R32ui,
      SPIRV_IF_Rgb10a2ui, SPIRV_IF_Rg32ui, SPIRV_IF_Rg16ui, SPIRV_IF_Rg8ui, SPIRV_IF_R16ui,
      SPIRV_IF_R8ui, SPIRV_IF_R64ui, SPIRV_IF_R64i
    ]>;

def SPIRV_IO_None               : I32BitEnumAttrCaseNone<"None">;
def SPIRV_IO_Bias               : I32BitEnumAttrCaseBit<"Bias", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_IO_Lod                : I32BitEnumAttrCaseBit<"Lod", 1>;
def SPIRV_IO_Grad               : I32BitEnumAttrCaseBit<"Grad", 2>;
def SPIRV_IO_ConstOffset        : I32BitEnumAttrCaseBit<"ConstOffset", 3>;
def SPIRV_IO_Offset             : I32BitEnumAttrCaseBit<"Offset", 4> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageGatherExtended]>
  ];
}
def SPIRV_IO_ConstOffsets       : I32BitEnumAttrCaseBit<"ConstOffsets", 5> {
  list<Availability> availability = [
    Capability<[SPIRV_C_ImageGatherExtended]>
  ];
}
def SPIRV_IO_Sample             : I32BitEnumAttrCaseBit<"Sample", 6>;
def SPIRV_IO_MinLod             : I32BitEnumAttrCaseBit<"MinLod", 7> {
  list<Availability> availability = [
    Capability<[SPIRV_C_MinLod]>
  ];
}
def SPIRV_IO_MakeTexelAvailable : I32BitEnumAttrCaseBit<"MakeTexelAvailable", 8> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_IO_MakeTexelVisible   : I32BitEnumAttrCaseBit<"MakeTexelVisible", 9> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_IO_NonPrivateTexel    : I32BitEnumAttrCaseBit<"NonPrivateTexel", 10> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_IO_VolatileTexel      : I32BitEnumAttrCaseBit<"VolatileTexel", 11> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_IO_SignExtend         : I32BitEnumAttrCaseBit<"SignExtend", 12> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_IO_Offsets            : I32BitEnumAttrCaseBit<"Offsets", 16>;
def SPIRV_IO_ZeroExtend         : I32BitEnumAttrCaseBit<"ZeroExtend", 13> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_IO_Nontemporal        : I32BitEnumAttrCaseBit<"Nontemporal", 14> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_6>
  ];
}

def SPIRV_ImageOperandsAttr :
    SPIRV_BitEnumAttr<"ImageOperands", "valid SPIR-V ImageOperands", "image_operands", [
      SPIRV_IO_None, SPIRV_IO_Bias, SPIRV_IO_Lod, SPIRV_IO_Grad, SPIRV_IO_ConstOffset,
      SPIRV_IO_Offset, SPIRV_IO_ConstOffsets, SPIRV_IO_Sample, SPIRV_IO_MinLod,
      SPIRV_IO_MakeTexelAvailable, SPIRV_IO_MakeTexelVisible, SPIRV_IO_NonPrivateTexel,
      SPIRV_IO_VolatileTexel, SPIRV_IO_SignExtend, SPIRV_IO_Offsets, SPIRV_IO_ZeroExtend,
      SPIRV_IO_Nontemporal
    ]>;

def SPIRV_LT_Export      : I32EnumAttrCase<"Export", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Linkage]>
  ];
}
def SPIRV_LT_Import      : I32EnumAttrCase<"Import", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Linkage]>
  ];
}
def SPIRV_LT_LinkOnceODR : I32EnumAttrCase<"LinkOnceODR", 2> {
  list<Availability> availability = [
    Extension<[SPV_KHR_linkonce_odr]>,
    Capability<[SPIRV_C_Linkage]>
  ];
}

def SPIRV_LinkageTypeAttr :
    SPIRV_I32EnumAttr<"LinkageType", "valid SPIR-V LinkageType", "linkage_type", [
      SPIRV_LT_Export, SPIRV_LT_Import, SPIRV_LT_LinkOnceODR
    ]>;

def SPIRV_LC_None                      : I32BitEnumAttrCaseNone<"None">;
def SPIRV_LC_Unroll                    : I32BitEnumAttrCaseBit<"Unroll", 0>;
def SPIRV_LC_DontUnroll                : I32BitEnumAttrCaseBit<"DontUnroll", 1>;
def SPIRV_LC_DependencyInfinite        : I32BitEnumAttrCaseBit<"DependencyInfinite", 2> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>
  ];
}
def SPIRV_LC_DependencyLength          : I32BitEnumAttrCaseBit<"DependencyLength", 3> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_1>
  ];
}
def SPIRV_LC_MinIterations             : I32BitEnumAttrCaseBit<"MinIterations", 4> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_LC_MaxIterations             : I32BitEnumAttrCaseBit<"MaxIterations", 5> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_LC_IterationMultiple         : I32BitEnumAttrCaseBit<"IterationMultiple", 6> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_LC_PeelCount                 : I32BitEnumAttrCaseBit<"PeelCount", 7> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_LC_PartialCount              : I32BitEnumAttrCaseBit<"PartialCount", 8> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_4>
  ];
}
def SPIRV_LC_InitiationIntervalINTEL   : I32BitEnumAttrCaseBit<"InitiationIntervalINTEL", 16> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_LoopCoalesceINTEL         : I32BitEnumAttrCaseBit<"LoopCoalesceINTEL", 20> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_MaxConcurrencyINTEL       : I32BitEnumAttrCaseBit<"MaxConcurrencyINTEL", 17> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_MaxInterleavingINTEL      : I32BitEnumAttrCaseBit<"MaxInterleavingINTEL", 21> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_DependencyArrayINTEL      : I32BitEnumAttrCaseBit<"DependencyArrayINTEL", 18> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_SpeculatedIterationsINTEL : I32BitEnumAttrCaseBit<"SpeculatedIterationsINTEL", 22> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_PipelineEnableINTEL       : I32BitEnumAttrCaseBit<"PipelineEnableINTEL", 19> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}
def SPIRV_LC_NoFusionINTEL             : I32BitEnumAttrCaseBit<"NoFusionINTEL", 23> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_fpga_loop_controls]>,
    Capability<[SPIRV_C_FPGALoopControlsINTEL]>
  ];
}

def SPIRV_LoopControlAttr :
    SPIRV_BitEnumAttr<"LoopControl", "valid SPIR-V LoopControl", "loop_control", [
      SPIRV_LC_None, SPIRV_LC_Unroll, SPIRV_LC_DontUnroll, SPIRV_LC_DependencyInfinite,
      SPIRV_LC_DependencyLength, SPIRV_LC_MinIterations, SPIRV_LC_MaxIterations,
      SPIRV_LC_IterationMultiple, SPIRV_LC_PeelCount, SPIRV_LC_PartialCount,
      SPIRV_LC_InitiationIntervalINTEL, SPIRV_LC_LoopCoalesceINTEL,
      SPIRV_LC_MaxConcurrencyINTEL, SPIRV_LC_MaxInterleavingINTEL,
      SPIRV_LC_DependencyArrayINTEL, SPIRV_LC_SpeculatedIterationsINTEL,
      SPIRV_LC_PipelineEnableINTEL, SPIRV_LC_NoFusionINTEL
    ]>;

def SPIRV_MA_None                 : I32BitEnumAttrCaseNone<"None">;
def SPIRV_MA_Volatile             : I32BitEnumAttrCaseBit<"Volatile", 0>;
def SPIRV_MA_Aligned              : I32BitEnumAttrCaseBit<"Aligned", 1>;
def SPIRV_MA_Nontemporal          : I32BitEnumAttrCaseBit<"Nontemporal", 2>;
def SPIRV_MA_MakePointerAvailable : I32BitEnumAttrCaseBit<"MakePointerAvailable", 3> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MA_MakePointerVisible   : I32BitEnumAttrCaseBit<"MakePointerVisible", 4> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MA_NonPrivatePointer    : I32BitEnumAttrCaseBit<"NonPrivatePointer", 5> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MA_AliasScopeINTELMask  : I32BitEnumAttrCaseBit<"AliasScopeINTELMask", 16> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_memory_access_aliasing]>,
    Capability<[SPIRV_C_MemoryAccessAliasingINTEL]>
  ];
}
def SPIRV_MA_NoAliasINTELMask     : I32BitEnumAttrCaseBit<"NoAliasINTELMask", 17> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_memory_access_aliasing]>,
    Capability<[SPIRV_C_MemoryAccessAliasingINTEL]>
  ];
}

def SPIRV_MemoryAccessAttr :
    SPIRV_BitEnumAttr<"MemoryAccess", "valid SPIR-V MemoryAccess", "memory_access", [
      SPIRV_MA_None, SPIRV_MA_Volatile, SPIRV_MA_Aligned, SPIRV_MA_Nontemporal,
      SPIRV_MA_MakePointerAvailable, SPIRV_MA_MakePointerVisible,
      SPIRV_MA_NonPrivatePointer, SPIRV_MA_AliasScopeINTELMask, SPIRV_MA_NoAliasINTELMask
    ]>;

def SPIRV_MM_Simple  : I32EnumAttrCase<"Simple", 0> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_MM_GLSL450 : I32EnumAttrCase<"GLSL450", 1> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_MM_OpenCL  : I32EnumAttrCase<"OpenCL", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Kernel]>
  ];
}
def SPIRV_MM_Vulkan  : I32EnumAttrCase<"Vulkan", 3> {
  list<Availability> availability = [
    Extension<[SPV_KHR_vulkan_memory_model]>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}

def SPIRV_MemoryModelAttr :
    SPIRV_I32EnumAttr<"MemoryModel", "valid SPIR-V MemoryModel", "memory_model", [
      SPIRV_MM_Simple, SPIRV_MM_GLSL450, SPIRV_MM_OpenCL, SPIRV_MM_Vulkan
    ]>;

def SPIRV_MS_None                   : I32BitEnumAttrCaseNone<"None">;
def SPIRV_MS_Acquire                : I32BitEnumAttrCaseBit<"Acquire", 1>;
def SPIRV_MS_Release                : I32BitEnumAttrCaseBit<"Release", 2>;
def SPIRV_MS_AcquireRelease         : I32BitEnumAttrCaseBit<"AcquireRelease", 3>;
def SPIRV_MS_SequentiallyConsistent : I32BitEnumAttrCaseBit<"SequentiallyConsistent", 4>;
def SPIRV_MS_UniformMemory          : I32BitEnumAttrCaseBit<"UniformMemory", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_MS_SubgroupMemory         : I32BitEnumAttrCaseBit<"SubgroupMemory", 7>;
def SPIRV_MS_WorkgroupMemory        : I32BitEnumAttrCaseBit<"WorkgroupMemory", 8>;
def SPIRV_MS_CrossWorkgroupMemory   : I32BitEnumAttrCaseBit<"CrossWorkgroupMemory", 9>;
def SPIRV_MS_AtomicCounterMemory    : I32BitEnumAttrCaseBit<"AtomicCounterMemory", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_AtomicStorage]>
  ];
}
def SPIRV_MS_ImageMemory            : I32BitEnumAttrCaseBit<"ImageMemory", 11>;
def SPIRV_MS_OutputMemory           : I32BitEnumAttrCaseBit<"OutputMemory", 12> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MS_MakeAvailable          : I32BitEnumAttrCaseBit<"MakeAvailable", 13> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MS_MakeVisible            : I32BitEnumAttrCaseBit<"MakeVisible", 14> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_MS_Volatile               : I32BitEnumAttrCaseBit<"Volatile", 15> {
  list<Availability> availability = [
    Extension<[SPV_KHR_vulkan_memory_model]>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}

def SPIRV_MemorySemanticsAttr :
    SPIRV_BitEnumAttr<"MemorySemantics", "valid SPIR-V MemorySemantics", "memory_semantics", [
      SPIRV_MS_None, SPIRV_MS_Acquire, SPIRV_MS_Release, SPIRV_MS_AcquireRelease,
      SPIRV_MS_SequentiallyConsistent, SPIRV_MS_UniformMemory, SPIRV_MS_SubgroupMemory,
      SPIRV_MS_WorkgroupMemory, SPIRV_MS_CrossWorkgroupMemory,
      SPIRV_MS_AtomicCounterMemory, SPIRV_MS_ImageMemory, SPIRV_MS_OutputMemory,
      SPIRV_MS_MakeAvailable, SPIRV_MS_MakeVisible, SPIRV_MS_Volatile
    ]>;

def SPIRV_S_CrossDevice   : I32EnumAttrCase<"CrossDevice", 0>;
def SPIRV_S_Device        : I32EnumAttrCase<"Device", 1>;
def SPIRV_S_Workgroup     : I32EnumAttrCase<"Workgroup", 2>;
def SPIRV_S_Subgroup      : I32EnumAttrCase<"Subgroup", 3>;
def SPIRV_S_Invocation    : I32EnumAttrCase<"Invocation", 4>;
def SPIRV_S_QueueFamily   : I32EnumAttrCase<"QueueFamily", 5> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_5>,
    Capability<[SPIRV_C_VulkanMemoryModel]>
  ];
}
def SPIRV_S_ShaderCallKHR : I32EnumAttrCase<"ShaderCallKHR", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_RayTracingKHR]>
  ];
}

def SPIRV_ScopeAttr :
    SPIRV_I32EnumAttr<"Scope", "valid SPIR-V Scope", "scope", [
      SPIRV_S_CrossDevice, SPIRV_S_Device, SPIRV_S_Workgroup, SPIRV_S_Subgroup,
      SPIRV_S_Invocation, SPIRV_S_QueueFamily, SPIRV_S_ShaderCallKHR
    ]>;

def SPIRV_SC_None        : I32BitEnumAttrCaseNone<"None">;
def SPIRV_SC_Flatten     : I32BitEnumAttrCaseBit<"Flatten", 0>;
def SPIRV_SC_DontFlatten : I32BitEnumAttrCaseBit<"DontFlatten", 1>;

def SPIRV_SelectionControlAttr :
    SPIRV_BitEnumAttr<"SelectionControl", "valid SPIR-V SelectionControl", "selection_control", [
      SPIRV_SC_None, SPIRV_SC_Flatten, SPIRV_SC_DontFlatten
    ]>;

def SPIRV_SC_UniformConstant         : I32EnumAttrCase<"UniformConstant", 0>;
def SPIRV_SC_Input                   : I32EnumAttrCase<"Input", 1>;
def SPIRV_SC_Uniform                 : I32EnumAttrCase<"Uniform", 2> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_SC_Output                  : I32EnumAttrCase<"Output", 3> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_SC_Workgroup               : I32EnumAttrCase<"Workgroup", 4>;
def SPIRV_SC_CrossWorkgroup          : I32EnumAttrCase<"CrossWorkgroup", 5>;
def SPIRV_SC_Private                 : I32EnumAttrCase<"Private", 6> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader, SPIRV_C_VectorComputeINTEL]>
  ];
}
def SPIRV_SC_Function                : I32EnumAttrCase<"Function", 7>;
def SPIRV_SC_Generic                 : I32EnumAttrCase<"Generic", 8> {
  list<Availability> availability = [
    Capability<[SPIRV_C_GenericPointer]>
  ];
}
def SPIRV_SC_PushConstant            : I32EnumAttrCase<"PushConstant", 9> {
  list<Availability> availability = [
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_SC_AtomicCounter           : I32EnumAttrCase<"AtomicCounter", 10> {
  list<Availability> availability = [
    Capability<[SPIRV_C_AtomicStorage]>
  ];
}
def SPIRV_SC_Image                   : I32EnumAttrCase<"Image", 11>;
def SPIRV_SC_StorageBuffer           : I32EnumAttrCase<"StorageBuffer", 12> {
  list<Availability> availability = [
    Extension<[SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    Capability<[SPIRV_C_Shader]>
  ];
}
def SPIRV_SC_CallableDataKHR         : I32EnumAttrCase<"CallableDataKHR", 5328> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_IncomingCallableDataKHR : I32EnumAttrCase<"IncomingCallableDataKHR", 5329> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_RayPayloadKHR           : I32EnumAttrCase<"RayPayloadKHR", 5338> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_HitAttributeKHR         : I32EnumAttrCase<"HitAttributeKHR", 5339> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_IncomingRayPayloadKHR   : I32EnumAttrCase<"IncomingRayPayloadKHR", 5342> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_ShaderRecordBufferKHR   : I32EnumAttrCase<"ShaderRecordBufferKHR", 5343> {
  list<Availability> availability = [
    Extension<[SPV_KHR_ray_tracing, SPV_NV_ray_tracing]>,
    Capability<[SPIRV_C_RayTracingKHR, SPIRV_C_RayTracingNV]>
  ];
}
def SPIRV_SC_PhysicalStorageBuffer   : I32EnumAttrCase<"PhysicalStorageBuffer", 5349> {
  list<Availability> availability = [
    Extension<[SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer]>,
    Capability<[SPIRV_C_PhysicalStorageBufferAddresses]>
  ];
}
def SPIRV_SC_CodeSectionINTEL        : I32EnumAttrCase<"CodeSectionINTEL", 5605> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_function_pointers]>,
    Capability<[SPIRV_C_FunctionPointersINTEL]>
  ];
}
def SPIRV_SC_DeviceOnlyINTEL         : I32EnumAttrCase<"DeviceOnlyINTEL", 5936> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_usm_storage_classes]>,
    Capability<[SPIRV_C_USMStorageClassesINTEL]>
  ];
}
def SPIRV_SC_HostOnlyINTEL           : I32EnumAttrCase<"HostOnlyINTEL", 5937> {
  list<Availability> availability = [
    Extension<[SPV_INTEL_usm_storage_classes]>,
    Capability<[SPIRV_C_USMStorageClassesINTEL]>
  ];
}

def SPIRV_StorageClassAttr :
    SPIRV_I32EnumAttr<"StorageClass", "valid SPIR-V StorageClass", "storage_class", [
      SPIRV_SC_UniformConstant, SPIRV_SC_Input, SPIRV_SC_Uniform, SPIRV_SC_Output,
      SPIRV_SC_Workgroup, SPIRV_SC_CrossWorkgroup, SPIRV_SC_Private, SPIRV_SC_Function,
      SPIRV_SC_Generic, SPIRV_SC_PushConstant, SPIRV_SC_AtomicCounter, SPIRV_SC_Image,
      SPIRV_SC_StorageBuffer, SPIRV_SC_CallableDataKHR, SPIRV_SC_IncomingCallableDataKHR,
      SPIRV_SC_RayPayloadKHR, SPIRV_SC_HitAttributeKHR, SPIRV_SC_IncomingRayPayloadKHR,
      SPIRV_SC_ShaderRecordBufferKHR, SPIRV_SC_PhysicalStorageBuffer,
      SPIRV_SC_CodeSectionINTEL, SPIRV_SC_DeviceOnlyINTEL, SPIRV_SC_HostOnlyINTEL
    ]>;

def SPIRV_PVF_PackedVectorFormat4x8Bit : I32EnumAttrCase<"PackedVectorFormat4x8Bit", 0> {
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_integer_dot_product]>
  ];
}

def SPIRV_PackedVectorFormatAttr :
    SPIRV_I32EnumAttr<"PackedVectorFormat", "valid SPIR-V PackedVectorFormat", "packed_vector_format", [
      SPIRV_PVF_PackedVectorFormat4x8Bit
    ]>;

// End enum section. Generated from SPIR-V spec; DO NOT MODIFY!

// Enums added manually that are not part of SPIR-V spec

def SPIRV_IDI_NoDepth      : I32EnumAttrCase<"NoDepth", 0>;
def SPIRV_IDI_IsDepth      : I32EnumAttrCase<"IsDepth", 1>;
def SPIRV_IDI_DepthUnknown : I32EnumAttrCase<"DepthUnknown", 2>;

def SPIRV_DepthAttr : SPIRV_I32EnumAttr<
  "ImageDepthInfo", "valid SPIR-V Image Depth specification",
  "image_depth_info", [SPIRV_IDI_NoDepth, SPIRV_IDI_IsDepth, SPIRV_IDI_DepthUnknown]>;

def SPIRV_IAI_NonArrayed : I32EnumAttrCase<"NonArrayed", 0>;
def SPIRV_IAI_Arrayed    : I32EnumAttrCase<"Arrayed", 1>;

def SPIRV_ArrayedAttr : SPIRV_I32EnumAttr<
  "ImageArrayedInfo", "valid SPIR-V Image Arrayed specification",
  "image_arrayed_info", [SPIRV_IAI_NonArrayed, SPIRV_IAI_Arrayed]>;

def SPIRV_ISI_SingleSampled : I32EnumAttrCase<"SingleSampled", 0>;
def SPIRV_ISI_MultiSampled  : I32EnumAttrCase<"MultiSampled", 1>;

def SPIRV_SamplingAttr : SPIRV_I32EnumAttr<
  "ImageSamplingInfo", "valid SPIR-V Image Sampling specification",
  "image_sampling_info", [SPIRV_ISI_SingleSampled, SPIRV_ISI_MultiSampled]>;

def SPIRV_ISUI_SamplerUnknown : I32EnumAttrCase<"SamplerUnknown", 0>;
def SPIRV_ISUI_NeedSampler    : I32EnumAttrCase<"NeedSampler", 1>;
def SPIRV_ISUI_NoSampler      : I32EnumAttrCase<"NoSampler", 2>;

def SPIRV_SamplerUseAttr: SPIRV_I32EnumAttr<
  "ImageSamplerUseInfo", "valid SPIR-V Sampler Use specification",
  "image_sampler_use_info",
  [SPIRV_ISUI_SamplerUnknown, SPIRV_ISUI_NeedSampler, SPIRV_ISUI_NoSampler]>;

def SPIRV_ML_ColumnMajor : I32EnumAttrCase<"ColumnMajor", 0>;
def SPIRV_ML_RowMajor    : I32EnumAttrCase<"RowMajor", 1>;
def SPIRV_ML_PackedA     : I32EnumAttrCase<"PackedA", 2>;
def SPIRV_ML_PackedB     : I32EnumAttrCase<"PackedB", 3>;

def SPIRV_MatrixLayoutAttr :
    SPIRV_I32EnumAttr<"MatrixLayout", "valid SPIR-V MatrixLayout", "matrixLayout", [
      SPIRV_ML_ColumnMajor, SPIRV_ML_RowMajor, SPIRV_ML_PackedA, SPIRV_ML_PackedB
    ]>;

// Cooperative Matrix Use for the SPV_KHR_cooperative_matrix extension.
def SPIRV_KHR_CMU_MatrixA   : I32EnumAttrCase<"MatrixA", 0>;
def SPIRV_KHR_CMU_MatrixB   : I32EnumAttrCase<"MatrixB", 1>;
def SPIRV_KHR_CMU_MatrixAcc : I32EnumAttrCase<"MatrixAcc", 2>;

// NOTE: This is an attribute in the SPIR-V *dialect* but a constant (<id>) in
// SPIR-V proper.
def SPIRV_KHR_CooperativeMatrixUseAttr :
    SPIRV_I32EnumAttr<"CooperativeMatrixUseKHR",
                      "valid SPIR-V Cooperative Matrix Use (KHR)",
                      "coop_matrix_use_khr", [
      SPIRV_KHR_CMU_MatrixA, SPIRV_KHR_CMU_MatrixB, SPIRV_KHR_CMU_MatrixAcc
    ]>;

// Cooperative Matrix Layout for the SPV_KHR_cooperative_matrix extension.
def SPIRV_KHR_CML_RowMajor    : I32EnumAttrCase<"RowMajor", 0>;
def SPIRV_KHR_CML_ColumnMajor : I32EnumAttrCase<"ColumnMajor", 1>;

// NOTE: This is an attribute in the SPIR-V *dialect* but a constant (<id>) in
// SPIR-V proper.
def SPIRV_KHR_CooperativeMatrixLayoutAttr :
    SPIRV_I32EnumAttr<"CooperativeMatrixLayoutKHR",
                      "valid SPIR-V Cooperative Matrix Layout (KHR)",
                      "coop_matrix_layout_khr", [
      SPIRV_KHR_CML_RowMajor, SPIRV_KHR_CML_ColumnMajor
    ]>;

// Cooperative Matrix Operands for the SPV_KHR_cooperative_matrix extension.
def SPIRV_KHR_CMO_None           : I32BitEnumAttrCaseNone<"None">;
def SPIRV_KHR_CMO_MatrixA_Signed : I32BitEnumAttrCaseBit<"ASigned", 0>;
def SPIRV_KHR_CMO_MatrixB_Signed : I32BitEnumAttrCaseBit<"BSigned", 1>;
def SPIRV_KHR_CMO_MatrixC_Signed : I32BitEnumAttrCaseBit<"CSigned", 2>;
def SPIRV_KHR_CMO_Result_Signed  : I32BitEnumAttrCaseBit<"ResultSigned", 3>;
def SPIRV_KHR_CMO_AccSat         : I32BitEnumAttrCaseBit<"AccSat", 4>;

def SPIRV_KHR_CooperativeMatrixOperandsAttr :
    SPIRV_BitEnumAttr<"CooperativeMatrixOperandsKHR",
                      "valid SPIR-V Cooperative Matrix Operands (KHR)",
                      "cooperative_matrix_operands_khr", [
      SPIRV_KHR_CMO_None, SPIRV_KHR_CMO_MatrixA_Signed,
      SPIRV_KHR_CMO_MatrixB_Signed, SPIRV_KHR_CMO_MatrixC_Signed,
      SPIRV_KHR_CMO_Result_Signed, SPIRV_KHR_CMO_AccSat
    ]>;

//===----------------------------------------------------------------------===//
// SPIR-V attribute definitions
//===----------------------------------------------------------------------===//

def SPIRV_VerCapExtAttr : DialectAttr<
    SPIRV_Dialect,
    CPred<"::llvm::isa<::mlir::spirv::VerCapExtAttr>($_self)">,
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
              !interleave(widths, "/") # "-bit signless/unsigned integer">;

def SPIRV_IsArrayType : CPred<"::llvm::isa<::mlir::spirv::ArrayType>($_self)">;
def SPIRV_IsCooperativeMatrixType :
  CPred<"::llvm::isa<::mlir::spirv::CooperativeMatrixType>($_self)">;
def SPIRV_IsImageType : CPred<"::llvm::isa<::mlir::spirv::ImageType>($_self)">;
def SPIRV_IsJointMatrixType :
  CPred<"::llvm::isa<::mlir::spirv::JointMatrixINTELType>($_self)">;
def SPIRV_IsMatrixType : CPred<"::llvm::isa<::mlir::spirv::MatrixType>($_self)">;
def SPIRV_IsPtrType : CPred<"::llvm::isa<::mlir::spirv::PointerType>($_self)">;
def SPIRV_IsRTArrayType : CPred<"::llvm::isa<::mlir::spirv::RuntimeArrayType>($_self)">;
def SPIRV_IsSampledImageType : CPred<"::llvm::isa<::mlir::spirv::SampledImageType>($_self)">;
def SPIRV_IsStructType : CPred<"::llvm::isa<::mlir::spirv::StructType>($_self)">;


// See https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_types
// for the definition of the following types and type categories.

def SPIRV_Void : TypeAlias<NoneType, "void">;
def SPIRV_Bool : TypeAlias<I1, "bool">;
def SPIRV_Integer : AnyIntOfWidths<[8, 16, 32, 64]>;
def SPIRV_Int16 : TypeAlias<I16, "Int16">;
def SPIRV_Int32 : TypeAlias<I32, "Int32">;
def SPIRV_Float32 : TypeAlias<F32, "Float32">;
def SPIRV_Float : FloatOfWidths<[16, 32, 64]>;
def SPIRV_Float16or32 : FloatOfWidths<[16, 32]>;
def SPIRV_Vector : VectorOfLengthAndType<[2, 3, 4, 8, 16],
                                       [SPIRV_Bool, SPIRV_Integer, SPIRV_Float]>;
// Component type check is done in the type parser for the following SPIR-V
// dialect-specific types so we use "Any" here.
def SPIRV_AnyPtr : DialectType<SPIRV_Dialect, SPIRV_IsPtrType,
                             "any SPIR-V pointer type">;
def SPIRV_AnyArray : DialectType<SPIRV_Dialect, SPIRV_IsArrayType,
                               "any SPIR-V array type">;
def SPIRV_AnyCooperativeMatrix : DialectType<SPIRV_Dialect,
                                   SPIRV_IsCooperativeMatrixType,
                                  "any SPIR-V cooperative matrix type">;
def SPIRV_AnyImage : DialectType<SPIRV_Dialect, SPIRV_IsImageType,
                                "any SPIR-V image type">;
def SPIRV_AnyJointMatrix : DialectType<SPIRV_Dialect, SPIRV_IsJointMatrixType,
                                "any SPIR-V joint matrix type">;
def SPIRV_AnyMatrix : DialectType<SPIRV_Dialect, SPIRV_IsMatrixType,
                                "any SPIR-V matrix type">;
def SPIRV_AnyRTArray : DialectType<SPIRV_Dialect, SPIRV_IsRTArrayType,
                                 "any SPIR-V runtime array type">;
def SPIRV_AnyStruct : DialectType<SPIRV_Dialect, SPIRV_IsStructType,
                                "any SPIR-V struct type">;
def SPIRV_AnySampledImage : DialectType<SPIRV_Dialect, SPIRV_IsSampledImageType,
                                "any SPIR-V sampled image type">;

def SPIRV_Numerical : AnyTypeOf<[SPIRV_Integer, SPIRV_Float]>;
def SPIRV_Scalar : AnyTypeOf<[SPIRV_Numerical, SPIRV_Bool]>;
def SPIRV_Aggregate : AnyTypeOf<[SPIRV_AnyArray, SPIRV_AnyRTArray, SPIRV_AnyStruct]>;
def SPIRV_Composite :
    AnyTypeOf<[SPIRV_Vector, SPIRV_AnyArray, SPIRV_AnyRTArray, SPIRV_AnyStruct,
               SPIRV_AnyCooperativeMatrix, SPIRV_AnyJointMatrix, SPIRV_AnyMatrix]>;
def SPIRV_Type : AnyTypeOf<[
    SPIRV_Void, SPIRV_Bool, SPIRV_Integer, SPIRV_Float, SPIRV_Vector,
    SPIRV_AnyPtr, SPIRV_AnyArray, SPIRV_AnyRTArray, SPIRV_AnyStruct,
    SPIRV_AnyCooperativeMatrix, SPIRV_AnyJointMatrix, SPIRV_AnyMatrix,
    SPIRV_AnySampledImage
  ]>;

def SPIRV_SignedInt : SignedIntOfWidths<[8, 16, 32, 64]>;
def SPIRV_SignlessOrUnsignedInt : SignlessOrUnsignedIntOfWidths<[8, 16, 32, 64]>;

class SPIRV_CoopMatrixOfType<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, SPIRV_IsCooperativeMatrixType,
    "::llvm::cast<::mlir::spirv::CooperativeMatrixType>($_self).getElementType()",
    "Cooperative Matrix">;

class SPIRV_JointMatrixOfType<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, SPIRV_IsJointMatrixType,
    "::llvm::cast<::mlir::spirv::JointMatrixINTELType>($_self).getElementType()",
    "Joint Matrix">;

class SPIRV_VectorOf<Type type> :
    VectorOfLengthAndType<[2, 3, 4, 8,16], [type]>;

class SPIRV_ScalarOrVectorOf<Type type> :
    AnyTypeOf<[type, SPIRV_VectorOf<type>]>;

class SPIRV_ScalarOrVectorOrCoopMatrixOf<Type type> :
    AnyTypeOf<[type, SPIRV_VectorOf<type>,
               SPIRV_CoopMatrixOfType<[type]>]>;

class SPIRV_MatrixOrCoopMatrixOf<Type type> :
    AnyTypeOf<[SPIRV_AnyMatrix,
               SPIRV_CoopMatrixOfType<[type]>]>;

def SPIRV_ScalarOrVector : AnyTypeOf<[SPIRV_Scalar, SPIRV_Vector]>;
def SPIRV_ScalarOrVectorOrPtr : AnyTypeOf<[SPIRV_ScalarOrVector, SPIRV_AnyPtr]>;

class SPIRV_Vec4<Type type> : VectorOfLengthAndType<[4], [type]>;
def SPIRV_IntVec4 : SPIRV_Vec4<SPIRV_Integer>;
def SPIRV_IOrUIVec4 : SPIRV_Vec4<SPIRV_SignlessOrUnsignedInt>;
def SPIRV_Int32Vec4 : SPIRV_Vec4<AnyI32>;

// TODO: From 1.4, this should also include Composite type.
def SPIRV_SelectType : AnyTypeOf<[SPIRV_Scalar, SPIRV_Vector, SPIRV_AnyPtr]>;

//===----------------------------------------------------------------------===//
// SPIR-V OpTrait definitions
//===----------------------------------------------------------------------===//

// Check that an op can only be used within the scope of a function-like op.
def InFunctionScope : PredOpTrait<
  "op must appear in a function-like op's block",
  CPred<"isNestedInFunctionOpInterface($_op.getParentOp())">>;

// Check that an op can only be used within the scope of a module-like op.
def InModuleScope : PredOpTrait<
  "op must appear in a module-like op's block",
  CPred<"isDirectInModuleLikeOp($_op.getParentOp())">>;

def UnsignedOp : NativeOpTrait<"spirv::UnsignedOp">;

def SignedOp : NativeOpTrait<"spirv::SignedOp">;

def UsableInSpecConstantOp : NativeOpTrait<"spirv::UsableInSpecConstantOp">;

//===----------------------------------------------------------------------===//
// SPIR-V opcode specification
//===----------------------------------------------------------------------===//

class SPIRV_OpCode<string name, int val> {
  // Name used as reference to retrieve the opcode
  string opname = name;

  // Opcode associated with the name
  int opcode = val;
}

// Begin opcode section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPIRV_OC_OpNop                        : I32EnumAttrCase<"OpNop", 0>;
def SPIRV_OC_OpUndef                      : I32EnumAttrCase<"OpUndef", 1>;
def SPIRV_OC_OpSourceContinued            : I32EnumAttrCase<"OpSourceContinued", 2>;
def SPIRV_OC_OpSource                     : I32EnumAttrCase<"OpSource", 3>;
def SPIRV_OC_OpSourceExtension            : I32EnumAttrCase<"OpSourceExtension", 4>;
def SPIRV_OC_OpName                       : I32EnumAttrCase<"OpName", 5>;
def SPIRV_OC_OpMemberName                 : I32EnumAttrCase<"OpMemberName", 6>;
def SPIRV_OC_OpString                     : I32EnumAttrCase<"OpString", 7>;
def SPIRV_OC_OpLine                       : I32EnumAttrCase<"OpLine", 8>;
def SPIRV_OC_OpExtension                  : I32EnumAttrCase<"OpExtension", 10>;
def SPIRV_OC_OpExtInstImport              : I32EnumAttrCase<"OpExtInstImport", 11>;
def SPIRV_OC_OpExtInst                    : I32EnumAttrCase<"OpExtInst", 12>;
def SPIRV_OC_OpMemoryModel                : I32EnumAttrCase<"OpMemoryModel", 14>;
def SPIRV_OC_OpEntryPoint                 : I32EnumAttrCase<"OpEntryPoint", 15>;
def SPIRV_OC_OpExecutionMode              : I32EnumAttrCase<"OpExecutionMode", 16>;
def SPIRV_OC_OpCapability                 : I32EnumAttrCase<"OpCapability", 17>;
def SPIRV_OC_OpTypeVoid                   : I32EnumAttrCase<"OpTypeVoid", 19>;
def SPIRV_OC_OpTypeBool                   : I32EnumAttrCase<"OpTypeBool", 20>;
def SPIRV_OC_OpTypeInt                    : I32EnumAttrCase<"OpTypeInt", 21>;
def SPIRV_OC_OpTypeFloat                  : I32EnumAttrCase<"OpTypeFloat", 22>;
def SPIRV_OC_OpTypeVector                 : I32EnumAttrCase<"OpTypeVector", 23>;
def SPIRV_OC_OpTypeMatrix                 : I32EnumAttrCase<"OpTypeMatrix", 24>;
def SPIRV_OC_OpTypeImage                  : I32EnumAttrCase<"OpTypeImage", 25>;
def SPIRV_OC_OpTypeSampledImage           : I32EnumAttrCase<"OpTypeSampledImage", 27>;
def SPIRV_OC_OpTypeArray                  : I32EnumAttrCase<"OpTypeArray", 28>;
def SPIRV_OC_OpTypeRuntimeArray           : I32EnumAttrCase<"OpTypeRuntimeArray", 29>;
def SPIRV_OC_OpTypeStruct                 : I32EnumAttrCase<"OpTypeStruct", 30>;
def SPIRV_OC_OpTypePointer                : I32EnumAttrCase<"OpTypePointer", 32>;
def SPIRV_OC_OpTypeFunction               : I32EnumAttrCase<"OpTypeFunction", 33>;
def SPIRV_OC_OpTypeForwardPointer         : I32EnumAttrCase<"OpTypeForwardPointer", 39>;
def SPIRV_OC_OpConstantTrue               : I32EnumAttrCase<"OpConstantTrue", 41>;
def SPIRV_OC_OpConstantFalse              : I32EnumAttrCase<"OpConstantFalse", 42>;
def SPIRV_OC_OpConstant                   : I32EnumAttrCase<"OpConstant", 43>;
def SPIRV_OC_OpConstantComposite          : I32EnumAttrCase<"OpConstantComposite", 44>;
def SPIRV_OC_OpConstantNull               : I32EnumAttrCase<"OpConstantNull", 46>;
def SPIRV_OC_OpSpecConstantTrue           : I32EnumAttrCase<"OpSpecConstantTrue", 48>;
def SPIRV_OC_OpSpecConstantFalse          : I32EnumAttrCase<"OpSpecConstantFalse", 49>;
def SPIRV_OC_OpSpecConstant               : I32EnumAttrCase<"OpSpecConstant", 50>;
def SPIRV_OC_OpSpecConstantComposite      : I32EnumAttrCase<"OpSpecConstantComposite", 51>;
def SPIRV_OC_OpSpecConstantOp             : I32EnumAttrCase<"OpSpecConstantOp", 52>;
def SPIRV_OC_OpFunction                   : I32EnumAttrCase<"OpFunction", 54>;
def SPIRV_OC_OpFunctionParameter          : I32EnumAttrCase<"OpFunctionParameter", 55>;
def SPIRV_OC_OpFunctionEnd                : I32EnumAttrCase<"OpFunctionEnd", 56>;
def SPIRV_OC_OpFunctionCall               : I32EnumAttrCase<"OpFunctionCall", 57>;
def SPIRV_OC_OpVariable                   : I32EnumAttrCase<"OpVariable", 59>;
def SPIRV_OC_OpLoad                       : I32EnumAttrCase<"OpLoad", 61>;
def SPIRV_OC_OpStore                      : I32EnumAttrCase<"OpStore", 62>;
def SPIRV_OC_OpCopyMemory                 : I32EnumAttrCase<"OpCopyMemory", 63>;
def SPIRV_OC_OpAccessChain                : I32EnumAttrCase<"OpAccessChain", 65>;
def SPIRV_OC_OpPtrAccessChain             : I32EnumAttrCase<"OpPtrAccessChain", 67>;
def SPIRV_OC_OpInBoundsPtrAccessChain     : I32EnumAttrCase<"OpInBoundsPtrAccessChain", 70>;
def SPIRV_OC_OpDecorate                   : I32EnumAttrCase<"OpDecorate", 71>;
def SPIRV_OC_OpMemberDecorate             : I32EnumAttrCase<"OpMemberDecorate", 72>;
def SPIRV_OC_OpVectorExtractDynamic       : I32EnumAttrCase<"OpVectorExtractDynamic", 77>;
def SPIRV_OC_OpVectorInsertDynamic        : I32EnumAttrCase<"OpVectorInsertDynamic", 78>;
def SPIRV_OC_OpVectorShuffle              : I32EnumAttrCase<"OpVectorShuffle", 79>;
def SPIRV_OC_OpCompositeConstruct         : I32EnumAttrCase<"OpCompositeConstruct", 80>;
def SPIRV_OC_OpCompositeExtract           : I32EnumAttrCase<"OpCompositeExtract", 81>;
def SPIRV_OC_OpCompositeInsert            : I32EnumAttrCase<"OpCompositeInsert", 82>;
def SPIRV_OC_OpTranspose                  : I32EnumAttrCase<"OpTranspose", 84>;
def SPIRV_OC_OpImageDrefGather            : I32EnumAttrCase<"OpImageDrefGather", 97>;
def SPIRV_OC_OpImage                      : I32EnumAttrCase<"OpImage", 100>;
def SPIRV_OC_OpImageQuerySize             : I32EnumAttrCase<"OpImageQuerySize", 104>;
def SPIRV_OC_OpConvertFToU                : I32EnumAttrCase<"OpConvertFToU", 109>;
def SPIRV_OC_OpConvertFToS                : I32EnumAttrCase<"OpConvertFToS", 110>;
def SPIRV_OC_OpConvertSToF                : I32EnumAttrCase<"OpConvertSToF", 111>;
def SPIRV_OC_OpConvertUToF                : I32EnumAttrCase<"OpConvertUToF", 112>;
def SPIRV_OC_OpUConvert                   : I32EnumAttrCase<"OpUConvert", 113>;
def SPIRV_OC_OpSConvert                   : I32EnumAttrCase<"OpSConvert", 114>;
def SPIRV_OC_OpFConvert                   : I32EnumAttrCase<"OpFConvert", 115>;
def SPIRV_OC_OpConvertPtrToU              : I32EnumAttrCase<"OpConvertPtrToU", 117>;
def SPIRV_OC_OpConvertUToPtr              : I32EnumAttrCase<"OpConvertUToPtr", 120>;
def SPIRV_OC_OpPtrCastToGeneric           : I32EnumAttrCase<"OpPtrCastToGeneric", 121>;
def SPIRV_OC_OpGenericCastToPtr           : I32EnumAttrCase<"OpGenericCastToPtr", 122>;
def SPIRV_OC_OpGenericCastToPtrExplicit   : I32EnumAttrCase<"OpGenericCastToPtrExplicit", 123>;
def SPIRV_OC_OpBitcast                    : I32EnumAttrCase<"OpBitcast", 124>;
def SPIRV_OC_OpSNegate                    : I32EnumAttrCase<"OpSNegate", 126>;
def SPIRV_OC_OpFNegate                    : I32EnumAttrCase<"OpFNegate", 127>;
def SPIRV_OC_OpIAdd                       : I32EnumAttrCase<"OpIAdd", 128>;
def SPIRV_OC_OpFAdd                       : I32EnumAttrCase<"OpFAdd", 129>;
def SPIRV_OC_OpISub                       : I32EnumAttrCase<"OpISub", 130>;
def SPIRV_OC_OpFSub                       : I32EnumAttrCase<"OpFSub", 131>;
def SPIRV_OC_OpIMul                       : I32EnumAttrCase<"OpIMul", 132>;
def SPIRV_OC_OpFMul                       : I32EnumAttrCase<"OpFMul", 133>;
def SPIRV_OC_OpUDiv                       : I32EnumAttrCase<"OpUDiv", 134>;
def SPIRV_OC_OpSDiv                       : I32EnumAttrCase<"OpSDiv", 135>;
def SPIRV_OC_OpFDiv                       : I32EnumAttrCase<"OpFDiv", 136>;
def SPIRV_OC_OpUMod                       : I32EnumAttrCase<"OpUMod", 137>;
def SPIRV_OC_OpSRem                       : I32EnumAttrCase<"OpSRem", 138>;
def SPIRV_OC_OpSMod                       : I32EnumAttrCase<"OpSMod", 139>;
def SPIRV_OC_OpFRem                       : I32EnumAttrCase<"OpFRem", 140>;
def SPIRV_OC_OpFMod                       : I32EnumAttrCase<"OpFMod", 141>;
def SPIRV_OC_OpVectorTimesScalar          : I32EnumAttrCase<"OpVectorTimesScalar", 142>;
def SPIRV_OC_OpMatrixTimesScalar          : I32EnumAttrCase<"OpMatrixTimesScalar", 143>;
def SPIRV_OC_OpMatrixTimesMatrix          : I32EnumAttrCase<"OpMatrixTimesMatrix", 146>;
def SPIRV_OC_OpDot                        : I32EnumAttrCase<"OpDot", 148>;
def SPIRV_OC_OpIAddCarry                  : I32EnumAttrCase<"OpIAddCarry", 149>;
def SPIRV_OC_OpISubBorrow                 : I32EnumAttrCase<"OpISubBorrow", 150>;
def SPIRV_OC_OpUMulExtended               : I32EnumAttrCase<"OpUMulExtended", 151>;
def SPIRV_OC_OpSMulExtended               : I32EnumAttrCase<"OpSMulExtended", 152>;
def SPIRV_OC_OpIsNan                      : I32EnumAttrCase<"OpIsNan", 156>;
def SPIRV_OC_OpIsInf                      : I32EnumAttrCase<"OpIsInf", 157>;
def SPIRV_OC_OpOrdered                    : I32EnumAttrCase<"OpOrdered", 162>;
def SPIRV_OC_OpUnordered                  : I32EnumAttrCase<"OpUnordered", 163>;
def SPIRV_OC_OpLogicalEqual               : I32EnumAttrCase<"OpLogicalEqual", 164>;
def SPIRV_OC_OpLogicalNotEqual            : I32EnumAttrCase<"OpLogicalNotEqual", 165>;
def SPIRV_OC_OpLogicalOr                  : I32EnumAttrCase<"OpLogicalOr", 166>;
def SPIRV_OC_OpLogicalAnd                 : I32EnumAttrCase<"OpLogicalAnd", 167>;
def SPIRV_OC_OpLogicalNot                 : I32EnumAttrCase<"OpLogicalNot", 168>;
def SPIRV_OC_OpSelect                     : I32EnumAttrCase<"OpSelect", 169>;
def SPIRV_OC_OpIEqual                     : I32EnumAttrCase<"OpIEqual", 170>;
def SPIRV_OC_OpINotEqual                  : I32EnumAttrCase<"OpINotEqual", 171>;
def SPIRV_OC_OpUGreaterThan               : I32EnumAttrCase<"OpUGreaterThan", 172>;
def SPIRV_OC_OpSGreaterThan               : I32EnumAttrCase<"OpSGreaterThan", 173>;
def SPIRV_OC_OpUGreaterThanEqual          : I32EnumAttrCase<"OpUGreaterThanEqual", 174>;
def SPIRV_OC_OpSGreaterThanEqual          : I32EnumAttrCase<"OpSGreaterThanEqual", 175>;
def SPIRV_OC_OpULessThan                  : I32EnumAttrCase<"OpULessThan", 176>;
def SPIRV_OC_OpSLessThan                  : I32EnumAttrCase<"OpSLessThan", 177>;
def SPIRV_OC_OpULessThanEqual             : I32EnumAttrCase<"OpULessThanEqual", 178>;
def SPIRV_OC_OpSLessThanEqual             : I32EnumAttrCase<"OpSLessThanEqual", 179>;
def SPIRV_OC_OpFOrdEqual                  : I32EnumAttrCase<"OpFOrdEqual", 180>;
def SPIRV_OC_OpFUnordEqual                : I32EnumAttrCase<"OpFUnordEqual", 181>;
def SPIRV_OC_OpFOrdNotEqual               : I32EnumAttrCase<"OpFOrdNotEqual", 182>;
def SPIRV_OC_OpFUnordNotEqual             : I32EnumAttrCase<"OpFUnordNotEqual", 183>;
def SPIRV_OC_OpFOrdLessThan               : I32EnumAttrCase<"OpFOrdLessThan", 184>;
def SPIRV_OC_OpFUnordLessThan             : I32EnumAttrCase<"OpFUnordLessThan", 185>;
def SPIRV_OC_OpFOrdGreaterThan            : I32EnumAttrCase<"OpFOrdGreaterThan", 186>;
def SPIRV_OC_OpFUnordGreaterThan          : I32EnumAttrCase<"OpFUnordGreaterThan", 187>;
def SPIRV_OC_OpFOrdLessThanEqual          : I32EnumAttrCase<"OpFOrdLessThanEqual", 188>;
def SPIRV_OC_OpFUnordLessThanEqual        : I32EnumAttrCase<"OpFUnordLessThanEqual", 189>;
def SPIRV_OC_OpFOrdGreaterThanEqual       : I32EnumAttrCase<"OpFOrdGreaterThanEqual", 190>;
def SPIRV_OC_OpFUnordGreaterThanEqual     : I32EnumAttrCase<"OpFUnordGreaterThanEqual", 191>;
def SPIRV_OC_OpShiftRightLogical          : I32EnumAttrCase<"OpShiftRightLogical", 194>;
def SPIRV_OC_OpShiftRightArithmetic       : I32EnumAttrCase<"OpShiftRightArithmetic", 195>;
def SPIRV_OC_OpShiftLeftLogical           : I32EnumAttrCase<"OpShiftLeftLogical", 196>;
def SPIRV_OC_OpBitwiseOr                  : I32EnumAttrCase<"OpBitwiseOr", 197>;
def SPIRV_OC_OpBitwiseXor                 : I32EnumAttrCase<"OpBitwiseXor", 198>;
def SPIRV_OC_OpBitwiseAnd                 : I32EnumAttrCase<"OpBitwiseAnd", 199>;
def SPIRV_OC_OpNot                        : I32EnumAttrCase<"OpNot", 200>;
def SPIRV_OC_OpBitFieldInsert             : I32EnumAttrCase<"OpBitFieldInsert", 201>;
def SPIRV_OC_OpBitFieldSExtract           : I32EnumAttrCase<"OpBitFieldSExtract", 202>;
def SPIRV_OC_OpBitFieldUExtract           : I32EnumAttrCase<"OpBitFieldUExtract", 203>;
def SPIRV_OC_OpBitReverse                 : I32EnumAttrCase<"OpBitReverse", 204>;
def SPIRV_OC_OpBitCount                   : I32EnumAttrCase<"OpBitCount", 205>;
def SPIRV_OC_OpControlBarrier             : I32EnumAttrCase<"OpControlBarrier", 224>;
def SPIRV_OC_OpMemoryBarrier              : I32EnumAttrCase<"OpMemoryBarrier", 225>;
def SPIRV_OC_OpAtomicExchange             : I32EnumAttrCase<"OpAtomicExchange", 229>;
def SPIRV_OC_OpAtomicCompareExchange      : I32EnumAttrCase<"OpAtomicCompareExchange", 230>;
def SPIRV_OC_OpAtomicCompareExchangeWeak  : I32EnumAttrCase<"OpAtomicCompareExchangeWeak", 231>;
def SPIRV_OC_OpAtomicIIncrement           : I32EnumAttrCase<"OpAtomicIIncrement", 232>;
def SPIRV_OC_OpAtomicIDecrement           : I32EnumAttrCase<"OpAtomicIDecrement", 233>;
def SPIRV_OC_OpAtomicIAdd                 : I32EnumAttrCase<"OpAtomicIAdd", 234>;
def SPIRV_OC_OpAtomicISub                 : I32EnumAttrCase<"OpAtomicISub", 235>;
def SPIRV_OC_OpAtomicSMin                 : I32EnumAttrCase<"OpAtomicSMin", 236>;
def SPIRV_OC_OpAtomicUMin                 : I32EnumAttrCase<"OpAtomicUMin", 237>;
def SPIRV_OC_OpAtomicSMax                 : I32EnumAttrCase<"OpAtomicSMax", 238>;
def SPIRV_OC_OpAtomicUMax                 : I32EnumAttrCase<"OpAtomicUMax", 239>;
def SPIRV_OC_OpAtomicAnd                  : I32EnumAttrCase<"OpAtomicAnd", 240>;
def SPIRV_OC_OpAtomicOr                   : I32EnumAttrCase<"OpAtomicOr", 241>;
def SPIRV_OC_OpAtomicXor                  : I32EnumAttrCase<"OpAtomicXor", 242>;
def SPIRV_OC_OpPhi                        : I32EnumAttrCase<"OpPhi", 245>;
def SPIRV_OC_OpLoopMerge                  : I32EnumAttrCase<"OpLoopMerge", 246>;
def SPIRV_OC_OpSelectionMerge             : I32EnumAttrCase<"OpSelectionMerge", 247>;
def SPIRV_OC_OpLabel                      : I32EnumAttrCase<"OpLabel", 248>;
def SPIRV_OC_OpBranch                     : I32EnumAttrCase<"OpBranch", 249>;
def SPIRV_OC_OpBranchConditional          : I32EnumAttrCase<"OpBranchConditional", 250>;
def SPIRV_OC_OpReturn                     : I32EnumAttrCase<"OpReturn", 253>;
def SPIRV_OC_OpReturnValue                : I32EnumAttrCase<"OpReturnValue", 254>;
def SPIRV_OC_OpUnreachable                : I32EnumAttrCase<"OpUnreachable", 255>;
def SPIRV_OC_OpGroupBroadcast             : I32EnumAttrCase<"OpGroupBroadcast", 263>;
def SPIRV_OC_OpGroupIAdd                  : I32EnumAttrCase<"OpGroupIAdd", 264>;
def SPIRV_OC_OpGroupFAdd                  : I32EnumAttrCase<"OpGroupFAdd", 265>;
def SPIRV_OC_OpGroupFMin                  : I32EnumAttrCase<"OpGroupFMin", 266>;
def SPIRV_OC_OpGroupUMin                  : I32EnumAttrCase<"OpGroupUMin", 267>;
def SPIRV_OC_OpGroupSMin                  : I32EnumAttrCase<"OpGroupSMin", 268>;
def SPIRV_OC_OpGroupFMax                  : I32EnumAttrCase<"OpGroupFMax", 269>;
def SPIRV_OC_OpGroupUMax                  : I32EnumAttrCase<"OpGroupUMax", 270>;
def SPIRV_OC_OpGroupSMax                  : I32EnumAttrCase<"OpGroupSMax", 271>;
def SPIRV_OC_OpNoLine                     : I32EnumAttrCase<"OpNoLine", 317>;
def SPIRV_OC_OpModuleProcessed            : I32EnumAttrCase<"OpModuleProcessed", 330>;
def SPIRV_OC_OpGroupNonUniformElect       : I32EnumAttrCase<"OpGroupNonUniformElect", 333>;
def SPIRV_OC_OpGroupNonUniformBroadcast   : I32EnumAttrCase<"OpGroupNonUniformBroadcast", 337>;
def SPIRV_OC_OpGroupNonUniformBallot      : I32EnumAttrCase<"OpGroupNonUniformBallot", 339>;
def SPIRV_OC_OpGroupNonUniformShuffle     : I32EnumAttrCase<"OpGroupNonUniformShuffle", 345>;
def SPIRV_OC_OpGroupNonUniformShuffleXor  : I32EnumAttrCase<"OpGroupNonUniformShuffleXor", 346>;
def SPIRV_OC_OpGroupNonUniformShuffleUp   : I32EnumAttrCase<"OpGroupNonUniformShuffleUp", 347>;
def SPIRV_OC_OpGroupNonUniformShuffleDown : I32EnumAttrCase<"OpGroupNonUniformShuffleDown", 348>;
def SPIRV_OC_OpGroupNonUniformIAdd        : I32EnumAttrCase<"OpGroupNonUniformIAdd", 349>;
def SPIRV_OC_OpGroupNonUniformFAdd        : I32EnumAttrCase<"OpGroupNonUniformFAdd", 350>;
def SPIRV_OC_OpGroupNonUniformIMul        : I32EnumAttrCase<"OpGroupNonUniformIMul", 351>;
def SPIRV_OC_OpGroupNonUniformFMul        : I32EnumAttrCase<"OpGroupNonUniformFMul", 352>;
def SPIRV_OC_OpGroupNonUniformSMin        : I32EnumAttrCase<"OpGroupNonUniformSMin", 353>;
def SPIRV_OC_OpGroupNonUniformUMin        : I32EnumAttrCase<"OpGroupNonUniformUMin", 354>;
def SPIRV_OC_OpGroupNonUniformFMin        : I32EnumAttrCase<"OpGroupNonUniformFMin", 355>;
def SPIRV_OC_OpGroupNonUniformSMax        : I32EnumAttrCase<"OpGroupNonUniformSMax", 356>;
def SPIRV_OC_OpGroupNonUniformUMax        : I32EnumAttrCase<"OpGroupNonUniformUMax", 357>;
def SPIRV_OC_OpGroupNonUniformFMax        : I32EnumAttrCase<"OpGroupNonUniformFMax", 358>;
def SPIRV_OC_OpGroupNonUniformBitwiseAnd  : I32EnumAttrCase<"OpGroupNonUniformBitwiseAnd", 359>;
def SPIRV_OC_OpGroupNonUniformBitwiseOr   : I32EnumAttrCase<"OpGroupNonUniformBitwiseOr", 360>;
def SPIRV_OC_OpGroupNonUniformBitwiseXor  : I32EnumAttrCase<"OpGroupNonUniformBitwiseXor", 361>;
def SPIRV_OC_OpGroupNonUniformLogicalAnd  : I32EnumAttrCase<"OpGroupNonUniformLogicalAnd", 362>;
def SPIRV_OC_OpGroupNonUniformLogicalOr   : I32EnumAttrCase<"OpGroupNonUniformLogicalOr", 363>;
def SPIRV_OC_OpGroupNonUniformLogicalXor  : I32EnumAttrCase<"OpGroupNonUniformLogicalXor", 364>;
def SPIRV_OC_OpSubgroupBallotKHR          : I32EnumAttrCase<"OpSubgroupBallotKHR", 4421>;
def SPIRV_OC_OpSDot                       : I32EnumAttrCase<"OpSDot", 4450>;
def SPIRV_OC_OpUDot                       : I32EnumAttrCase<"OpUDot", 4451>;
def SPIRV_OC_OpSUDot                      : I32EnumAttrCase<"OpSUDot", 4452>;
def SPIRV_OC_OpSDotAccSat                 : I32EnumAttrCase<"OpSDotAccSat", 4453>;
def SPIRV_OC_OpUDotAccSat                 : I32EnumAttrCase<"OpUDotAccSat", 4454>;
def SPIRV_OC_OpSUDotAccSat                : I32EnumAttrCase<"OpSUDotAccSat", 4455>;
def SPIRV_OC_OpTypeCooperativeMatrixKHR   : I32EnumAttrCase<"OpTypeCooperativeMatrixKHR", 4456>;
def SPIRV_OC_OpCooperativeMatrixLoadKHR   : I32EnumAttrCase<"OpCooperativeMatrixLoadKHR", 4457>;
def SPIRV_OC_OpCooperativeMatrixStoreKHR  : I32EnumAttrCase<"OpCooperativeMatrixStoreKHR", 4458>;
def SPIRV_OC_OpCooperativeMatrixMulAddKHR : I32EnumAttrCase<"OpCooperativeMatrixMulAddKHR", 4459>;
def SPIRV_OC_OpCooperativeMatrixLengthKHR : I32EnumAttrCase<"OpCooperativeMatrixLengthKHR", 4460>;
def SPIRV_OC_OpSubgroupBlockReadINTEL     : I32EnumAttrCase<"OpSubgroupBlockReadINTEL", 5575>;
def SPIRV_OC_OpSubgroupBlockWriteINTEL    : I32EnumAttrCase<"OpSubgroupBlockWriteINTEL", 5576>;
def SPIRV_OC_OpAssumeTrueKHR              : I32EnumAttrCase<"OpAssumeTrueKHR", 5630>;
def SPIRV_OC_OpAtomicFAddEXT              : I32EnumAttrCase<"OpAtomicFAddEXT", 6035>;
def SPIRV_OC_OpGroupIMulKHR               : I32EnumAttrCase<"OpGroupIMulKHR", 6401>;
def SPIRV_OC_OpGroupFMulKHR               : I32EnumAttrCase<"OpGroupFMulKHR", 6402>;

def SPIRV_OC_OpTypeJointMatrixINTEL       : I32EnumAttrCase<"OpTypeJointMatrixINTEL", 6119>;
def SPIRV_OC_OpJointMatrixLoadINTEL       : I32EnumAttrCase<"OpJointMatrixLoadINTEL", 6120>;
def SPIRV_OC_OpJointMatrixStoreINTEL      : I32EnumAttrCase<"OpJointMatrixStoreINTEL", 6121>;
def SPIRV_OC_OpJointMatrixMadINTEL        : I32EnumAttrCase<"OpJointMatrixMadINTEL", 6122>;
def SPIRV_OC_OpTypejointMatrixWorkItemLengthINTEL : I32EnumAttrCase<"OpJointMatrixWorkItemLengthINTEL", 6410>;

def SPIRV_OC_OpConvertFToBF16INTEL        : I32EnumAttrCase<"OpConvertFToBF16INTEL", 6116>;
def SPIRV_OC_OpConvertBF16ToFINTEL        : I32EnumAttrCase<"OpConvertBF16ToFINTEL", 6117>;

def SPIRV_OpcodeAttr :
    SPIRV_I32EnumAttr<"Opcode", "valid SPIR-V instructions", "opcode", [
      SPIRV_OC_OpNop, SPIRV_OC_OpUndef, SPIRV_OC_OpSourceContinued,
      SPIRV_OC_OpSource, SPIRV_OC_OpSourceExtension, SPIRV_OC_OpName,
      SPIRV_OC_OpMemberName, SPIRV_OC_OpString, SPIRV_OC_OpLine,
      SPIRV_OC_OpExtension, SPIRV_OC_OpExtInstImport, SPIRV_OC_OpExtInst,
      SPIRV_OC_OpMemoryModel, SPIRV_OC_OpEntryPoint, SPIRV_OC_OpExecutionMode,
      SPIRV_OC_OpCapability, SPIRV_OC_OpTypeVoid, SPIRV_OC_OpTypeBool,
      SPIRV_OC_OpTypeInt, SPIRV_OC_OpTypeFloat, SPIRV_OC_OpTypeVector,
      SPIRV_OC_OpTypeMatrix, SPIRV_OC_OpTypeImage, SPIRV_OC_OpTypeSampledImage,
      SPIRV_OC_OpTypeArray, SPIRV_OC_OpTypeRuntimeArray, SPIRV_OC_OpTypeStruct,
      SPIRV_OC_OpTypePointer, SPIRV_OC_OpTypeFunction, SPIRV_OC_OpTypeForwardPointer,
      SPIRV_OC_OpConstantTrue, SPIRV_OC_OpConstantFalse, SPIRV_OC_OpConstant,
      SPIRV_OC_OpConstantComposite, SPIRV_OC_OpConstantNull,
      SPIRV_OC_OpSpecConstantTrue, SPIRV_OC_OpSpecConstantFalse,
      SPIRV_OC_OpSpecConstant, SPIRV_OC_OpSpecConstantComposite,
      SPIRV_OC_OpSpecConstantOp, SPIRV_OC_OpFunction, SPIRV_OC_OpFunctionParameter,
      SPIRV_OC_OpFunctionEnd, SPIRV_OC_OpFunctionCall, SPIRV_OC_OpVariable,
      SPIRV_OC_OpLoad, SPIRV_OC_OpStore, SPIRV_OC_OpCopyMemory,
      SPIRV_OC_OpAccessChain, SPIRV_OC_OpPtrAccessChain,
      SPIRV_OC_OpInBoundsPtrAccessChain, SPIRV_OC_OpDecorate,
      SPIRV_OC_OpMemberDecorate, SPIRV_OC_OpVectorExtractDynamic,
      SPIRV_OC_OpVectorInsertDynamic, SPIRV_OC_OpVectorShuffle,
      SPIRV_OC_OpCompositeConstruct, SPIRV_OC_OpCompositeExtract,
      SPIRV_OC_OpCompositeInsert, SPIRV_OC_OpTranspose, SPIRV_OC_OpImageDrefGather,
      SPIRV_OC_OpImage, SPIRV_OC_OpImageQuerySize, SPIRV_OC_OpConvertFToU,
      SPIRV_OC_OpConvertFToS, SPIRV_OC_OpConvertSToF, SPIRV_OC_OpConvertUToF,
      SPIRV_OC_OpUConvert, SPIRV_OC_OpSConvert, SPIRV_OC_OpFConvert,
      SPIRV_OC_OpConvertPtrToU, SPIRV_OC_OpConvertUToPtr,
      SPIRV_OC_OpPtrCastToGeneric, SPIRV_OC_OpGenericCastToPtr,
      SPIRV_OC_OpGenericCastToPtrExplicit, SPIRV_OC_OpBitcast, SPIRV_OC_OpSNegate,
      SPIRV_OC_OpFNegate, SPIRV_OC_OpIAdd, SPIRV_OC_OpFAdd, SPIRV_OC_OpISub,
      SPIRV_OC_OpFSub, SPIRV_OC_OpIMul, SPIRV_OC_OpFMul, SPIRV_OC_OpUDiv,
      SPIRV_OC_OpSDiv, SPIRV_OC_OpFDiv, SPIRV_OC_OpUMod, SPIRV_OC_OpSRem,
      SPIRV_OC_OpSMod, SPIRV_OC_OpFRem, SPIRV_OC_OpFMod,
      SPIRV_OC_OpVectorTimesScalar, SPIRV_OC_OpMatrixTimesScalar, SPIRV_OC_OpDot,
      SPIRV_OC_OpMatrixTimesMatrix, SPIRV_OC_OpIAddCarry, SPIRV_OC_OpISubBorrow,
      SPIRV_OC_OpUMulExtended, SPIRV_OC_OpSMulExtended, SPIRV_OC_OpIsNan,
      SPIRV_OC_OpIsInf, SPIRV_OC_OpOrdered, SPIRV_OC_OpUnordered,
      SPIRV_OC_OpLogicalEqual, SPIRV_OC_OpLogicalNotEqual, SPIRV_OC_OpLogicalOr,
      SPIRV_OC_OpLogicalAnd, SPIRV_OC_OpLogicalNot, SPIRV_OC_OpSelect,
      SPIRV_OC_OpIEqual, SPIRV_OC_OpINotEqual, SPIRV_OC_OpUGreaterThan,
      SPIRV_OC_OpSGreaterThan, SPIRV_OC_OpUGreaterThanEqual,
      SPIRV_OC_OpSGreaterThanEqual, SPIRV_OC_OpULessThan, SPIRV_OC_OpSLessThan,
      SPIRV_OC_OpULessThanEqual, SPIRV_OC_OpSLessThanEqual, SPIRV_OC_OpFOrdEqual,
      SPIRV_OC_OpFUnordEqual, SPIRV_OC_OpFOrdNotEqual, SPIRV_OC_OpFUnordNotEqual,
      SPIRV_OC_OpFOrdLessThan, SPIRV_OC_OpFUnordLessThan, SPIRV_OC_OpFOrdGreaterThan,
      SPIRV_OC_OpFUnordGreaterThan, SPIRV_OC_OpFOrdLessThanEqual,
      SPIRV_OC_OpFUnordLessThanEqual, SPIRV_OC_OpFOrdGreaterThanEqual,
      SPIRV_OC_OpFUnordGreaterThanEqual, SPIRV_OC_OpShiftRightLogical,
      SPIRV_OC_OpShiftRightArithmetic, SPIRV_OC_OpShiftLeftLogical,
      SPIRV_OC_OpBitwiseOr, SPIRV_OC_OpBitwiseXor, SPIRV_OC_OpBitwiseAnd,
      SPIRV_OC_OpNot, SPIRV_OC_OpBitFieldInsert, SPIRV_OC_OpBitFieldSExtract,
      SPIRV_OC_OpBitFieldUExtract, SPIRV_OC_OpBitReverse, SPIRV_OC_OpBitCount,
      SPIRV_OC_OpControlBarrier, SPIRV_OC_OpMemoryBarrier, SPIRV_OC_OpAtomicExchange,
      SPIRV_OC_OpAtomicCompareExchange, SPIRV_OC_OpAtomicCompareExchangeWeak,
      SPIRV_OC_OpAtomicIIncrement, SPIRV_OC_OpAtomicIDecrement,
      SPIRV_OC_OpAtomicIAdd, SPIRV_OC_OpAtomicISub, SPIRV_OC_OpAtomicSMin,
      SPIRV_OC_OpAtomicUMin, SPIRV_OC_OpAtomicSMax, SPIRV_OC_OpAtomicUMax,
      SPIRV_OC_OpAtomicAnd, SPIRV_OC_OpAtomicOr, SPIRV_OC_OpAtomicXor,
      SPIRV_OC_OpPhi, SPIRV_OC_OpLoopMerge, SPIRV_OC_OpSelectionMerge,
      SPIRV_OC_OpLabel, SPIRV_OC_OpBranch, SPIRV_OC_OpBranchConditional,
      SPIRV_OC_OpReturn, SPIRV_OC_OpReturnValue, SPIRV_OC_OpUnreachable,
      SPIRV_OC_OpGroupBroadcast, SPIRV_OC_OpGroupIAdd, SPIRV_OC_OpGroupFAdd,
      SPIRV_OC_OpGroupFMin, SPIRV_OC_OpGroupUMin, SPIRV_OC_OpGroupSMin,
      SPIRV_OC_OpGroupFMax, SPIRV_OC_OpGroupUMax, SPIRV_OC_OpGroupSMax,
      SPIRV_OC_OpNoLine, SPIRV_OC_OpModuleProcessed, SPIRV_OC_OpGroupNonUniformElect,
      SPIRV_OC_OpGroupNonUniformBroadcast, SPIRV_OC_OpGroupNonUniformBallot,
      SPIRV_OC_OpGroupNonUniformShuffle, SPIRV_OC_OpGroupNonUniformShuffleXor,
      SPIRV_OC_OpGroupNonUniformShuffleUp, SPIRV_OC_OpGroupNonUniformShuffleDown,
      SPIRV_OC_OpGroupNonUniformIAdd, SPIRV_OC_OpGroupNonUniformFAdd,
      SPIRV_OC_OpGroupNonUniformIMul, SPIRV_OC_OpGroupNonUniformFMul,
      SPIRV_OC_OpGroupNonUniformSMin, SPIRV_OC_OpGroupNonUniformUMin,
      SPIRV_OC_OpGroupNonUniformFMin, SPIRV_OC_OpGroupNonUniformSMax,
      SPIRV_OC_OpGroupNonUniformUMax, SPIRV_OC_OpGroupNonUniformFMax,
      SPIRV_OC_OpGroupNonUniformBitwiseAnd, SPIRV_OC_OpGroupNonUniformBitwiseOr,
      SPIRV_OC_OpGroupNonUniformBitwiseXor, SPIRV_OC_OpGroupNonUniformLogicalAnd,
      SPIRV_OC_OpGroupNonUniformLogicalOr, SPIRV_OC_OpGroupNonUniformLogicalXor,
      SPIRV_OC_OpSubgroupBallotKHR, SPIRV_OC_OpSDot, SPIRV_OC_OpUDot,
      SPIRV_OC_OpSUDot, SPIRV_OC_OpSDotAccSat, SPIRV_OC_OpUDotAccSat,
      SPIRV_OC_OpSUDotAccSat,
      SPIRV_OC_OpTypeCooperativeMatrixKHR, SPIRV_OC_OpCooperativeMatrixLoadKHR,
      SPIRV_OC_OpCooperativeMatrixStoreKHR, SPIRV_OC_OpCooperativeMatrixMulAddKHR,
      SPIRV_OC_OpCooperativeMatrixLengthKHR,
      SPIRV_OC_OpSubgroupBlockReadINTEL, SPIRV_OC_OpSubgroupBlockWriteINTEL,
      SPIRV_OC_OpAssumeTrueKHR, SPIRV_OC_OpAtomicFAddEXT, SPIRV_OC_OpGroupIMulKHR,
      SPIRV_OC_OpGroupFMulKHR,

      SPIRV_OC_OpTypeJointMatrixINTEL, SPIRV_OC_OpJointMatrixLoadINTEL,
      SPIRV_OC_OpJointMatrixStoreINTEL, SPIRV_OC_OpJointMatrixMadINTEL,
      SPIRV_OC_OpTypejointMatrixWorkItemLengthINTEL,

      SPIRV_OC_OpConvertFToBF16INTEL, SPIRV_OC_OpConvertBF16ToFINTEL
    ]>;

// End opcode section. Generated from SPIR-V spec; DO NOT MODIFY!

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPIRV_Op<string mnemonic, list<Trait> traits = []> :
    Op<SPIRV_Dialect, mnemonic, !listconcat(traits, [
         // TODO: We don't need all of the following traits for every op; only
         // the suitable ones should be added automatically after ODS supports
         // dialect-specific contents.
         DeclareOpInterfaceMethods<QueryMinVersionInterface>,
         DeclareOpInterfaceMethods<QueryMaxVersionInterface>,
         DeclareOpInterfaceMethods<QueryExtensionInterface>,
         DeclareOpInterfaceMethods<QueryCapabilityInterface>
       ])> {
  // Availability specification for this op itself.
  list<Availability> availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[]>
  ];

  // Controls whether to auto-generate this op's availability specification.
  // If set, generates the following methods:
  //
  // ```c++
  // SmallVector<ArrayRef<Capability>, 1> OpTy::getCapabilities();
  // SmallVector<ArrayRef<Extension>, 1>  OpTy::getExtensions();
  // Optional<Version>                    OpTy::getMinVersion();
  // Optional<Version>                    OpTy::getMaxVersion();
  // ```
  //
  // When not set, manual implementation of these methods is required.
  bit autogenAvailability = 1;

  // For each SPIR-V op, the following static functions need to be defined
  // in SPIRVOps.cpp:
  //
  // * ParseResult <op-c++-class-name>::parse(OpAsmParser &parser,
  //                                          OperationState &result)
  // * void <op-c++-class-name>::print(OpAsmPrinter &p)
  // * LogicalResult <op-c++-class-name>::verify()
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

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
  //    dispatch needs to be handled manually. SPIRV_ExtInstOps are an
  //    example of this.
  bit autogenSerialization = 1;
}

class SPIRV_UnaryOp<string mnemonic, Type resultType, Type operandType,
                    list<Trait> traits = []> :
      SPIRV_Op<mnemonic, traits> {
  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  let assemblyFormat = "$operand `:` type($operand) attr-dict";
  // No additional verification needed in addition to the ODS-generated ones.
  let hasVerifier = 0;
}

class SPIRV_BinaryOp<string mnemonic, Type resultType, Type operandsType,
                     list<Trait> traits = []> :
      SPIRV_Op<mnemonic, traits> {
  let arguments = (ins
    SPIRV_ScalarOrVectorOf<operandsType>:$operand1,
    SPIRV_ScalarOrVectorOf<operandsType>:$operand2
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<resultType>:$result
  );

  // No additional verification needed in addition to the ODS-generated ones.
  let hasVerifier = 0;
}

class SPIRV_ExtInstOp<string mnemonic, string setPrefix, string setName,
                      int opcode, list<Trait> traits = []> :
  SPIRV_Op<setPrefix # "." # mnemonic, traits> {

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

// Base classes for SPIR-V vendor ops. These have opcode in the form of
// Op<Name><VENDOR>, e.g., OpCooperativeMatrixStoreNV.
class SPIRV_VendorOp<string mnemonic, string vendorName,
                     list<Trait> traits = []> :
  SPIRV_Op<vendorName # "." # mnemonic, traits> {
  string spirvOpName = "Op" # mnemonic # vendorName;
}

class SPIRV_ExtVendorOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_VendorOp<mnemonic, "EXT", traits> {
}

class SPIRV_KhrVendorOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_VendorOp<mnemonic, "KHR", traits> {
}

class SPIRV_IntelVendorOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_VendorOp<mnemonic, "INTEL", traits> {
}

class SPIRV_NvVendorOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_VendorOp<mnemonic, "NV", traits> {
}

def SPIRV_FPFMM_None         : I32BitEnumAttrCaseNone<"None">;
def SPIRV_FPFMM_NotNaN       : I32BitEnumAttrCaseBit<"NotNaN", 0>;
def SPIRV_FPFMM_NotInf       : I32BitEnumAttrCaseBit<"NotInf", 1>;
def SPIRV_FPFMM_NSZ          : I32BitEnumAttrCaseBit<"NSZ", 2>;
def SPIRV_FPFMM_AllowRecip   : I32BitEnumAttrCaseBit<"AllowRecip", 3>;
def SPIRV_FPFMM_Fast         : I32BitEnumAttrCaseBit<"Fast", 4>;
def SPIRV_FPFMM_AllowContractFastINTEL : I32BitEnumAttrCaseBit<"AllowContractFastINTEL", 16> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPFastMathModeINTEL]>
  ];
}
def SPIRV_FPFMM_AllowReassocINTEL : I32BitEnumAttrCaseBit<"AllowReassocINTEL", 17> {
  list<Availability> availability = [
    Capability<[SPIRV_C_FPFastMathModeINTEL]>
  ];
}

def SPIRV_FPFastMathModeAttr :
    SPIRV_BitEnumAttr<"FPFastMathMode", "Indicates a floating-point fast math flag", "fastmath_mode", [
      SPIRV_FPFMM_None, SPIRV_FPFMM_NotNaN, SPIRV_FPFMM_NotInf, SPIRV_FPFMM_NSZ,
      SPIRV_FPFMM_AllowRecip, SPIRV_FPFMM_Fast, SPIRV_FPFMM_AllowContractFastINTEL,
      SPIRV_FPFMM_AllowReassocINTEL
    ]>;

#endif // MLIR_DIALECT_SPIRV_IR_BASE
