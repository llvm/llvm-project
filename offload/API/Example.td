//===-- Example.td - Example definitions for Offload -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file serves as an example for the Offload tablegen framework.
// It is NOT an actual representation of the API. It is based off a random
// selection of features from Unified Runtime.
//
//===----------------------------------------------------------------------===//


def : Macro {
  let name = "OL_MAKE_VERSION( _major, _minor )";
  let desc = "Generates generic API versions";
  let value = "(( _major << 16 )|( _minor & 0x0000ffff))";
}

def : Macro {
  let name = "OL_MAJOR_VERSION( _ver )";
  let desc = "Extracts API major version";
  let value = "( _ver >> 16 )";
}

def : Macro {
  let name = "OL_MINOR_VERSION( _ver )";
  let desc = "Extracts API minor version";
  let value = "( _ver & 0x0000ffff )";
}

def : Macro {
  let name = "OL_APICALL";
  let desc = "Calling convention for all API functions";
  let condition = "defined(_WIN32)";
  let value = "__cdecl";
  let alt_value = "";
}

def : Macro {
  let name = "OL_APIEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
  let alt_value = "";
}

def : Macro {
  let name = "OL_DLLEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
}

def : Macro {
  let name = "OL_DLLEXPORT";
  let desc = "GCC-specific dllexport storage-class attribute";
  let condition = "__GNUC__ >= 4";
  let value = "__attribute__ ((visibility (\"default\")))";
  let alt_value = "";
}

def : Typedef {
  let name = "ol_bool_t";
  let value = "uint8_t";
  let desc = "compiler-independent type";
}

def : Handle {
  let name = "ol_loader_config_handle_t";
  let desc = "Handle of a loader config object";
}

def : Handle {
  let name = "ol_adapter_handle_t";
  let desc = "Handle of an adapter instance";
}

def : Handle {
  let name = "ol_platform_handle_t";
  let desc = "Handle of a platform instance";
}

def : Handle {
  let name = "ol_device_handle_t";
  let desc = "Handle of platform's device object";
}

def : Handle {
  let name = "ol_context_handle_t";
  let desc = "Handle of context object";
}

def : Handle {
  let name = "ol_event_handle_t";
  let desc = "Handle of event object";
}

def : Handle {
  let name = "ol_program_handle_t";
  let desc = "Handle of Program object";
}

def : Handle {
  let name = "ol_kernel_handle_t";
  let desc = "Handle of program's Kernel object";
}

def : Handle {
  let name = "ol_queue_handle_t";
  let desc = "Handle of a queue object";
}

def : Handle {
  let name = "ol_native_handle_t";
  let desc = "Handle of a native object";
}

def : Handle {
  let name = "ol_sampler_handle_t";
  let desc = "Handle of a Sampler object";
}

def : Handle {
  let name = "ol_mem_handle_t";
  let desc = "Handle of memory object which can either be buffer or image";
}

def : Handle {
  let name = "ol_physical_mem_handle_t";
  let desc = "Handle of physical memory object";
}

def : Macro {
  let name = "OL_BIT( _i )";
  let desc = "Generic macro for enumerator bit masks";
  let value = "( 1 << _i )";
}

def : Enum {
  let name = "ol_result_t";
  let desc = "Defines Return/Error codes";
  let etors =[
    Etor<"SUCCESS", "Success">,
    Etor<"ERROR_INVALID_OPERATION", "Invalid operation">,
    Etor<"ERROR_INVALID_QUEUE_PROPERTIES", "Invalid queue properties">,
    Etor<"ERROR_INVALID_QUEUE", "Invalid queue">,
    Etor<"ERROR_INVALID_VALUE", "Invalid Value">,
    Etor<"ERROR_INVALID_CONTEXT", "Invalid context">,
    Etor<"ERROR_INVALID_PLATFORM", "Invalid platform">,
    Etor<"ERROR_INVALID_BINARY", "Invalid binary">,
    Etor<"ERROR_INVALID_PROGRAM", "Invalid program">,
    Etor<"ERROR_INVALID_SAMPLER", "Invalid sampler">,
    Etor<"ERROR_INVALID_BUFFER_SIZE", "Invalid buffer size">,
    Etor<"ERROR_INVALID_MEM_OBJECT", "Invalid memory object">,
    Etor<"ERROR_INVALID_EVENT", "Invalid event">,
    Etor<"ERROR_INVALID_EVENT_WAIT_LIST", "Returned when the event wait list or the events in the wait list are invalid.">,
    Etor<"ERROR_MISALIGNED_SUB_BUFFER_OFFSET", "Misaligned sub buffer offset">,
    Etor<"ERROR_INVALID_WORK_GROUP_SIZE", "Invalid work group size">,
    Etor<"ERROR_COMPILER_NOT_AVAILABLE", "Compiler not available">,
    Etor<"ERROR_PROFILING_INFO_NOT_AVAILABLE", "Profiling info not available">,
    Etor<"ERROR_DEVICE_NOT_FOUND", "Device not found">,
    Etor<"ERROR_INVALID_DEVICE", "Invalid device">,
    Etor<"ERROR_DEVICE_LOST", "Device hung, reset, was removed, or adapter update occurred">,
    Etor<"ERROR_DEVICE_REQUIRES_RESET", "Device requires a reset">,
    Etor<"ERROR_DEVICE_IN_LOW_POWER_STATE", "Device currently in low power state">,
    Etor<"ERROR_DEVICE_PARTITION_FAILED", "Device partitioning failed">,
    Etor<"ERROR_INVALID_DEVICE_PARTITION_COUNT", "Invalid counts provided with OL_DEVICE_PARTITION_BY_COUNTS">,
    Etor<"ERROR_INVALID_WORK_ITEM_SIZE", "Invalid work item size">,
    Etor<"ERROR_INVALID_WORK_DIMENSION", "Invalid work dimension">,
    Etor<"ERROR_INVALID_KERNEL_ARGS", "Invalid kernel args">,
    Etor<"ERROR_INVALID_KERNEL", "Invalid kernel">,
    Etor<"ERROR_INVALID_KERNEL_NAME", "[Validation] kernel name is not found in the program">,
    Etor<"ERROR_INVALID_KERNEL_ARGUMENT_INDEX", "[Validation] kernel argument index is not valid for kernel">,
    Etor<"ERROR_INVALID_KERNEL_ARGUMENT_SIZE", "[Validation] kernel argument size does not match kernel">,
    Etor<"ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE", "[Validation] value of kernel attribute is not valid for the kernel or device">,
    Etor<"ERROR_INVALID_IMAGE_SIZE", "Invalid image size">,
    Etor<"ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR", "Invalid image format descriptor">,
    Etor<"ERROR_IMAGE_FORMAT_NOT_SUPPORTED", "Image format not supported">,
    Etor<"ERROR_MEM_OBJECT_ALLOCATION_FAILURE", "Memory object allocation failure">,
    Etor<"ERROR_INVALID_PROGRAM_EXECUTABLE", "Program object parameter is invalid.">,
    Etor<"ERROR_UNINITIALIZED", "[Validation] adapter is not initialized or specific entry-point is not implemented">,
    Etor<"ERROR_OUT_OF_HOST_MEMORY", "Insufficient host memory to satisfy call">,
    Etor<"ERROR_OUT_OF_DEVICE_MEMORY", "Insufficient device memory to satisfy call">,
    Etor<"ERROR_OUT_OF_RESOURCES", "Out of resources">,
    Etor<"ERROR_PROGRAM_BUILD_FAILURE", "Error occurred when building program, see build log for details">,
    Etor<"ERROR_PROGRAM_LINK_FAILURE", "Error occurred when linking programs, see build log for details">,
    Etor<"ERROR_UNSUPPORTED_VERSION", "[Validation] generic error code for unsupported versions">,
    Etor<"ERROR_UNSUPPORTED_FEATURE", "[Validation] generic error code for unsupported features">,
    Etor<"ERROR_INVALID_ARGUMENT", "[Validation] generic error code for invalid arguments">,
    Etor<"ERROR_INVALID_NULL_HANDLE", "[Validation] handle argument is not valid">,
    Etor<"ERROR_HANDLE_OBJECT_IN_USE", "[Validation] object pointed to by handle still in-use by device">,
    Etor<"ERROR_INVALID_NULL_POINTER", "[Validation] pointer argument may not be nullptr">,
    Etor<"ERROR_INVALID_SIZE", "[Validation] invalid size or dimensions (e.g., must not be zero, or is out of bounds)">,
    Etor<"ERROR_UNSUPPORTED_SIZE", "[Validation] size argument is not supported by the device (e.g., too large)">,
    Etor<"ERROR_UNSUPPORTED_ALIGNMENT", "[Validation] alignment argument is not supported by the device (e.g., too small)">,
    Etor<"ERROR_INVALID_SYNCHRONIZATION_OBJECT", "[Validation] synchronization object in invalid state">,
    Etor<"ERROR_INVALID_ENUMERATION", "[Validation] enumerator argument is not valid">,
    Etor<"ERROR_UNSUPPORTED_ENUMERATION", "[Validation] enumerator argument is not supported by the device">,
    Etor<"ERROR_UNSUPPORTED_IMAGE_FORMAT", "[Validation] image format is not supported by the device">,
    Etor<"ERROR_INVALID_NATIVE_BINARY", "[Validation] native binary is not supported by the device">,
    Etor<"ERROR_INVALID_GLOBAL_NAME", "[Validation] global variable is not found in the program">,
    Etor<"ERROR_INVALID_FUNCTION_NAME", "[Validation] function name is not found in the program">,
    Etor<"ERROR_INVALID_GROUP_SIZE_DIMENSION", "[Validation] group size dimension is not valid for the kernel or device">,
    Etor<"ERROR_INVALID_GLOBAL_WIDTH_DIMENSION", "[Validation] global width dimension is not valid for the kernel or device">,
    Etor<"ERROR_PROGRAM_UNLINKED", "[Validation] compiled program or program with imports needs to be linked before kernels can be created from it.">,
    Etor<"ERROR_OVERLAPPING_REGIONS", "[Validation] copy operations do not support overlapping regions of memory">,
    Etor<"ERROR_INVALID_HOST_PTR", "Invalid host pointer">,
    Etor<"ERROR_INVALID_USM_SIZE", "Invalid USM size">,
    Etor<"ERROR_OBJECT_ALLOCATION_FAILURE", "Objection allocation failure">,
    Etor<"ERROR_ADAPTER_SPECIFIC", "An adapter specific warning/error has been reported and can be retrieved via the urPlatformGetLastError entry point.">,
    Etor<"ERROR_LAYER_NOT_PRESENT", "A requested layer was not found by the loader.">,
    Etor<"ERROR_IN_EVENT_LIST_EXEC_STATUS", "An event in the provided wait list has OL_EVENT_STATUS_ERROR.">,
    Etor<"ERROR_UNKNOWN", "Unknown or internal error">
  ];
}

def : Struct {
  let name = "ol_base_properties_t";
  let desc = "Base for all properties types";
  let members = [
    StructMember<"ol_structure_type_t", "stype", "[in] type of this structure">,
    StructMember<"void*", "pNext", "[in,out][optional] pointer to extension-specific structure">
  ];
}

def : Struct {
  let name = "ol_base_desc_t";
  let desc = "Base for all descriptor types";
  let members = [
    StructMember<"ol_structure_type_t", "stype", "[in] type of this structure">,
    StructMember<"const void*", "pNext", "[in][optional] pointer to extension-specific structure">
  ];
}

def : Struct {
  let name = "ol_rect_offset_t";
  let desc = "3D offset argument passed to buffer rect operations";
  let members = [
    StructMember<"uint64_t", "x", "[in] x offset (bytes)">,
    StructMember<"uint64_t", "y", "[in] y offset (scalar)">,
    StructMember<"uint64_t", "z", "[in] z offset (scalar)">
  ];
}

def : Struct {
  let name = "ol_rect_region_t";
  let desc = "3D region argument passed to buffer rect operations";
  let members = [
    StructMember<"uint64_t", "width", "[in] width (bytes)">,
    StructMember<"uint64_t", "height", "[in] height (scalar)">,
    StructMember<"uint64_t", "depth", "[in] scalar (scalar)">
  ];
}

def : Enum {
  let name = "ol_queue_info_t";
  let desc = "Query queue info";
    let is_typed = 1;
  let etors =[
    Etor<"CONTEXT", "[ol_context_handle_t] context associated with this queue.">,
    Etor<"DEVICE", "[ol_device_handle_t] device associated with this queue.">,
    Etor<"DEVICE_DEFAULT", "[ol_queue_handle_t] the current default queue of the underlying device.">,
    Etor<"FLAGS", "[ol_queue_flags_t] the properties associated with ol_queue_properties_t::flags.">,
    Etor<"REFERENCE_COUNT", [{[uint32_t] Reference count of the queue object.
The reference count returned should be considered immediately stale. 
It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.}]>,
    Etor<"SIZE", "[uint32_t] The size of the queue">,
    Etor<"EMPTY", "[ol_bool_t] return true if the queue was empty at the time of the query">
  ];
}

def : Enum {
  let name = "ol_queue_flags_t";
  let desc = "Queue property flags";
  let etors =[
    Etor<"OUT_OF_ORDER_EXEC_MODE_ENABLE", "Enable/disable out of order execution">,
    Etor<"PROFILING_ENABLE", "Enable/disable profiling">,
    Etor<"ON_DEVICE", "Is a device queue">,
    Etor<"ON_DEVICE_DEFAULT", "Is the default queue for a device">,
    Etor<"DISCARD_EVENTS", "Events will be discarded">,
    Etor<"PRIORITY_LOW", "Low priority queue">,
    Etor<"PRIORITY_HIGH", "High priority queue">,
    Etor<"SUBMISSION_BATCHED", "Hint: enqueue and submit in a batch later. No change in queue semantics. Implementation chooses submission mode.">,
    Etor<"SUBMISSION_IMMEDIATE", "Hint: enqueue and submit immediately. No change in queue semantics. Implementation chooses submission mode.">,
    Etor<"USE_DEFAULT_STREAM", "Use the default stream. Only meaningful for CUDA. Other platforms may ignore this flag.">,
    Etor<"SYNC_WITH_DEFAULT_STREAM", "Synchronize with the default stream. Only meaningful for CUDA. Other platforms may ignore this flag.">
  ];
}

def : Function<"olQueue"> {
  let name = "GetInfo";
  let desc = "Query information about a command queue";
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue object", PARAM_IN>,
    Param<"ol_queue_info_t", "propName", "name of the queue property to query", PARAM_IN>,
    Param<"size_t", "propSize", "size in bytes of the queue property value provided", PARAM_IN>,
    Param<"void*", "pPropValue", "[typename(propName, propSize)] value of the queue property", !or(PARAM_OUT, PARAM_OPTIONAL)>,
    Param<"size_t*", "pPropSizeRet", "size in bytes returned in queue property value", !or(PARAM_OUT, PARAM_OPTIONAL)>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_UNSUPPORTED_ENUMERATION", [
      "If `propName` is not supported by the adapter."
    ]>,
    Return<"OL_RESULT_ERROR_INVALID_SIZE", [
      "`propSize == 0 && pPropValue != NULL`",
      "If `propSize` is less than the real number of bytes needed to return the info."
    ]>,
    Return<"OL_RESULT_ERROR_INVALID_NULL_POINTER", [
      "`propSize != 0 && pPropValue == NULL`",
      "`pPropValue == NULL && pPropSizeRet == NULL`"
    ]>,
    Return<"OL_RESULT_ERROR_INVALID_QUEUE">,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">,
    Return<"OL_RESULT_ERROR_OUT_OF_RESOURCES">
  ];
}

def : Struct {
  let name = "ol_queue_properties_t";
  let desc = "Queue creation properties";
  let base_class = "ol_base_properties_t";
  let members = [
    StructMember<"ol_queue_flags_t", "flags", "[in] Bitfield of queue creation flags">
  ];
}

def : Struct {
  let name = "ol_queue_index_properties_t";
  let desc = "Queue index creation properties";
  let base_class = "ol_base_properties_t";
  let members = [
    StructMember<"uint32_t", "computeIndex", "[in] Specifies the compute index as described in the sycl_ext_intel_queue_index extension.">
  ];
}

def : Function<"olQueue"> {
  let name = "Create";
  let desc = "Create a command queue for a device in a context";
  let details = [
    "See also ol_queue_index_properties_t."
  ];
  let params = [
    Param<"ol_context_handle_t", "hContext", "handle of the context object", PARAM_IN>,
    Param<"ol_device_handle_t", "hDevice", "handle of the device object", PARAM_IN>,
    Param<"const ol_queue_properties_t*", "pProperties", "pointer to queue creation properties.", !or(PARAM_IN, PARAM_OPTIONAL)>,
    Param<"ol_queue_handle_t*", "phQueue", "pointer to handle of queue object created", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_INVALID_CONTEXT">,
    Return<"OL_RESULT_ERROR_INVALID_DEVICE">,
    Return<"OL_RESULT_ERROR_INVALID_QUEUE_PROPERTIES", [
      "`pProperties != NULL && pProperties->flags & OL_QUEUE_FLAG_PRIORITY_HIGH && pProperties->flags & OL_QUEUE_FLAG_PRIORITY_LOW`",
      "`pProperties != NULL && pProperties->flags & OL_QUEUE_FLAG_SUBMISSION_BATCHED && pProperties->flags & OL_QUEUE_FLAG_SUBMISSION_IMMEDIATE`"
    ]>,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">,
    Return<"OL_RESULT_ERROR_OUT_OF_RESOURCES">
  ];
}

def : Function<"olQueue"> {
  let name = "Retain";
  let desc = "Get a reference to the command queue handle. Increment the command queue's reference count";
  let details = [
    "Useful in library function to retain access to the command queue after the caller released the queue."
  ];
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue object to get access", PARAM_IN>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_INVALID_QUEUE">,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">,
    Return<"OL_RESULT_ERROR_OUT_OF_RESOURCES">
  ];
}

def : Function<"olQueue"> {
  let name = "Release";
  let desc = "Decrement the command queue's reference count and delete the command queue if the reference count becomes zero.";
  let details = [
    "After the command queue reference count becomes zero and all queued commands in the queue have finished, the queue is deleted.",
    "It also performs an implicit flush to issue all previously queued commands in the queue."
  ];
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue object to release", PARAM_IN>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_INVALID_QUEUE">,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">,
    Return<"OL_RESULT_ERROR_OUT_OF_RESOURCES">
  ];
}

def : Struct {
  let name = "ol_queue_native_desc_t";
  let desc = "Descriptor for olQueueGetNativeHandle and olQueueCreateWithNativeHandle.";
  let base_class = "ol_base_desc_t";
  let members = [
    StructMember<"void*", "pNativeData", "[in][optional] Adapter-specific metadata needed to create the handle.">
  ];
}

def : Function<"olQueue"> {
  let name = "GetNativeHandle";
  let desc = "Return queue native queue handle.";
  let details = [
    "Retrieved native handle can be used for direct interaction with the native platform driver.",
    "Use interoperability queue extensions to convert native handle to native type.",
    "The application may call this function from simultaneous threads for the same context.",
    "The implementation of this function should be thread-safe."
  ];
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue.", PARAM_IN>,
    Param<"ol_queue_native_desc_t*", "pDesc", "pointer to native descriptor", !or(PARAM_IN, PARAM_OPTIONAL)>,
    Param<"ol_native_handle_t*", "phNativeQueue", "a pointer to the native handle of the queue.", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_UNSUPPORTED_FEATURE", [
      "If the adapter has no underlying equivalent handle."
    ]>
  ];
}

def : Struct {
  let name = "ol_queue_native_properties_t";
  let desc = "Properties for for olQueueCreateWithNativeHandle.";
  let base_class = "ol_base_properties_t";
  let members = [
    StructMember<"bool", "isNativeHandleOwned", [{[in] Indicates UR owns the native handle or if it came from an interoperability
operation in the application that asked to not transfer the ownership to
the unified-runtime.}]>
  ];
}

def : Function<"olQueue"> {
  let name = "CreateWithNativeHandle";
  let desc = "Create runtime queue object from native queue handle.";
  let details = [
    "Creates runtime queue handle from native driver queue handle.",
    "The application may call this function from simultaneous threads for the same context.",
    "The implementation of this function should be thread-safe."
  ];
  let params = [
    Param<"ol_native_handle_t", "hNativeQueue", "[nocheck] the native handle of the queue.", PARAM_IN>,
    Param<"ol_context_handle_t", "hContext", "handle of the context object", PARAM_IN>,
    Param<"ol_device_handle_t", "hDevice", "handle of the device object", PARAM_IN>,
    Param<"const ol_queue_native_properties_t*", "pProperties", "pointer to native queue properties struct", !or(PARAM_IN, PARAM_OPTIONAL)>,
    Param<"ol_queue_handle_t*", "phQueue", "pointer to the handle of the queue object created.", PARAM_OUT>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_UNSUPPORTED_FEATURE", [
      "If the adapter has no underlying equivalent handle."
    ]>
  ];
}

def : Function<"olQueue"> {
  let name = "Finish";
  let desc = "Blocks until all previously issued commands to the command queue are finished.";
  let details = [
    "Blocks until all previously issued commands to the command queue are issued and completed.",
    "olQueueFinish does not return until all enqueued commands have been processed and finished.",
    "olQueueFinish acts as a synchronization point."
  ];
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue to be finished.", PARAM_IN>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_INVALID_QUEUE">,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">
  ];
}

def : Function<"olQueue"> {
  let name = "Flush";
  let desc = "Issues all previously enqueued commands in a command queue to the device.";
  let details = [
    "Guarantees that all enqueued commands will be issued to the appropriate device.",
    "There is no guarantee that they will be completed after olQueueFlush returns."
  ];
  let params = [
    Param<"ol_queue_handle_t", "hQueue", "handle of the queue to be flushed.", PARAM_IN>
  ];
  let returns = [
    Return<"OL_RESULT_ERROR_INVALID_QUEUE">,
    Return<"OL_RESULT_ERROR_OUT_OF_HOST_MEMORY">
  ];
}
