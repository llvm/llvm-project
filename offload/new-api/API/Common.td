def : Macro {
  let name = "OFFLOAD_APICALL";
  let desc = "Calling convention for all API functions";
  let condition = "defined(_WIN32)";
  let value = "__cdecl";
  let alt_value = "";
}

def : Macro {
  let name = "OFFLOAD_APIEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
  let alt_value = "";
}

def : Macro {
  let name = "OFFLOAD_DLLEXPORT";
  let desc = "Microsoft-specific dllexport storage-class attribute";
  let condition = "defined(_WIN32)";
  let value = "__declspec(dllexport)";
}

def : Macro {
  let name = "OFFLOAD_DLLEXPORT";
  let desc = "GCC-specific dllexport storage-class attribute";
  let condition = "__GNUC__ >= 4";
  let value = "__attribute__ ((visibility (\"default\")))";
  let alt_value = "";
}

def : Handle {
  let name = "offload_platform_handle_t";
  let desc = "Handle of a platform instance";
}

def : Handle {
  let name = "offload_device_handle_t";
  let desc = "Handle of platform's device object";
}

def : Handle {
  let name = "offload_context_handle_t";
  let desc = "Handle of context object";
}

def : Enum {
  let name = "offload_errc_t";
  let desc = "Defines Return/Error codes";
  let etors =[
    Etor<"SUCCESS", "Success">,
    Etor<"INVALID_VALUE", "Invalid Value">,
    Etor<"INVALID_PLATFORM", "Invalid platform">,
    Etor<"DEVICE_NOT_FOUND", "Device not found">,
    Etor<"INVALID_DEVICE", "Invalid device">,
    Etor<"DEVICE_LOST", "Device hung, reset, was removed, or driver update occurred">,
    Etor<"UNINITIALIZED", "plugin is not initialized or specific entry-point is not implemented">,
    Etor<"OUT_OF_RESOURCES", "Out of resources">,
    Etor<"UNSUPPORTED_VERSION", "generic error code for unsupported versions">,
    Etor<"UNSUPPORTED_FEATURE", "generic error code for unsupported features">,
    Etor<"INVALID_ARGUMENT", "generic error code for invalid arguments">,
    Etor<"INVALID_NULL_HANDLE", "handle argument is not valid">,
    Etor<"INVALID_NULL_POINTER", "pointer argument may not be nullptr">,
    Etor<"INVALID_SIZE", "invalid size or dimensions (e.g., must not be zero, or is out of bounds)">,
    Etor<"INVALID_ENUMERATION", "enumerator argument is not valid">,
    Etor<"UNSUPPORTED_ENUMERATION", "enumerator argument is not supported by the device">,
    Etor<"UNKNOWN", "Unknown or internal error">
  ];
}

def : Struct {
  let name = "offload_error_struct_t";
  let desc = "Details of the error condition returned by an API call";
  let members = [
    StructMember<"offload_errc_t", "code", "The error code">,
    StructMember<"const char*", "details", "String containing error details">
  ];
}

def : Typedef {
  let name = "offload_result_t";
  let desc = "Result type returned by all entry points.";
  let value = "const offload_error_struct_t*";
}

def : Macro {
  let name = "OFFLOAD_SUCCESS";
  let condition = "!defined(OFFLOAD_SUCCESS)";
  let desc = "Success condition";
  let value = "NULL";
}
