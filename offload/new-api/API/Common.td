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
  let name = "offload_result_t";
  let desc = "Defines Return/Error codes";
  let etors =[
    Etor<"SUCCESS", "Success">,
    Etor<"ERROR_INVALID_VALUE", "Invalid Value">,
    Etor<"ERROR_INVALID_PLATFORM", "Invalid platform">,
    Etor<"ERROR_DEVICE_NOT_FOUND", "Device not found">,
    Etor<"ERROR_INVALID_DEVICE", "Invalid device">,
    Etor<"ERROR_DEVICE_LOST", "Device hung, reset, was removed, or driver update occurred">,
    Etor<"ERROR_UNINITIALIZED", "plugin is not initialized or specific entry-point is not implemented">,
    Etor<"ERROR_OUT_OF_RESOURCES", "Out of resources">,
    Etor<"ERROR_UNSUPPORTED_VERSION", "generic error code for unsupported versions">,
    Etor<"ERROR_UNSUPPORTED_FEATURE", "generic error code for unsupported features">,
    Etor<"ERROR_INVALID_ARGUMENT", "generic error code for invalid arguments">,
    Etor<"ERROR_INVALID_NULL_HANDLE", "handle argument is not valid">,
    Etor<"ERROR_INVALID_NULL_POINTER", "pointer argument may not be nullptr">,
    Etor<"ERROR_INVALID_SIZE", "invalid size or dimensions (e.g., must not be zero, or is out of bounds)">,
    Etor<"ERROR_INVALID_ENUMERATION", "enumerator argument is not valid">,
    Etor<"ERROR_UNSUPPORTED_ENUMERATION", "enumerator argument is not supported by the device">,
    Etor<"ERROR_UNKNOWN", "Unknown or internal error">
  ];
}

def : Function {
  let name = "offloadGetErrorDetails";
  let desc = "Get a detailed error message for the last error that occurred on this thread, if it exists";
  let details = [
    "When an Offload API call returns a return value other than OFFLOAD_RESULT_SUCCESS, the implementation *may* set an additional error message.",
    "If a further Offload call (excluding this function) is made on the same thread without checking "
    "its detailed error message with this function, that message should be considered lost.",
    "The returned char* is only valid until the next Offload function call on the same thread (excluding further calls to this function.)"
  ];
  let params = [
    Param<"size_t*", "SizeRet", "Pointer to return the size of the available error message. A size of 0 indicates no message.", PARAM_OUT_OPTIONAL>,
    Param<"const char**", "DetailStringRet", "Pointer to return the error message string.", PARAM_OUT_OPTIONAL>
  ];
  let returns = []; // Only SUCCESS is expected
}
