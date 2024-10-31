def : Macro {
  let name = "OL_VERSION_MAJOR";
  let desc = "Major version of the Offload API";
  let value = "0";
}

def : Macro {
  let name = "OL_VERSION_MINOR";
  let desc = "Minor version of the Offload API";
  let value = "0";
}

def : Macro {
  let name = "OL_VERSION_PATCH";
  let desc = "Patch version of the Offload API";
  let value = "1";
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

def : Enum {
  let name = "ol_errc_t";
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
  let name = "ol_error_struct_t";
  let desc = "Details of the error condition returned by an API call";
  let members = [
    StructMember<"ol_errc_t", "Code", "The error code">,
    StructMember<"const char*", "Details", "String containing error details">
  ];
}

def : Typedef {
  let name = "ol_result_t";
  let desc = "Result type returned by all entry points.";
  let value = "const ol_error_struct_t*";
}

def : Macro {
  let name = "OL_SUCCESS";
  let desc = "Success condition";
  let value = "NULL";
}

def : Struct {
  let name = "ol_code_location_t";
  let desc = "Code location information that can optionally be associated with an API call";
  let members = [
    StructMember<"const char*", "FunctionName", "Function name">,
    StructMember<"const char*", "SourceFile", "Source code file">,
    StructMember<"uint32_t", "LineNumber", "Source code line number">,
    StructMember<"uint32_t", "ColumnNumber", "Source code column number">
  ];
}

def : Function {
  let name = "olInit";
  let desc = "Perform initialization of the Offload library and plugins";
  let details = [
    "This must be the first API call made by a user of the Offload library",
    "Each call will increment an internal reference count that is decremented by `olShutDown`"
  ];
  let params = [];
  let returns = [];
}

def : Function {
  let name = "olShutDown";
  let desc = "Release the resources in use by Offload";
  let details = [
    "This decrements an internal reference count. When this reaches 0, all resources will be released",
    "Subsequent API calls made after this are not valid"
  ];
  let params = [];
  let returns = [];
}
