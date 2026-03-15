// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API
// RUN: %offload-tblgen -gen-entry-points -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-VALIDATION

// Check implicit returns are included in documentation and the validation
// wrappers where applicable

include "APIDefs.td"

def ol_foo_handle_t : Handle {
    let desc = "Example handle type";
}

def FunctionA : Function {
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
    let params = [
        Param<"uint32_t", "ParamValue", "A plain value parameter">,
        Param<"ol_foo_handle_t", "ParamHandle", "A handle parameter">,
        Param<"uint32_t*", "ParamPointer", "A pointer parameter">,
        Param<"uint32_t*", "ParamPointerOpt", "An optional pointer parameter", PARAM_OUT_OPTIONAL>
    ];
    let returns = [];
}

// CHECK-API: /// @returns
// CHECK-API: OL_RESULT_SUCCESS
// CHECK-API: OL_ERRC_INVALID_NULL_HANDLE
// CHECK-API-NEXT: `NULL == ParamHandle`
// CHECK-API: OL_ERRC_INVALID_NULL_POINTER
// CHECK-API-NEXT: `NULL == ParamPointer`
// CHECK-API-NOT: `NULL == ParamPointerOpt`

// CHECK-VALIDATION: FunctionA_val
// CHECK-VALIDATION: if (NULL == ParamHandle)
// CHECK-VALIDATION-NEXT: return createOffloadError(error::ErrorCode::INVALID_NULL_HANDLE, "validation failure: NULL == ParamHandle");
// CHECK-VALIDATION: if (NULL == ParamPointer)
// CHECK-VALIDATION-NEXT: return createOffloadError(error::ErrorCode::INVALID_NULL_POINTER, "validation failure: NULL == ParamPointer");
// CHECK-VALIDATION-NOT: if (NULL == ParamPointerOpt)
