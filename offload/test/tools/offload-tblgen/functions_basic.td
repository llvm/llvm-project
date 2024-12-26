// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API
// RUN: %offload-tblgen -gen-exports -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-EXPORTS
// RUN: %offload-tblgen -gen-func-names -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-FUNC-MACRO

// Check basic support for API functions

include "APIDefs.td"

def : Function {
    let name = "FunctionA";
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
    let params = [
        Param<"uint32_t", "ParamA", "Parameter A description">,
        Param<"uint32_t*", "ParamB", "Parameter B description">,
    ];
    let returns = [
        Return<"OL_ERRC_INVALID_VALUE", ["When a value is invalid"]>
    ];
}

// CHECK-API: /// @brief Function A description
// CHECK-API: /// @details
// CHECK-API-NEXT: Function A detailed information
// CHECK-API: /// @returns
// CHECK-API: OL_ERRC_INVALID_VALUE
// CHECK-API-NEXT: When a value is invalid

// CHECK-API: ol_result_t
// CHECK-API-SAME: FunctionA

// CHECK-API: // Parameter A description
// CHECK-API-NEXT: uint32_t ParamA
// CHECK-API: // Parameter B description
// CHECK-API-NEXT: uint32_t* ParamB

// CHECK-EXPORTS: FunctionA

// CHECK-FUNC-MACRO: OFFLOAD_FUNC(FunctionA)
