// RUN: %offload-tblgen -gen-api -I %S/../../../new-api/API %s | %fcheck-generic --check-prefix=CHECK-API
// RUN: %offload-tblgen -gen-exports -I %S/../../../new-api/API %s | %fcheck-generic --check-prefix=CHECK-EXPORTS
// RUN: %offload-tblgen -gen-func-names -I %S/../../../new-api/API %s | %fcheck-generic --check-prefix=CHECK-FUNC-MACRO

// Check that the function variant with code location information is generated
// and is otherwise the same as the regular function

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
        Return<"OFFLOAD_ERRC_INVALID_VALUE", ["When a value is invalid"]>
    ];
}

// CHECK-API-DAG: offload_result_t{{.*}} FunctionA
// CHECK-API-DAG: offload_result_t{{.*}} FunctionAWithCodeLoc
// CHECK-EXPORTS: FunctionAWithCodeLoc
// CHECK-FUNC-MACRO: OFFLOAD_FUNC(FunctionAWithCodeLoc)
