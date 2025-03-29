// RUN: %offload-tblgen -gen-print-header -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-PRINT
// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API

// Check that print helpers are created for functions

include "APIDefs.td"

def : Handle {
    let name = "ol_foo_handle_t";
    let desc = "Example handle type";
}

def : Function {
    let name = "FunctionA";
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
    let params = [
        Param<"uint32_t", "ParamValue", "A plain value parameter">,
        Param<"ol_foo_handle_t", "ParamHandle", "A handle parameter">,
        Param<"uint32_t*", "ParamPointer", "A pointer parameter">,
    ];
    let returns = [];
}

// CHECK-API: typedef struct function_a_params_t {
// CHECK-API-NEXT: uint32_t* pParamValue;
// CHECK-API-NEXT: ol_foo_handle_t* pParamHandle;
// CHECK-API-NEXT: uint32_t** pParamPointer;

// CHECK-PRINT: inline std::ostream &operator<<(std::ostream &os, const struct function_a_params_t *params)
// CHECK-PRINT: os << ".ParamValue = ";
// CHECK-PRINT: os << *params->pParamValue;
// CHECK-PRINT: os << ", ";
// CHECK-PRINT: os << ".ParamHandle = ";
// CHECK-PRINT: printPtr(os, *params->pParamHandle);
// CHECK-PRINT: os << ", ";
// CHECK-PRINT: os << ".ParamPointer = ";
// CHECK-PRINT: printPtr(os, *params->pParamPointer);
