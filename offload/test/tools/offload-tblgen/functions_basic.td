// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API
// RUN: %offload-tblgen -gen-exports -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-EXPORTS
// RUN: %offload-tblgen -gen-func-names -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-FUNC-MACRO

// Check basic support for API functions

include "APIDefs.td"

def OL_NODISCARD : Macro {
  let desc = "Marks a function whose result should not be discarded";
  let condition = "(defined(__cplusplus) && __cplusplus >= 201703L) || "
                  "(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)";
  let value = "[[nodiscard]]";
  let alt_value = "";
}

def FunctionA : Function {
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

// A function whose result should not be discarded.
def FunctionB : Function {
    let desc = "Function B description";
    let nodiscard = 1;
    let params = [
        Param<"uint32_t", "ParamA", "Parameter A description">,
    ];
    let returns = [];
}

// The OL_NODISCARD macro is emitted as a conditional #define.
// CHECK-API: #ifndef OL_NODISCARD
// CHECK-API: #define OL_NODISCARD {{\[\[}}nodiscard{{\]\]}}

// CHECK-API: /// @brief Function A description
// CHECK-API: /// @details
// CHECK-API-NEXT: Function A detailed information
// CHECK-API: /// @returns
// CHECK-API: OL_ERRC_INVALID_VALUE
// CHECK-API-NEXT: When a value is invalid

// FunctionA is not marked nodiscard, so its declaration and its WithCodeLoc
// variant must not carry the attribute.
// CHECK-API-NOT: OL_NODISCARD
// CHECK-API: ol_result_t
// CHECK-API-SAME: FunctionA

// CHECK-API: // Parameter A description
// CHECK-API-NEXT: uint32_t ParamA
// CHECK-API: // Parameter B description
// CHECK-API-NEXT: uint32_t* ParamB

// FunctionB is marked nodiscard: both the primary declaration and the
// WithCodeLoc variant are prefixed with OL_NODISCARD.
// CHECK-API: OL_NODISCARD OL_APIEXPORT ol_result_t OL_APICALL FunctionB(
// CHECK-API: OL_NODISCARD OL_APIEXPORT ol_result_t OL_APICALL FunctionBWithCodeLoc(

// CHECK-EXPORTS: FunctionA

// CHECK-FUNC-MACRO: OFFLOAD_FUNC(FunctionA)
