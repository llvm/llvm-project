// RUN: %offload-tblgen -gen-print-header -I %S/../../../liboffload/API %s | %fcheck-generic

// Check that ranged function parameters are implemented correctly. These
// are pointers to an array of an arbitrary size. Their size is described as a
// range between two values. This is typically between 0 and a parameter such
// as NumItems. The range information helps the printing code print the entire
// range of the output rather than just the pointer or the first element.

include "APIDefs.td"

def : Handle {
    let name = "some_handle_t";
    let desc = "An example handle type";
}

def : Function {
    let name = "FunctionA";
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
  let params = [
    Param<"size_t", "OutCount", "the number of things to write out", PARAM_IN>,
    RangedParam<"some_handle_t*", "OutPtr", "pointer to the output things.", PARAM_OUT,
      Range<"0", "OutCount">>
  ];
  let returns = [];
}

// CHECK: inline std::ostream &operator<<(std::ostream &os, const struct function_a_params_t *params) {
// CHECK:   os << ".OutPtr = ";
// CHECK:   for (size_t i = 0; i < *params->pOutCount; i++) {
// CHECK:     if (i > 0) {
// CHECK:       os << ", ";
// CHECK:     }
// CHECK:     printPtr(os, (*params->pOutPtr)[i]);
// CHECK:   }
// CHECK:   os << "}";
