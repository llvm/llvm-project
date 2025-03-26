// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API
// RUN: %offload-tblgen -gen-print-header -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-PRINT

// Check that type-tagged enumerators are implemented correctly. They enable
// functions to return data of an arbitrary type and size via a void*, using
// the value of an enum parameter to indicate which type is being returned.
// This allows, for example, for a single olGetDeviceInfo function, rather
// than requiring a separate entry point for every possible query.

include "APIDefs.td"

def : Handle {
    let name = "some_handle_t";
    let desc = "An example handle type";
}

def : Enum {
  let name = "my_type_tagged_enum_t";
  let desc = "Example type tagged enum";
  let is_typed = 1;
  let etors = [
    TaggedEtor<"VALUE_ONE", "uint32_t", "Value one.">,
    TaggedEtor<"VALUE_TWO", "char[]", "Value two.">,
    TaggedEtor<"VALUE_THREE", "some_handle_t", "Value three.">
  ];
}

// Check the tagged types appear in the comments
// CHECK-API: typedef enum my_type_tagged_enum_t {
// CHECK-API-NEXT: [uint32_t] Value one.
// CHECK-API-NEXT: MY_TYPE_TAGGED_ENUM_VALUE_ONE = 0,
// CHECK-API-NEXT: [char[]] Value two.
// CHECK-API-NEXT: MY_TYPE_TAGGED_ENUM_VALUE_TWO = 1,
// CHECK-API-NEXT: [some_handle_t] Value three.
// CHECK-API-NEXT: MY_TYPE_TAGGED_ENUM_VALUE_THREE = 2,

def : Function {
    let name = "FunctionA";
    let desc = "Function A description";
    let details = [ "Function A detailed information" ];
  let params = [
    Param<"my_type_tagged_enum_t", "PropName", "type of the info to retrieve", PARAM_IN>,
    Param<"size_t", "PropSize", "the number of bytes pointed to by PropValue.", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info. "
      "If PropSize is not equal to or greater to the real number of bytes needed to return the info "
      "then the OL_ERRC_INVALID_SIZE error is returned and PropValue is not used.", PARAM_OUT,
      TypeInfo<"PropName" , "PropSize">>
  ];
  let returns = [];
}

// Check that a tagged enum print function definition is generated
// CHECK-PRINT: void printTagged(std::ostream &os, const void *ptr, my_type_tagged_enum_t value, size_t size) {
// CHECK-PRINT: case MY_TYPE_TAGGED_ENUM_VALUE_ONE: {
// CHECK-PRINT:   const uint32_t * const tptr = (const uint32_t * const)ptr;
// CHECK-PRINT:   os << (const void *)tptr << " (";
// CHECK-PRINT:   os << *tptr;
// CHECK-PRINT:   os << ")";
// CHECK-PRINT:   break;
// CHECK-PRINT: }
// CHECK-PRINT: case MY_TYPE_TAGGED_ENUM_VALUE_TWO: {
// CHECK-PRINT:   printPtr(os, (const char*) ptr);
// CHECK-PRINT:   break;
// CHECK-PRINT: }
// CHECK-PRINT: case MY_TYPE_TAGGED_ENUM_VALUE_THREE: {
// CHECK-PRINT:   const some_handle_t * const tptr = (const some_handle_t * const)ptr;
// CHECK-PRINT:   os << (const void *)tptr << " (";
// CHECK-PRINT:   os << *tptr;
// CHECK-PRINT:   os << ")";
// CHECK-PRINT:   break;
// CHECK-PRINT: }

// Check that the tagged type information is used when printing function parameters
// CHECK-PRINT: std::ostream &operator<<(std::ostream &os, const struct function_a_params_t *params) {
// CHECK-PRINT: os << ".PropValue = " 
// CHECK-PRINT-NEXT: printTagged(os, *params->pPropValue, *params->pPropName, *params->pPropSize);
