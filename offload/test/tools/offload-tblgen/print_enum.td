// RUN: %offload-tblgen -gen-print-header -I %S/../../../liboffload/API %s | %fcheck-generic

// Check that print helpers are created for enums

include "APIDefs.td"

// CHECK: #include <type_traits>
// CHECK: #ifdef OFFLOAD_PRINT_USE_STD_OSTREAM
// CHECK-NEXT: #include <ostream>
// CHECK-NEXT: namespace offload::detail {
// CHECK-NEXT: using print_ostream = std::ostream;
// CHECK-NEXT: }
// CHECK-NEXT: #else
// CHECK-NEXT: #include <llvm/Support/raw_ostream.h>
// CHECK-NEXT: namespace offload::detail {
// CHECK-NEXT: using print_ostream = llvm::raw_ostream;
// CHECK-NEXT: }
// CHECK-NEXT: #endif

def my_enum_t : Enum {
  let desc = "An example enum";
  let etors =[
    Etor<"VALUE_ONE", "The first enum value">,
    Etor<"VALUE_TWO", "The second enum value">,
    Etor<"VALUE_THREE", "The third enum value">,
    Etor<"VALUE_FOUR", "The fourth enum value">,
  ];
}

// CHECK: inline offload::detail::print_ostream &operator<<(offload::detail::print_ostream &os, enum my_enum_t value)
// CHECK: switch (value) {
// CHECK: case MY_ENUM_VALUE_ONE:
// CHECK:   os << "MY_ENUM_VALUE_ONE";
// CHECK:   break;
// CHECK: case MY_ENUM_VALUE_TWO:
// CHECK:   os << "MY_ENUM_VALUE_TWO";
// CHECK:   break;
// CHECK: case MY_ENUM_VALUE_THREE:
// CHECK:   os << "MY_ENUM_VALUE_THREE";
// CHECK:   break;
// CHECK: case MY_ENUM_VALUE_FOUR:
// CHECK:   os << "MY_ENUM_VALUE_FOUR";
// CHECK:   break;
// CHECK: default:
// CHECK:   os << "unknown enumerator";
// CHECK:   break;
