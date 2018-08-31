// RUN: dsymutil -f -oso-prepend-path=%p/../Inputs/modules-pruning \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --name isRef -p - | FileCheck %s

/* Compile with:
   cat >modules.modulemap <<EOF
   module Outer {
     module Template {
       header "template.h"
       export *
     }
   }
EOF
   clang++ -D TEMPLATE_H -E -o template.h modules-pruning.cpp
   clang++ -c -fcxx-modules -fmodules -fmodule-map-file=modules.modulemap \
     -g -gmodules -fmodules-cache-path=. \
     -Xclang -fdisable-module-hash modules-pruning.cpp -o 1.o
*/

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_TAG_module
// CHECK:     DW_TAG_module
// CHECK:       DW_TAG_class
// CHECK:         DW_TAG_member
// CHECK:           DW_AT_name ("isRef")
// CHECK:           DW_AT_declaration (true)
// CHECK:           DW_AT_const_value (1)
// CHECK-NOT: DW_TAG

#ifdef TEMPLATE_H

namespace M {
struct false_type {
  static const bool value = false;
};
struct true_type {
  static const bool value = true;
};

template <class T> struct is_reference      : false_type {};
template <class T> struct is_reference<T&>  : true_type {};

template<class T>
class Template {
public:
  static const bool isRef = is_reference<T>::value;
  Template() {}
};
}
#else

#include "template.h"

void foo() {
  M::Template<bool&> TB1;
  TB1.isRef;
}

#endif
