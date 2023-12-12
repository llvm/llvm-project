/* Compile with:
   cat >modules.modulemap <<EOF
     module Empty {
       header "Empty.h"
     }
EOF
   touch Empty.h
   clang -c -fmodules -fmodule-map-file=modules.modulemap \
     -g -gmodules -fmodules-cache-path=. \
     -Xclang -fdisable-module-hash modules-empty.m -o 1.o
*/

// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: cp %p/../Inputs/modules-empty/1.o %p/../Inputs/modules-empty/Empty.pcm \
// RUN: %t.dir
// RUN: dsymutil -f -oso-prepend-path=%t.dir \
// RUN:   -verify \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-info - | FileCheck %s
// RUN: dsymutil --linker llvm -f -oso-prepend-path=%t.dir \
// RUN:   -verify \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-info - | FileCheck %s

#include "Empty.h"
int main() {
  return 0;
}

// The empty CU from the pcm should not get copied into the dSYM.
// Check that module name occured only once.
// CHECK: "Empty"
// CHECK-NOT: "Empty"
