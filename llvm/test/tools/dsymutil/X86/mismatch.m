/* Compile with:
   cat >modules.modulemap <<EOF
     module mismatch {
       header "mismatch.h"
     }
   EOF
   echo "struct s;"> mismatch.h
   clang -cc1 -emit-obj -fmodules -fmodule-map-file=modules.modulemap \
      -fmodule-format=obj -g -dwarf-ext-refs -fmodules-cache-path=. \
      -fdisable-module-hash mismatch.m -o 1.o
   echo > mismatch.h
   clang -cc1 -emit-obj -fmodules -fmodule-map-file=modules.modulemap \
      -fmodule-format=obj -g -dwarf-ext-refs -fmodules-cache-path=. \
      -fdisable-module-hash mismatch.m -o /dev/null
*/

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/mismatch \
// RUN:   -y %p/dummy-debug-map.map -o %t.bin 2>&1 | FileCheck %s

@import mismatch;

void f() {}
// CHECK: warning: hash mismatch
