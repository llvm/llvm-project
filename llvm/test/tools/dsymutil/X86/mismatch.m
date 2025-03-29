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

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: cp %p/../Inputs/mismatch/1.o %p/../Inputs/mismatch/mismatch.pcm %t.dir
// RUN: cp %p/../Inputs/mismatch/1.o %t.dir/2.o
// RUN: dsymutil --verbose -f -oso-prepend-path=%t.dir \
// RUN:   -y %p/dummy-debug-map.map -o %t.bin 2>&1 | FileCheck %s
// RUN: dsymutil --linker parallel --verbose -f -oso-prepend-path=%t.dir \
// RUN:   -y %p/dummy-debug-map.map -o %t.bin 2>&1 | FileCheck %s

@import mismatch;

void f() {}
// Mismatch after importing the module.
// CHECK: warning: hash mismatch
// Mismatch in the cache.
// CHECK: warning: hash mismatch
// CHECK: cached
