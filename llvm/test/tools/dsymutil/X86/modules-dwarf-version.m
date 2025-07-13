/* Compile with (using the module from the modules.m teastcase):
   clang -c -fmodules -fmodule-map-file=modules.modulemap \
     -gdwarf-2 -gmodules -fmodules-cache-path=. \
     -Xclang -fdisable-module-hash modules.m -o 1.o
*/

// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: cp %p/../Inputs/modules/Bar.pcm %t.dir
// RUN: cp %p/../Inputs/modules-dwarf-version/1.o %t.dir
// RUN: dsymutil -verify -f -oso-prepend-path=%t.dir \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-info - | FileCheck %s
// RUN: dsymutil --linker parallel -verify -f -oso-prepend-path=%t.dir \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-info - | FileCheck %s

@import Bar;
int main(int argc, char **argv) {
  struct Bar bar;
  bar.value = argc;
  return bar.value;
}

// CHECK: Compile Unit: {{.*}}version = 0x0004
// CHECK: Compile Unit: {{.*}}version = 0x0002
