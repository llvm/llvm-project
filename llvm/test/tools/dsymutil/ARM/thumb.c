// RUN: llvm-dsymutil -f -oso-prepend-path=%p/.. %p/../Inputs/thumb.armv7m -o - | llvm-dwarfdump - | FileCheck %s
// RUN: llvm-dsymutil -arch armv7m -f -oso-prepend-path=%p/.. %p/../Inputs/thumb.armv7m -o - | llvm-dwarfdump - | FileCheck %s

/* Compile with:
   clang -c thumb.c -arch armv7m -g
   clang thumb.o -o thumb.armv7m -arch armv7m -nostdlib -static -Wl,-e,_start
*/

void start() {
}

CHECK: DW_AT_name{{.*}}"thumb.c"
CHECK: DW_AT_name{{.*}}"start"
