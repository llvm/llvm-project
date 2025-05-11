// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | llvm-readelf --arch-specific - | FileCheck %s --check-prefix=CHECK

// test llvm-readelf with empty file.

// CHECK: BuildAttributes {
// CHECK-NEXT: }
