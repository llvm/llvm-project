// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-readobj -h | FileCheck %s
// REQUIRES: timestamps

// CHECK: ImageFileHeader {
// CHECK:   TimeDateStamp:
// CHECK-NOT: 1970-01-01 00:00:00 (0x0)
