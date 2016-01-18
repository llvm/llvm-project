// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-readobj -h | FileCheck %s

// CHECK: ImageFileHeader {
// CHECK:   TimeDateStamp: {{.*}}
