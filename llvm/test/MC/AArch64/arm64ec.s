// RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj %s -o - | llvm-objdump -d -r - | FileCheck %s
// CHECK: file format coff-arm64ec
// CHECK: add x0, x1, x2
add x0, x1, x2
