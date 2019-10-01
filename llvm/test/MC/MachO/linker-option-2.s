// RUN: llvm-mc -n -triple x86_64-apple-darwin10 %s -filetype=obj | llvm-readobj --macho-linker-options | FileCheck %s

.linker_option "a"
.linker_option "a", "b"

// CHECK: Linker Options {
// CHECK:   Size: 16
// CHECK:   Strings [
// CHECK:     Value: a
// CHECK:   ]
// CHECK: }
// CHECK: Linker Options {
// CHECK:   Size: 16
// CHECK:   Strings [
// CHECK:     Value: a
// CHECK:     Value: b
// CHECK:   ]
// CHECK: }
