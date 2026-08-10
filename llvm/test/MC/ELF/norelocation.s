// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck  %s

        call bar
bar:

// CHECK:      Relocations [
// CHECK-NEXT: ]
