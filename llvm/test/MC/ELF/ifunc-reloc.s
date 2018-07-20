// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r | FileCheck %s
        .global sym
        .type sym, @gnu_indirect_function
alias: 
        .global alias
        .type alias, @function
        .set sym, alias


        callq sym

// CHECK: Relocations [
// CHECK-NEXT:   Section {{.*}} .rela.text {
// CHECK-NEXT:     0x1 R_X86_64_PLT32 sym 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:   }
// CHECK-NEXT: ]
