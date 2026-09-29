// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .size bar, . - bar
.Ltmp01:
       .size foo, .Ltmp01 - foo
.Ltmp0:
       .size qux, .Ltmp0 - qux

// CHECK: .Ltmp0:
// CHECK: .size  bar, .Ltmp0-bar
// CHECK: .Ltmp01
// CHECK: .size foo, .Ltmp01-foo
// CHECK: .Ltmp00
// CHECK: .size qux, .Ltmp00-qux
