// RUN: llvm-mc -triple i386-apple-darwin9 %s -o - | FileCheck %s

.text
// CHECK: .section __TEXT,__text

.pushsection __DATA, __data
// CHECK: .section __DATA,__data

.pushsection __TEXT, initcode
// CHECK: .section __TEXT,initcode
        
.popsection
// CHECK: .section __DATA,__data
        
.popsection
// CHECK: .section __TEXT,__text
