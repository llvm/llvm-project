; RUN: llvm-as %S/Inputs/linkage.b.ll -o %t.b.bc
; RUN: llvm-as %S/Inputs/linkage.c.ll -o %t.c.bc
; RUN: llvm-link -S %t.b.bc %t.c.bc | FileCheck %s -check-prefix=B -check-prefix=C -check-prefix=CU
; RUN: llvm-link -S -only-needed %t.b.bc %t.c.bc | FileCheck %s -check-prefix=B -check-prefix=C -check-prefix=CN
; RUN: llvm-link -S -internalize %t.b.bc %t.c.bc | FileCheck %s -check-prefix=B -check-prefix=CI
; RUN: llvm-link -S -internalize -only-needed %t.b.bc %t.c.bc | FileCheck %s -check-prefix=B -check-prefix=CN

C-LABEL: @X = global i32 5
CI-LABEL: @X = internal global i32 5
CU-LABEL:@U = global i32 6
CI-LABEL:@U = internal global i32 6
CN-NOT:@U

B-LABEL: define void @bar() {

C-LABEL: define i32 @foo()
CI-LABEL: define internal i32 @foo()

CU-LABEL:define i32 @unused() {
CI-LABEL:define internal i32 @unused() {
CN-NOT:@unused()
