; RUN: llvm-link -S %S/Inputs/linkage.b.ll %S/Inputs/linkage.c.ll | FileCheck %s -check-prefix=B -check-prefix=C -check-prefix=CU
; RUN: llvm-link -S -only-needed %S/Inputs/linkage.b.ll %S/Inputs/linkage.c.ll | FileCheck %s -check-prefix=B -check-prefix=C -check-prefix=CN
; RUN: llvm-link -S -internalize %S/Inputs/linkage.b.ll %S/Inputs/linkage.c.ll | FileCheck %s -check-prefix=B -check-prefix=CI
; RUN: llvm-link -S -internalize -only-needed %S/Inputs/linkage.b.ll %S/Inputs/linkage.c.ll | FileCheck %s -check-prefix=B -check-prefix=CN
; RUN: llvm-link -S -internalize %S/Inputs/linkage.b.ll %S/Inputs/linkage.c.ll %S/Inputs/linkage.d.ll | FileCheck %s -check-prefix=B -check-prefix=DI

C-LABEL: @X = global i32 5
CI-LABEL: @X = internal global i32 5
CU-LABEL:@U = global i32 6
CI-LABEL:@U = internal global i32 6
CN-NOT:@U
DI-LABEL: @llvm.used = appending global [2 x ptr] [ptr @Y, ptr @foo], section "llvm.metadata"
DI-LABEL: @Y = global i8 42

B-LABEL: define void @bar() {

C-LABEL: define i32 @foo()
CI-LABEL: define internal i32 @foo()

CU-LABEL:define i32 @unused() {
CI-LABEL:define internal i32 @unused() {
CN-NOT:@unused()

DI-LABEL: define internal i32 @foo.6()
DI-LABEL: define i64 @foo()
