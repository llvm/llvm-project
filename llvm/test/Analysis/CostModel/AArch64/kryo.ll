; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mcpu=kryo | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: vectorInstrCost
define void @vectorInstrCost() {

    ; Vector extracts - extracting the first element should have a zero cost;
    ; all other elements should have a cost of two.
    ;
    ; CHECK: cost of 0 {{.*}} extractelement <2 x i64> undef, i32 0
    ; CHECK: cost of 2 {{.*}} extractelement <2 x i64> undef, i32 1
    %t1 = extractelement <2 x i64> undef, i32 0
    %t2 = extractelement <2 x i64> undef, i32 1

    ; Vector inserts - inserting the first element should have a zero cost; all
    ; other elements should have a cost of two.
    ;
    ; CHECK: cost of 0 {{.*}} insertelement <2 x i64> undef, i64 undef, i32 0
    ; CHECK: cost of 2 {{.*}} insertelement <2 x i64> undef, i64 undef, i32 1
    %t3 = insertelement <2 x i64> undef, i64 undef, i32 0
    %t4 = insertelement <2 x i64> undef, i64 undef, i32 1

    ret void
}
