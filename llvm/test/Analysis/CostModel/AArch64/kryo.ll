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

; CHECK-LABEL: vectorInstrExtractCost
define i64 @vectorInstrExtractCost(<4 x i64> %vecreg) {
    
    ; Vector extracts - extracting each element at index 0 is considered
    ; free in the current implementation. When extracting element at index
    ; 2, 2 is rounded to 0, so extracting element at index 2 has cost 0 as 
    ; well.
    ;
    ; CHECK: cost of 2 {{.*}} extractelement <4 x i64> %vecreg, i32 1
    ; CHECK: cost of 0 {{.*}} extractelement <4 x i64> %vecreg, i32 2
    %t1 = extractelement <4 x i64> %vecreg, i32 1
    %t2 = extractelement <4 x i64> %vecreg, i32 2
    %ele = add i64 %t2, 1
    %cond = icmp eq i64 %t1, %ele

    ; CHECK: cost of 0 {{.*}} extractelement <4 x i64> %vecreg, i32 0
    ; CHECK: cost of 2 {{.*}} extractelement <4 x i64> %vecreg, i32 3
    %t0 = extractelement <4 x i64> %vecreg, i32 0
    %t3 = extractelement <4 x i64> %vecreg, i32 3
    %val = select i1 %cond, i64 %t0 , i64 %t3

    ret i64 %val
}
