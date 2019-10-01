; Default behavior
; RUN: opt < %s -passes='unroll' -S | FileCheck %s -check-prefixes=ENABLE,COMMON

; Pass option
; RUN: opt < %s -passes='unroll<full-unroll-max=0>'  -S | FileCheck %s -check-prefixes=DISABLE,COMMON
; RUN: opt < %s -passes='unroll<full-unroll-max=30>' -S | FileCheck %s -check-prefixes=DISABLE,COMMON
; RUN: opt < %s -passes='unroll<full-unroll-max=36>' -S | FileCheck %s -check-prefixes=ENABLE,COMMON

; cl::opt option
; RUN: opt < %s -passes='unroll' -unroll-full-max-count=0 -S | FileCheck %s -check-prefixes=DISABLE,COMMON
; RUN: opt < %s -passes='unroll' -unroll-full-max-count=30 -S | FileCheck %s -check-prefixes=DISABLE,COMMON
; RUN: opt < %s -passes='unroll' -unroll-full-max-count=36 -S | FileCheck %s -check-prefixes=ENABLE,COMMON

; Pass option has a priority over cl::opt
; RUN: opt < %s -passes='unroll<full-unroll-max=30>' -unroll-full-max-count=36 -S | FileCheck %s -check-prefixes=DISABLE,COMMON
; RUN: opt < %s -passes='unroll<full-unroll-max=36>' -unroll-full-max-count=30 -S | FileCheck %s -check-prefixes=ENABLE,COMMON

define void @test() {
; COMMON-LABEL: @test(
 entry:
  br label %loop

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add i32 %idx, 1
  %be = icmp slt i32 %idx, 32
  br i1 %be, label %loop, label %exit

; COMMON: loop:
; DISABLE:  %be = icmp slt i32 %idx, 32
; ENABLE-NOT:  %be = icmp slt i32 %idx, 32

 exit:
  ret void
}
