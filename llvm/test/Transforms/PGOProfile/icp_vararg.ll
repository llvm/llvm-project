; RUN: opt < %s -pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICALL-PROM
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global i32 (i32, ...)* null, align 8

define i32 @va_func(i32 %num, ...) {
entry:
  ret i32 0
}

define i32 @bar() #1 {
entry:
  %tmp = load i32 (i32, ...)*, i32 (i32, ...)** @foo, align 8
; ICALL-PROM:  [[BITCAST:%[0-9]+]] = bitcast i32 (i32, ...)* %tmp to i8*
; ICALL-PROM:  [[CMP:%[0-9]+]] = icmp eq i8* [[BITCAST]], bitcast (i32 (i32, ...)* @va_func to i8*)
; ICALL-PROM:  br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICALL-PROM:if.true.direct_targ:
; ICALL-PROM:  [[DIRCALL_RET:%[0-9]+]] = call i32 (i32, ...) @va_func(i32 3, i32 12, i32 22, i32 4)
; ICALL-PROM:  br label %if.end.icp
  %call = call i32 (i32, ...) %tmp(i32 3, i32 12, i32 22, i32 4), !prof !1
; ICALL-PROM:if.false.orig_indirect:
; ICALL-PROM:  %call = call i32 (i32, ...) %tmp(i32 3, i32 12, i32 22, i32 4)
; ICALL-PROM:  br label %if.end.icp
  ret i32 %call
; ICALL-PROM:if.end.icp:
; ICALL-PROM:  [[PHI_RET:%[0-9]+]] = phi i32 [ %call, %if.false.orig_indirect ], [ [[DIRCALL_RET]], %if.true.direct_targ ]
; ICALL-PROM:  ret i32 [[PHI_RET]] 

}

!1 = !{!"VP", i32 0, i64 12345, i64 989055279648259519, i64 12345}
; ICALL-PROM: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 12345, i32 0}
