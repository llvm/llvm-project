; RUN: llc -verify-machineinstrs -enable-misched -misched=shuffle -enable-aa-sched-mi -use-tbaa-in-sched-mi=0 -post-RA-scheduler=0 -mcpu=ppc64 < %s | FileCheck %s

; REQUIRES: asserts
; -misched=shuffle is NDEBUG only!

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%"class.llvm::MCOperand" = type { i8, %union.anon.110 }
%union.anon.110 = type { i64 }

define void @foo(i32 %v) {
entry:
  %MCOp = alloca %"class.llvm::MCOperand", align 8
  br label %next

; CHECK-LABEL: @foo

next:
  %sunkaddr18 = ptrtoint ptr %MCOp to i64
  %sunkaddr19 = add i64 %sunkaddr18, 8
  %sunkaddr20 = inttoptr i64 %sunkaddr19 to ptr
  store double 0.000000e+00, ptr %sunkaddr20, align 8, !tbaa !1
  %sunkaddr21 = ptrtoint ptr %MCOp to i64
  %sunkaddr22 = add i64 %sunkaddr21, 8
  %sunkaddr23 = inttoptr i64 %sunkaddr22 to ptr
  store i32 %v, ptr %sunkaddr23, align 8, !tbaa !2
  ret void

; Make sure that the 64-bit store comes first, regardless of what TBAA says
; about the two not aliasing!
; CHECK: li [[REG:[0-9]+]], 0
; CHECK: std [[REG]], -[[OFF:[0-9]+]](1)
; CHECK: stw 3, -[[OFF]](1)
; CHECK: blr
}

!0 = !{ !"root" }
!1 = !{ !"set1", !0 }
!2 = !{ !"set2", !0 }

