; RUN: opt -S -mtriple=aarch64-apple-ios -loop-vectorize -enable-interleaved-mem-accesses -force-vector-width=2 < %s | FileCheck %s

; Reproducer for address space fault in the LoopVectorizer (pr31900). Added
; different sized address space pointers (p:16:16-p4:32:16) to the aarch64
; datalayout to reproduce the fault.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128-p:16:16-p4:32:16"

; Check that all the loads are scalarized
; CHECK: load i16, i16*
; CHECK: load i16, i16*
; CHECK: load i16, i16 addrspace(4)*
; CHECK: load i16, i16 addrspace(4)*

%rec1445 = type { i16, i16, i16, i16, i16 }

define void @foo() {
bb1:
  br label %bb4

bb4:
  %tmp1 = phi i16 [ undef, %bb1 ], [ %_tmp1013, %bb4 ]
  %tmp2 = phi %rec1445* [ undef, %bb1 ], [ %_tmp1015, %bb4 ]
  %tmp3 = phi %rec1445 addrspace(4)* [ undef, %bb1 ], [ %_tmp1017, %bb4 ]
  %0 = getelementptr %rec1445, %rec1445* %tmp2, i16 0, i32 1
  %_tmp987 = load i16, i16* %0, align 1
  %1 = getelementptr %rec1445, %rec1445 addrspace(4)* %tmp3, i32 0, i32 1
  %_tmp993 = load i16, i16 addrspace(4)* %1, align 1
  %_tmp1013 = add i16 %tmp1, 1
  %_tmp1015 = getelementptr %rec1445, %rec1445* %tmp2, i16 1
  %_tmp1017 = getelementptr %rec1445, %rec1445 addrspace(4)* %tmp3, i32 1
  %_tmp1019 = icmp ult i16 %_tmp1013, 24
  br i1 %_tmp1019, label %bb4, label %bb16

bb16:
  unreachable
}
