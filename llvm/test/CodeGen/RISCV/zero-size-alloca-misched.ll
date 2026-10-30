; RUN: llc -mtriple=riscv64 < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -global-isel < %s | FileCheck %s

; A zero-size alloca whose frame object disagreed with it let BasicAA report
; NoAlias for an access that can reach the merged slot. Stack coloring merges
; %slot0 into %empty's slot and rewrites its uses, so %empty describes the slot
; for alias analysis.
;
; Both slot stores must precede the load through %p, and both reloads must
; follow the store through %p.

define i16 @flip_selected_slot(i1 %c, i8 %v0, i8 %v1) nounwind {
; CHECK-LABEL: flip_selected_slot:
; CHECK-DAG:     sb {{[a-z0-9]+}}, [[SLOT_A:[0-9]+]](sp)
; CHECK-DAG:     sb {{[a-z0-9]+}}, [[SLOT_B:[0-9]+]](sp)
; CHECK:         lbu [[OLD:[a-z0-9]+]], 0([[P:[a-z0-9]+]])
; CHECK:         xori [[NEW:[a-z0-9]+]], [[OLD]], 1
; CHECK:         sb [[NEW]], 0([[P]])
; CHECK-DAG:     lbu {{[a-z0-9]+}}, [[SLOT_A]](sp)
; CHECK-DAG:     lbu {{[a-z0-9]+}}, [[SLOT_B]](sp)
; CHECK:         ret
entry:
  %empty = alloca [0 x i8], align 1
  %slot0 = alloca i8, align 1
  %slot1 = alloca i8, align 1
  ; Unused; makes the frame large enough for stack coloring to run.
  %pad = alloca [16 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr %empty)
  call void @use(ptr %empty)
  call void @llvm.lifetime.end.p0(ptr %empty)
  call void @llvm.lifetime.start.p0(ptr %slot0)
  call void @llvm.lifetime.start.p0(ptr %slot1)
  br i1 %c, label %pick0, label %pick1

pick0:
  br label %join

pick1:
  br label %join

join:
  %p = phi ptr [ %slot0, %pick0 ], [ %slot1, %pick1 ]
  store i8 %v0, ptr %slot0, align 1
  store i8 %v1, ptr %slot1, align 1
  %old = load i8, ptr %p, align 1
  %new = xor i8 %old, 1
  store i8 %new, ptr %p, align 1
  %r0 = load i8, ptr %slot0, align 1
  %r1 = load i8, ptr %slot1, align 1
  call void @llvm.lifetime.end.p0(ptr %slot0)
  call void @llvm.lifetime.end.p0(ptr %slot1)
  %z0 = zext i8 %r0 to i16
  %z1 = zext i8 %r1 to i16
  %hi = shl i16 %z1, 8
  %res = or i16 %hi, %z0
  ret i16 %res
}

declare void @use(ptr)
