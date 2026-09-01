; REQUIRES: x86-registered-target, amdgpu-registered-target

; RUN: opt -mtriple=x86_64 -passes='default<O2>' -S %s -o - | FileCheck %s --check-prefix=X86
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -passes='default<O2>' -S %s -o - | FileCheck %s --check-prefix=GFX908
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes='default<O2>' -S %s -o - | FileCheck %s --check-prefix=GFX900
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='default<O2>' -S %s -o - | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs -filetype=null
; RUN: opt -passes='default<O2>' -S %s -o - | FileCheck %s --check-prefix=NO-TARGET
; RUN: opt -mtriple=x86_64 -passes='lto<O2>' -S %s -o - | FileCheck %s --check-prefix=X86
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -passes='lto<O2>' -S %s -o - | FileCheck %s --check-prefix=GFX908
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes='lto<O2>' -S %s -o - | FileCheck %s --check-prefix=GFX900
; RUN: opt -passes='lto<O2>' -S %s -o - | FileCheck %s --check-prefix=NO-TARGET
; RUN: echo 'Contexts: []' | llvm-ctxprof-util fromYAML --input=- --output=%t.ctxprofdata
; RUN: opt -passes=assign-guid -S %s -o %t.guid.ll
; RUN: opt -mtriple=x86_64 -use-ctx-profile=%t.ctxprofdata -passes='thinlto<O2>' -S %t.guid.ll -o - | FileCheck %s --check-prefix=X86

define i32 @vgpr(i32 %x) {
; X86-LABEL: define i32 @vgpr(
; X86-NEXT:    [[SUM:%.*]] = add i32 [[X:%.*]], 1
; X86-NEXT:    ret i32 [[SUM]]
;
; GFX908-LABEL: define i32 @vgpr(
; GFX908:       [[HANDOFF:%.*]] = {{(tail )?}}call i32 @llvm.experimental.regalloc.handoff(i32 [[X:%.*]], metadata [[VGPR:![0-9]+]])
; GFX908-NEXT:  [[SUM:%.*]] = add i32 [[HANDOFF]], 1
; GFX908-NEXT:  ret i32 [[SUM]]
;
; GFX900-LABEL: define i32 @vgpr(
; GFX900:       [[HANDOFF:%.*]] = {{(tail )?}}call i32 @llvm.experimental.regalloc.handoff(i32 [[X:%.*]], metadata [[VGPR:![0-9]+]])
; GFX900-NEXT:  [[SUM:%.*]] = add i32 [[HANDOFF]], 1
; GFX900-NEXT:  ret i32 [[SUM]]
;
; NO-TARGET-LABEL: define i32 @vgpr(
; NO-TARGET:       call i32 @llvm.experimental.regalloc.handoff
  %handoff = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !0)
  %sum = add i32 %handoff, 1
  ret i32 %sum
}

define i32 @agpr(i32 %x) {
; X86-LABEL: define i32 @agpr(
; X86-NEXT:    [[SUM:%.*]] = add i32 [[X:%.*]], 1
; X86-NEXT:    ret i32 [[SUM]]
;
; GFX908-LABEL: define i32 @agpr(
; GFX908:       [[HANDOFF:%.*]] = {{(tail )?}}call i32 @llvm.experimental.regalloc.handoff(i32 [[X:%.*]], metadata [[AGPR:![0-9]+]])
; GFX908-NEXT:  [[SUM:%.*]] = add i32 [[HANDOFF]], 1
; GFX908-NEXT:  ret i32 [[SUM]]
;
; GFX900-LABEL: define i32 @agpr(
; GFX900-NEXT:    [[SUM:%.*]] = add i32 [[X:%.*]], 1
; GFX900-NEXT:    ret i32 [[SUM]]
;
; NO-TARGET-LABEL: define i32 @agpr(
; NO-TARGET:       call i32 @llvm.experimental.regalloc.handoff
  %handoff = call i32 @llvm.experimental.regalloc.handoff(i32 %x, metadata !1)
  %sum = add i32 %handoff, 1
  ret i32 %sum
}

define void @unused_agpr_loop(i32 %n) {
; X86-LABEL: define void @unused_agpr_loop(
; X86-NEXT:  entry:
; X86-NEXT:    ret void
;
; GFX908-LABEL: define void @unused_agpr_loop(
; GFX908:       call i32 @llvm.experimental.regalloc.handoff
; GFX908:       ret void
;
; GFX900-LABEL: define void @unused_agpr_loop(
; GFX900-NEXT:  entry:
; GFX900-NEXT:    ret void
entry:
  %positive = icmp sgt i32 %n, 0
  br i1 %positive, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %handoff = call i32 @llvm.experimental.regalloc.handoff(i32 %i, metadata !1)
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare i32 @llvm.experimental.regalloc.handoff(i32, metadata)

!0 = !{!"amdgpu.vgpr"}
!1 = !{!"amdgpu.agpr"}
