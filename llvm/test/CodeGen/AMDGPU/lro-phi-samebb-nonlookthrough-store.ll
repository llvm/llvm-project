; RUN: opt -S -passes=amdgpu-late-codegenprepare \
; RUN:   -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a %s | FileCheck %s

; Goal: With a loop-header PHI in illegal vector type and a same-BB
; non-lookthrough user (vector add) in the header, LRO should still coerce
; the PHI to i32 because a profitable sink (store) exists across BB.

define amdgpu_kernel void @phi_samebb_nonlookthrough_store(
    ptr addrspace(1) %out, <4 x i8> %v, i1 %exit) {
; CHECK-LABEL: @phi_samebb_nonlookthrough_store(
entry:
  br label %loop

loop:                                             ; preds = %entry, %loop
  ; Loop-carried PHI in illegal vector type.
  %acc = phi <4 x i8> [ zeroinitializer, %entry ], [ %acc.next, %loop ]

  ; Same-BB non-lookthrough use in header.
  %acc.next = add <4 x i8> %acc, %v

  ; Make it a real loop: either iterate or exit to the sink block.
  br i1 %exit, label %store, label %loop

store:                                            ; preds = %loop
  ; The across-BB sink: storing the PHI coerced to i32.
  %acc.bc = bitcast <4 x i8> %acc to i32
  store i32 %acc.bc, ptr addrspace(1) %out, align 4
  ret void
}

; After AMDGPULateCodeGenPrepare we expect:
;  - PHI is coerced to i32
;  - A header bitcast materializes for the add
; This proves the same-BB non-lookthrough user (add) did not get pruned
; when the def is a PHI.

; CHECK: loop:
; CHECK:   %[[ACC_TC:[^ ]+]] = phi i32
; CHECK:   %[[ACC_TC_BC:[^ ]+]] = bitcast i32 %[[ACC_TC]] to <4 x i8>
; CHECK:   %[[ACC_NEXT:[^ ]+]] = add <4 x i8> %[[ACC_TC_BC]], %v
; CHECK:   br i1 %exit, label %store, label %loop
; CHECK: store:
; CHECK:   %[[ACC_TC_BC2:[^ ]+]] = bitcast i32 %[[ACC_TC]] to <4 x i8>
; CHECK:   %[[ST_I32:[^ ]+]] = bitcast <4 x i8> %[[ACC_TC_BC2]] to i32
; CHECK:   store i32 %[[ST_I32]],

