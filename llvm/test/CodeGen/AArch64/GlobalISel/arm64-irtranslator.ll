; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s
; REQUIRES: global-isel
; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

; Tests for add.
; CHECK: name: addi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_ADD { s64 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0 
define i64 @addi64(i64 %arg1, i64 %arg2) {
  %res = add i64 %arg1, %arg2
  ret i64 %res
}

; Tests for alloca
; CHECK: name: allocai64
; CHECK: stack:
; CHECK-NEXT:   - { id: 0, name: ptr1, offset: 0, size: 8, alignment: 8 }
; CHECK-NEXT:   - { id: 1, name: ptr2, offset: 0, size: 8, alignment: 1 }
; CHECK-NEXT:   - { id: 2, name: ptr3, offset: 0, size: 128, alignment: 8 }
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX { p0 } 0
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX { p0 } 1
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX { p0 } 2
define void @allocai64() {
  %ptr1 = alloca i64
  %ptr2 = alloca i64, align 1
  %ptr3 = alloca i64, i32 16
  ret void
}

; Tests for br.
; CHECK: name: uncondbr
; CHECK: body:
;
; Entry basic block.
; CHECK: {{[0-9a-zA-Z._-]+}}:
;
; Make sure we have one successor and only one.
; CHECK-NEXT: successors: %[[END:[0-9a-zA-Z._-]+]]({{0x[a-f0-9]+ / 0x[a-f0-9]+}} = 100.00%)
;
; Check that we emit the correct branch.
; CHECK: G_BR { unsized } %[[END]]
;
; Check that end contains the return instruction.
; CHECK: [[END]]:
; CHECK-NEXT: RET_ReallyLR
define void @uncondbr() {
  br label %end
end:
  ret void
}

; Tests for or.
; CHECK: name: ori64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_OR { s64 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @ori64(i64 %arg1, i64 %arg2) {
  %res = or i64 %arg1, %arg2
  ret i64 %res
}

; CHECK: name: ori32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_OR { s32 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @ori32(i32 %arg1, i32 %arg2) {
  %res = or i32 %arg1, %arg2
  ret i32 %res
}

; Tests for and.
; CHECK: name: andi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_AND { s64 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @andi64(i64 %arg1, i64 %arg2) {
  %res = and i64 %arg1, %arg2
  ret i64 %res
}

; CHECK: name: andi32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_AND { s32 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @andi32(i32 %arg1, i32 %arg2) {
  %res = and i32 %arg1, %arg2
  ret i32 %res
}

; Tests for sub.
; CHECK: name: subi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_SUB { s64 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @subi64(i64 %arg1, i64 %arg2) {
  %res = sub i64 %arg1, %arg2
  ret i64 %res
}

; CHECK: name: subi32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_SUB { s32 } [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @subi32(i32 %arg1, i32 %arg2) {
  %res = sub i32 %arg1, %arg2
  ret i32 %res
}
