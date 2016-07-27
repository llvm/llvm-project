; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s
; REQUIRES: global-isel
; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

; Tests for add.
; CHECK-LABEL: name: addi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_ADD s64 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0 
define i64 @addi64(i64 %arg1, i64 %arg2) {
  %res = add i64 %arg1, %arg2
  ret i64 %res
}

; Tests for alloca
; CHECK-LABEL: name: allocai64
; CHECK: stack:
; CHECK-NEXT:   - { id: 0, name: ptr1, offset: 0, size: 8, alignment: 8 }
; CHECK-NEXT:   - { id: 1, name: ptr2, offset: 0, size: 8, alignment: 1 }
; CHECK-NEXT:   - { id: 2, name: ptr3, offset: 0, size: 128, alignment: 8 }
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX p0 %stack.0.ptr1
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX p0 %stack.1.ptr2
; CHECK: %{{[0-9]+}}(64) = G_FRAME_INDEX p0 %stack.2.ptr3
define void @allocai64() {
  %ptr1 = alloca i64
  %ptr2 = alloca i64, align 1
  %ptr3 = alloca i64, i32 16
  ret void
}

; Tests for br.
; CHECK-LABEL: name: uncondbr
; CHECK: body:
;
; Entry basic block.
; CHECK: {{[0-9a-zA-Z._-]+}}:
;
; Make sure we have one successor and only one.
; CHECK-NEXT: successors: %[[END:[0-9a-zA-Z._-]+]]({{0x[a-f0-9]+ / 0x[a-f0-9]+}} = 100.00%)
;
; Check that we emit the correct branch.
; CHECK: G_BR unsized %[[END]]
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
; CHECK-LABEL: name: ori64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_OR s64 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @ori64(i64 %arg1, i64 %arg2) {
  %res = or i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: ori32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_OR s32 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @ori32(i32 %arg1, i32 %arg2) {
  %res = or i32 %arg1, %arg2
  ret i32 %res
}

; Tests for and.
; CHECK-LABEL: name: andi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_AND s64 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @andi64(i64 %arg1, i64 %arg2) {
  %res = and i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: andi32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_AND s32 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @andi32(i32 %arg1, i32 %arg2) {
  %res = and i32 %arg1, %arg2
  ret i32 %res
}

; Tests for sub.
; CHECK-LABEL: name: subi64
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](64) = G_SUB s64 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @subi64(i64 %arg1, i64 %arg2) {
  %res = sub i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: subi32
; CHECK: [[ARG1:%[0-9]+]](32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](32) = G_SUB s32 [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @subi32(i32 %arg1, i32 %arg2) {
  %res = sub i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: ptrtoint
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK: [[RES:%[0-9]+]](64) = G_PTRTOINT { s64, p0 } [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @ptrtoint(i64* %a) {
  %val = ptrtoint i64* %a to i64
  ret i64 %val
}

; CHECK-LABEL: name: inttoptr
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK: [[RES:%[0-9]+]](64) = G_INTTOPTR { p0, s64 } [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @inttoptr(i64 %a) {
  %val = inttoptr i64 %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK: [[RES:%[0-9]+]](64) = COPY [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @trivial_bitcast(i8* %a) {
  %val = bitcast i8* %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: bitcast
; CHECK: [[ARG1:%[0-9]+]](64) = COPY %x0
; CHECK: [[RES1:%[0-9]+]](64) = G_BITCAST { <2 x s32>, s64 } [[ARG1]]
; CHECK: [[RES2:%[0-9]+]](64) = G_BITCAST { s64, <2 x s32> } [[RES1]]
; CHECK: %x0 = COPY [[RES2]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @bitcast(i64 %a) {
  %res1 = bitcast i64 %a to <2 x i32>
  %res2 = bitcast <2 x i32> %res1 to i64
  ret i64 %res2
}

; CHECK-LABEL: name: load
; CHECK: [[ADDR:%[0-9]+]](64) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]](64) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]](64) = G_LOAD { s64, p0 } [[ADDR]] :: (load 8 from %ir.addr, align 16)
; CHECK: [[VAL2:%[0-9]+]](64) = G_LOAD { s64, p42 } [[ADDR42]] :: (load 8 from %ir.addr42)
; CHECK: [[SUM:%.*]](64) = G_ADD s64 [[VAL1]], [[VAL2]]
; CHECK: %x0 = COPY [[SUM]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @load(i64* %addr, i64 addrspace(42)* %addr42) {
  %val1 = load i64, i64* %addr, align 16
  %val2 = load i64, i64 addrspace(42)* %addr42
  %sum = add i64 %val1, %val2
  ret i64 %sum
}

; CHECK-LABEL: name: store
; CHECK: [[ADDR:%[0-9]+]](64) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]](64) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]](64) = COPY %x2
; CHECK: [[VAL2:%[0-9]+]](64) = COPY %x3
; CHECK: G_STORE { s64, p0 } [[VAL1]], [[ADDR]] :: (store 8 into %ir.addr, align 16)
; CHECK: G_STORE { s64, p42 } [[VAL2]], [[ADDR42]] :: (store 8 into %ir.addr42)
; CHECK: RET_ReallyLR
define void @store(i64* %addr, i64 addrspace(42)* %addr42, i64 %val1, i64 %val2) {
  store i64 %val1, i64* %addr, align 16
  store i64 %val2, i64 addrspace(42)* %addr42
  %sum = add i64 %val1, %val2
  ret void
}
