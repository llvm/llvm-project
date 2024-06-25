; RUN: opt < %s -aarch64-stack-tagging -stack-tagging-use-stack-safety=0 -S -o - | FileCheck %s --check-prefixes=CHECK
; RUN: opt < %s -aarch64-stack-tagging -stack-tagging-use-stack-safety=0 -S -stack-tagging-record-stack-history=instr -o - | FileCheck %s --check-prefixes=INSTR
; RUN llc -mattr=+mte -stack-tagging-use-stack-safety=0 -stack-tagging-record-stack-history=instr %s -o - | FileCheck %s --check-prefixes=ASMINSTR


target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

declare void @use32(ptr)

define void @OneVar() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @use32(ptr %x)
  ret void
}

; CHECK-LABEL: define void @OneVar(
; CHECK:  [[BASE:%.*]] = call ptr @llvm.aarch64.irg.sp(i64 0)
; CHECK:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; CHECK:  [[TX:%.*]] = call ptr @llvm.aarch64.tagp.{{.*}}(ptr [[X]], ptr [[BASE]], i64 0)
; CHECK:  ret void

; INSTR-LABEL: define void @OneVar(
; INSTR:  [[BASE:%.*]] = call ptr @llvm.aarch64.irg.sp(i64 0)
; INSTR:  [[TLS:%.*]] = call ptr @llvm.thread.pointer()
; INSTR:  [[TLS_SLOT:%.*]] = getelementptr i8, ptr [[TLS]], i32 -24
; INSTR:  [[TLS_VALUE:%.*]] = load i64, ptr [[TLS_SLOT]], align 8
; INSTR:  [[FP:%.*]] = call ptr @llvm.frameaddress.p0(i32 0)
; INSTR:  [[FP_INT:%.*]] = ptrtoint ptr [[FP]] to i64
; INSTR:  [[BASE_INT:%.*]] = ptrtoint ptr [[BASE]] to i64
; INSTR:  [[BASE_TAG:%.*]] = and i64 [[BASE_INT]], 1080863910568919040
; INSTR:  [[TAGGED_FP:%.*]] = or i64 [[FP_INT]], [[BASE_TAG]]
; INSTR:  [[PC:%.*]] = call i64 @llvm.read_register.i64(metadata !0)
; INSTR:  [[TLS_VALUE_PTR:%.*]] = inttoptr i64 [[TLS_VALUE]] to ptr
; INSTR:  store i64 [[PC]], ptr [[TLS_VALUE_PTR]], align 8
; INSTR:  [[SECOND_SLOT:%.*]] = getelementptr i64, ptr [[TLS_VALUE_PTR]], i64 1
; INSTR:  store i64 [[TAGGED_FP]], ptr [[SECOND_SLOT]], align 8
; INSTR:  [[SIZE_IN_PAGES:%.*]] = ashr i64 [[TLS_VALUE]], 56
; INSTR:  [[WRAP_MASK_INTERMEDIARY:%.*]] = shl nuw nsw i64 [[SIZE_IN_PAGES]], 12
; INSTR:  [[WRAP_MASK:%.*]] = xor i64 [[WRAP_MASK_INTERMEDIARY]], -1
; INSTR:  [[NEXT_TLS_VALUE_BEFORE_WRAP:%.*]] = add i64 [[TLS_VALUE]], 16
; INSTR:  [[NEXT_TLS_VALUE:%.*]] = and i64 [[NEXT_TLS_VALUE_BEFORE_WRAP]], [[WRAP_MASK]]
; INSTR:  store i64 [[NEXT_TLS_VALUE]], ptr [[TLS_SLOT]], align 8
; INSTR:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; INSTR:  [[TX:%.*]] = call ptr @llvm.aarch64.tagp.{{.*}}(ptr [[X]], ptr [[BASE]], i64 0)
; INSTR:  [[PC:!.*]] = !{!"pc"}

; ASMINSTR-LABEL: OneVar:
; ASMINSTR:  mrs	[[TLS:x.*]], TPIDR_EL0
; ASMINSTR:  irg	[[BASE:x.*]], sp
; ASMINSTR:  adr	[[PC:x.*]], #0
; ASMINSTR:  ldur	[[TLS_SLOT:x.*]], [[[TLS]], #-24]
; ASMINSTR:  and	[[SP_TAG:x.*]], [[BASE]], #0xf00000000000000
; ASMINSTR:  orr	[[TAGGED_FP]], x29, [[SP_TAG]]
; ASMINSTR:  asr	[[TLS_SIZE:x.*]], [[TLS_SLOT]], #56
; ASMINSTR:  add	[[NEXT_TLS_VALUE_BEFORE_WRAP:x.*]], [[TLS_SLOT]], #16
; ASMINSTR:  stp	[[PC]], [[TAGGED_FP]], [[[TLS_SLOT]]]
; ASMINSTR:  bic	[[NEXT_TLS_VALUE:x.*]], [[NEXT_TLS_VALUE_BEFORE_WRAP]], [[TLS_SIZE]], lsl #12
; ASMINSTR:  stur	[[NEXT_TLS_VALUE]], [[[TLS]], #-24]
; ASMINSTR:  stg	[[BASE]], [[[BASE]]]
