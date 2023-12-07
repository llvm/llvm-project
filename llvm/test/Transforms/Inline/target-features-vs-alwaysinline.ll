; RUN: opt < %s -passes=inline -pass-remarks-missed=inline -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure that we do not inline callee into caller.  If we inline
; callee into caller, caller would end pu with AVX512 intrinsics even
; though it is not allowed to use AVX512 instructions.
; CHECK: remark: [[MSG:.*]] because it should never be inlined (cost=never): conflicting target attributes

define void @caller(ptr %0) {
; CHECK-LABEL: define void @caller
; CHECK-SAME: (ptr [[TMP0:%.*]]) {
; CHECK-NEXT:    call void @callee(ptr [[TMP0]], i64 0, i32 0) #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    ret void
;
  call void @callee(ptr %0, i64 0, i32 0) #1
  ret void
}

define available_externally void @callee(ptr %0, i64 %1, i32 %2) #0 {
; CHECK-LABEL: define available_externally void @callee
; CHECK-SAME: (ptr [[TMP0:%.*]], i64 [[TMP1:%.*]], i32 [[TMP2:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    [[TMP4:%.*]] = call <16 x float> @llvm.x86.avx512.min.ps.512(<16 x float> zeroinitializer, <16 x float> zeroinitializer, i32 0)
; CHECK-NEXT:    store <16 x float> [[TMP4]], ptr [[TMP0]], align 1
; CHECK-NEXT:    ret void
;
  %4 = call <16 x float> @llvm.x86.avx512.min.ps.512(<16 x float> zeroinitializer, <16 x float> zeroinitializer, i32 0)
  store <16 x float> %4, ptr %0, align 1
  ret void
}

declare <16 x float> @llvm.x86.avx512.min.ps.512(<16 x float>, <16 x float>, i32 immarg)

attributes #0 = { "target-features"="+aes,+avx,+avx2,+avx512bw,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt" }
attributes #1 = { alwaysinline }
