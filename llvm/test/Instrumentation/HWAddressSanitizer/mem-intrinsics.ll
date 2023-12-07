; RUN: opt -S -passes=hwasan -hwasan-use-stack-safety=0 %s | FileCheck --check-prefixes=CHECK,CHECK-PREFIX %s
; RUN: opt -S -passes=hwasan -hwasan-kernel -hwasan-use-stack-safety=0 %s | FileCheck --check-prefixes=CHECK,CHECK-NOPREFIX %s
; RUN: opt -S -passes=hwasan -hwasan-kernel -hwasan-kernel-mem-intrinsic-prefix -hwasan-use-stack-safety=0 %s | FileCheck --check-prefixes=CHECK,CHECK-PREFIX %s
; RUN: opt -S -passes=hwasan -hwasan-use-stack-safety=0 -hwasan-match-all-tag=0 %s | FileCheck --check-prefixes=CHECK,CHECK-MATCH-ALL-TAG %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() sanitize_hwaddress {
; CHECK-LABEL: main
entry:
  %retval = alloca i32, align 4
  %Q = alloca [10 x i8], align 1
  %P = alloca [10 x i8], align 1
  store i32 0, ptr %retval, align 4

  call void @llvm.memset.p0.i64(ptr align 1 %Q, i8 0, i64 10, i1 false)
; CHECK-PREFIX: call ptr @__hwasan_memset(
; CHECK-NOPREFIX: call ptr @memset(
; CHECK-MATCH-ALL-TAG: call ptr @__hwasan_memset_match_all(ptr %Q.hwasan, i32 0, i64 10, i8 0)

  %add.ptr = getelementptr inbounds i8, ptr %Q, i64 5

  call void @llvm.memmove.p0.p0.i64(ptr align 1 %Q, ptr align 1 %add.ptr, i64 5, i1 false)
; CHECK-PREFIX: call ptr @__hwasan_memmove(
; CHECK-NOPREFIX: call ptr @memmove(
; CHECK-MATCH-ALL-TAG: call ptr @__hwasan_memmove_match_all(ptr %Q.hwasan, ptr %add.ptr, i64 5, i8 0)


  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 10, i1 false)
; CHECK-PREFIX: call ptr @__hwasan_memcpy(
; CHECK-NOPREFIX: call ptr @memcpy(
; CHECK-MATCH-ALL-TAG: call ptr @__hwasan_memcpy_match_all(ptr %P.hwasan, ptr %Q.hwasan, i64 10, i8 0)
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

define void @memintr_test_nosanitize(ptr %a, ptr %b) nounwind uwtable {
  entry:
  tail call void @llvm.memset.p0.i64(ptr %a, i8 0, i64 100, i1 false)
  tail call void @llvm.memmove.p0.p0.i64(ptr %a, ptr %b, i64 100, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 100, i1 false)
  ret void
}
; CHECK-LABEL: memintr_test_nosanitize
; CHECK: @llvm.memset
; CHECK: @llvm.memmove
; CHECK: @llvm.memcpy
; CHECK: ret void
