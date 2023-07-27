; RUN: llc -mtriple aarch64-unknown-windows -swift-async-fp=never -filetype asm -o - %s | FileCheck %s

; ModuleID = '_Concurrency.ll'
source_filename = "_Concurrency.ll"
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.32.31302"

%swift.context = type { ptr, ptr }

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #0

; Function Attrs: nounwind
define hidden swifttailcc void @"$ss23withCheckedContinuation8function_xSS_yScCyxs5NeverOGXEtYalFTQ0_"(ptr nocapture readonly %0) #1 {
entryresume.0:
  %1 = load ptr, ptr %0, align 8
  %2 = tail call ptr @llvm.swift.async.context.addr() #4
  store ptr %1, ptr %2, align 8
  %async.ctx.frameptr1 = getelementptr inbounds i8, ptr %1, i64 16
  %.reload.addr4 = getelementptr inbounds i8, ptr %1, i64 24
  %.reload5 = load ptr, ptr %.reload.addr4, align 8
  %.reload = load ptr, ptr %async.ctx.frameptr1, align 8
  %3 = load ptr, ptr %0, align 8
  store ptr %3, ptr %2, align 8
  tail call swiftcc void @swift_task_dealloc(ptr %.reload5) #4
  tail call void @llvm.lifetime.end.p0(i64 -1, ptr %.reload5)
  tail call swiftcc void @swift_task_dealloc(ptr %.reload) #4
  %4 = getelementptr inbounds i8, ptr %3, i64 8
  %5 = load ptr, ptr %4, align 8
  musttail call swifttailcc void %5(ptr %3) #4
  ret void
}

; NOTE: we do not see the canonical windows frame setup due to the `nounwind`
; attribtue on the function.

; CHECK: sub sp, sp, #48
; CHECK: stp x30, x29, [sp, #24]
; CHECK: add x29, sp, #24
; CHECK: str x19, [sp, #40]
; CHECK: sub x8, x29, #8
; CHECK: ldr x9, [x0]
; CHECK: str x9, [x8]

; Function Attrs: nounwind readnone
declare ptr @llvm.swift.async.context.addr() #2

; Function Attrs: argmemonly nounwind
declare dllimport swiftcc void @swift_task_dealloc(ptr) local_unnamed_addr #3

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
attributes #1 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #2 = { nounwind readnone }
attributes #3 = { argmemonly nounwind }
attributes #4 = { nounwind }

