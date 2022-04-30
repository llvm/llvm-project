; RUN: llc -mtriple aarch64-unknown-windows -swift-async-fp=never -filetype asm -o - %s | FileCheck %s

; ModuleID = '_Concurrency.ll'
source_filename = "_Concurrency.ll"
target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-windows-msvc19.32.31302"

%swift.context = type { %swift.context*, void (%swift.context*)* }

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nounwind
define hidden swifttailcc void @"$ss23withCheckedContinuation8function_xSS_yScCyxs5NeverOGXEtYalFTQ0_"(i8* nocapture readonly %0) #1 {
entryresume.0:
  %1 = bitcast i8* %0 to i8**
  %2 = load i8*, i8** %1, align 8
  %3 = tail call i8** @llvm.swift.async.context.addr() #4
  store i8* %2, i8** %3, align 8
  %async.ctx.frameptr1 = getelementptr inbounds i8, i8* %2, i64 16
  %.reload.addr4 = getelementptr inbounds i8, i8* %2, i64 24
  %4 = bitcast i8* %.reload.addr4 to i8**
  %.reload5 = load i8*, i8** %4, align 8
  %.reload.addr = bitcast i8* %async.ctx.frameptr1 to i8**
  %.reload = load i8*, i8** %.reload.addr, align 8
  %5 = load i8*, i8** %1, align 8
  store i8* %5, i8** %3, align 8
  tail call swiftcc void @swift_task_dealloc(i8* %.reload5) #4
  tail call void @llvm.lifetime.end.p0i8(i64 -1, i8* %.reload5)
  tail call swiftcc void @swift_task_dealloc(i8* %.reload) #4
  %6 = getelementptr inbounds i8, i8* %5, i64 8
  %7 = bitcast i8* %6 to void (%swift.context*)**
  %8 = load void (%swift.context*)*, void (%swift.context*)** %7, align 8
  %9 = bitcast i8* %5 to %swift.context*
  musttail call swifttailcc void %8(%swift.context* %9) #4
  ret void
}

; NOTE: we do not see the canonical windows frame setup due to the `nounwind`
; attribtue on the function.

; CHECK: sub sp, sp, #64
; CHECK: stp x30, x29, [sp, #16]
; CHECK: add x29, sp, #16
; CHECK: stp x22, x21, [sp, #32]
; CHECK: sub x8, x29, #8
; CHECK: stp x20, x19, [sp, #48]
; CHECK: ldr x9, [x0]
; CHECK: str x9, [x8]

; Function Attrs: nounwind readnone
declare i8** @llvm.swift.async.context.addr() #2

; Function Attrs: argmemonly nounwind
declare dllimport swiftcc void @swift_task_dealloc(i8*) local_unnamed_addr #3

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
attributes #1 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #2 = { nounwind readnone }
attributes #3 = { argmemonly nounwind }
attributes #4 = { nounwind }

