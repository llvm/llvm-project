; RUN: opt -safe-stack -S -mtriple=aarch64-linux-android < %s -o - | FileCheck --check-prefixes=TLS,ANDROID %s
; RUN: opt -safe-stack -S -mtriple=aarch64-unknown-fuchsia < %s -o - | FileCheck --check-prefixes=TLS,FUCHSIA %s

define void @foo() nounwind uwtable safestack sspreq {
entry:
; The first @llvm.thread.pointer is for the unsafe stack pointer, skip it.
; TLS: call i8* @llvm.thread.pointer()

; TLS: %[[TP2:.*]] = call i8* @llvm.thread.pointer()
; ANDROID: %[[B:.*]] = getelementptr i8, i8* %[[TP2]], i32 40
; FUCHSIA: %[[B:.*]] = getelementptr i8, i8* %[[TP2]], i32 -16
; TLS: %[[C:.*]] = bitcast i8* %[[B]] to i8**
; TLS: %[[StackGuard:.*]] = load i8*, i8** %[[C]]
; TLS: store i8* %[[StackGuard]], i8** %[[StackGuardSlot:.*]]
  %a = alloca i128, align 16
  call void @Capture(i128* %a)

; TLS: %[[A:.*]] = load i8*, i8** %[[StackGuardSlot]]
; TLS: icmp ne i8* %[[StackGuard]], %[[A]]
  ret void
}

declare void @Capture(i128*)
