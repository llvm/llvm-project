; RUN: opt -safe-stack -S -mtriple=aarch64-linux-android < %s -o - | FileCheck --check-prefixes=TLS,ANDROID %s
; RUN: opt -safe-stack -S -mtriple=aarch64-unknown-fuchsia < %s -o - | FileCheck --check-prefixes=TLS,FUCHSIA %s

define void @foo() nounwind uwtable safestack sspreq {
entry:
; The first @llvm.thread.pointer is for the unsafe stack pointer, skip it.
; TLS: call ptr @llvm.thread.pointer()

; TLS: %[[TP2:.*]] = call ptr @llvm.thread.pointer()
; ANDROID: %[[B:.*]] = getelementptr i8, ptr %[[TP2]], i32 40
; FUCHSIA: %[[B:.*]] = getelementptr i8, ptr %[[TP2]], i32 -16
; TLS: %[[StackGuard:.*]] = load ptr, ptr %[[B]]
; TLS: store ptr %[[StackGuard]], ptr %[[StackGuardSlot:.*]]
  %a = alloca i128, align 16
  call void @Capture(ptr %a)

; TLS: %[[A:.*]] = load ptr, ptr %[[StackGuardSlot]]
; TLS: icmp ne ptr %[[StackGuard]], %[[A]]
  ret void
}

declare void @Capture(ptr)
