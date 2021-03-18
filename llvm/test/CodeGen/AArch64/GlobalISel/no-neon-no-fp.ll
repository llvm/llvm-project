; RUN: llc -o - -verify-machineinstrs -global-isel -global-isel-abort=2 %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-unknown"

; We should fall back in the translator if we don't have no-neon/no-fp support.
; CHECK: Instruction selection used fallback path for foo
define void @foo(i128 *%ptr) #0 align 2 {
entry:
  store i128 0, i128* %ptr, align 16
  ret void
}

; This test below will crash the legalizer due to trying to use legacy rules,
; if we don't fall back in the translator.
declare i1 @zoo()
; CHECK: Instruction selection used fallback path for bar
define i32 @bar() #0 {
  %1 = call zeroext i1 @zoo()
  %2 = zext i1 %1 to i32
  ret i32 %2
}

attributes #0 = { "use-soft-float"="false" "target-features"="-fp-armv8,-neon" }

