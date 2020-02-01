; RUN: opt -S -early-cse < %s | FileCheck %s
; RUN: opt -S -basicaa -early-cse-memssa < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define i64 @f(i64 %x) #0 {
entry:
  %0 = call i64 @llvm.read_register.i64(metadata !0)
  call void bitcast (void (...)* @foo to void ()*)()
  %1 = call i64 @llvm.read_register.i64(metadata !0)
  %add = add nsw i64 %0, %1
  ret i64 %add
}

; CHECK-LABEL: @f
; CHECK: call i64 @llvm.read_register.i64
; CHECK: call i64 @llvm.read_register.i64

; Function Attrs: nounwind readnone
declare i64 @llvm.read_register.i64(metadata) #1

; Function Attrs: nounwind
declare void @llvm.write_register.i64(metadata, i64) #2

declare void @foo(...)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.named.register.r1 = !{!0}

!0 = !{!"r1"}

