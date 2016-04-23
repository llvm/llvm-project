; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-code16"

; Function Attrs: nounwind
define i32 @main() #0 {
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0
}

; CHECK: .code16
; CHECK-LABEL: main


attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 265439) (llvm/trunk 265567)"}