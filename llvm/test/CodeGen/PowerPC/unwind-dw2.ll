; RUN: llc -verify-machineinstrs < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  call void @llvm.eh.unwind.init()
  ret void
}

; Function Attrs: nounwind
declare void @llvm.eh.unwind.init() #0

attributes #0 = { nounwind }
