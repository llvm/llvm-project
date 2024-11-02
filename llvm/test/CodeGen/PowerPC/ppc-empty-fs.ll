; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; This guarantees that we add the default set of features to the current feature
; string. We won't successfully legalize the types here without +64bit being
; silently added.
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.fab = type { float, float }

; Function Attrs: nounwind
define void @func_fab(ptr noalias sret(%struct.fab) %agg.result, i64 %x.coerce) #0 {
entry:
  %x = alloca %struct.fab, align 8
  store i64 %x.coerce, ptr %x, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.result, ptr align 4 %x, i64 8, i1 false)
  ret void
}

; CHECK: func_fab

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "target-features"="" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 233227) (llvm/trunk 233226)"}
