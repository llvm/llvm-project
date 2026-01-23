; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: lsr{{.*}}#31

target datalayout = "e-m:e-p:32:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind readnone
define i64 @foo(i64 %x) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S2.asr.i.p(i64 %x, i32 32)
  ret i64 %0
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32) #1

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR (based on LLVM 3.7.0)"}
