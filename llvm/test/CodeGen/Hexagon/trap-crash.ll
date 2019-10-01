; RUN: llc -march=hexagon < %s | FileCheck %s

; Generate code that is guaranteed to crash. At the moment, it's a
; misaligned load.
; CHECK: memd(##3134984174)

target triple = "hexagon"

; Function Attrs: noreturn nounwind
define i32 @f0() #0 {
entry:
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { noreturn nounwind "target-cpu"="hexagonv60" }
attributes #1 = { cold noreturn nounwind }
