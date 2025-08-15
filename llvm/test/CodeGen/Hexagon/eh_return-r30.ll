; RUN: llc -mtriple=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: noreturn nounwind
define void @f0(i32 %a0, ptr %a1) #0 {
b0:
  tail call void @llvm.eh.return.i32(i32 %a0, ptr %a1)
  unreachable
}

; Function Attrs: nounwind
declare void @llvm.eh.return.i32(i32, ptr) #1

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind }
