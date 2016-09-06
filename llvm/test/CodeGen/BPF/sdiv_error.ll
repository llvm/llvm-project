; RUN: not llc -march=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: Unsupport signed division

; Function Attrs: norecurse nounwind readnone
define i32 @test(i32 %len) #0 {
  %1 = srem i32 %len, 15
  ret i32 %1
}
