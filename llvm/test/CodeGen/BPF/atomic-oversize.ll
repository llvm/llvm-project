; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: unsupported atomic store
; CHECK: error: unsupported atomic load

define void @test(ptr %a) nounwind {
  %1 = load atomic i128, ptr %a monotonic, align 16
  store atomic i128 %1, ptr %a monotonic, align 16
  ret void
}
