; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: unsupported atomic operation, please use 32/64 bit version

define i8 @test(ptr %p) {
  %val = atomicrmw add ptr %p, i8 1 seq_cst
  ret i8 %val
}
