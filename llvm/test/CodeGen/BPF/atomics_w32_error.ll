; RUN: not llc -mtriple=bpf -mcpu=v1 < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: unsupported atomic operation, please use 64 bit version

define i32 @test(ptr %p) {
  %val = atomicrmw and ptr %p, i32 1 seq_cst
  ret i32 %val
}
