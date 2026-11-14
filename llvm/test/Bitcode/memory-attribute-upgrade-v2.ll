; RUN: llvm-dis < %S/Inputs/memory-attribute-upgrade-v2.bc | FileCheck %s

; The input was produced by llvm-as from this file, at a revision that wrote
; version 2 of the memory attribute encoding, which did not have the threadid
; location and placed the target memory locations right after other memory.
; The threadid location takes the access kind of other memory, and the target
; memory locations must not be shifted into it.

target triple = "aarch64-unknown-linux-gnu"

; CHECK: ; Function Attrs: memory(target_mem0: read)
; CHECK-NEXT: define void @test_target_mem0_read()
define void @test_target_mem0_read() #0 {
  ret void
}

; CHECK: ; Function Attrs: memory(read, target_mem1: write)
; CHECK-NEXT: define void @test_read_target_mem1_write()
define void @test_read_target_mem1_write() #1 {
  ret void
}

; CHECK: ; Function Attrs: memory(readwrite, argmem: none, inaccessiblemem: read, errnomem: write, target_mem0: none, target_mem1: read)
; CHECK-NEXT: define void @test_mixed()
define void @test_mixed() #2 {
  ret void
}

; CHECK: ; Function Attrs: memory(argmem: readwrite, target_mem: read)
; CHECK-NEXT: define void @test_argmem_readwrite_target_mem_read()
define void @test_argmem_readwrite_target_mem_read() #3 {
  ret void
}

; CHECK: ; Function Attrs: memory(none)
; CHECK-NEXT: define void @test_none()
define void @test_none() #4 {
  ret void
}

; CHECK: ; Function Attrs: memory(readwrite)
; CHECK-NEXT: define void @test_readwrite()
define void @test_readwrite() #5 {
  ret void
}

attributes #0 = { memory(target_mem0: read) }
attributes #1 = { memory(read, target_mem1: write) }
attributes #2 = { memory(readwrite, argmem: none, inaccessiblemem: read, errnomem: write, target_mem0: none, target_mem1: read) }
attributes #3 = { memory(argmem: readwrite, target_mem: read) }
attributes #4 = { memory(none) }
attributes #5 = { memory(readwrite) }
