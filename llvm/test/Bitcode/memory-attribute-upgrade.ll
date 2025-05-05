; RUN: llvm-dis < %S/Inputs/memory-attribute-upgrade.bc | FileCheck %s

; CHECK: ; Function Attrs: memory(write, argmem: read)
; CHECK-NEXT: define void @test_any_write_argmem_read(ptr %p)

; CHECK: ; Function Attrs: memory(read, argmem: readwrite, inaccessiblemem: none)
; CHECK-NEXT: define void @test_any_read_argmem_readwrite(ptr %p)
