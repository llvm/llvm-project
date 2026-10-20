; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test coverage for HexagonAsmPrinter: exercise the constant pool index
; and global address operand printing paths.

@global_arr = external global [100 x i32]

; CHECK-LABEL: test_constpool:
; CHECK: ##
define float @test_constpool(float %x) {
entry:
  %add = fadd float %x, 0x400921FB60000000
  ret float %add
}

; CHECK-LABEL: test_global_addr:
; CHECK: ##global_arr
define ptr @test_global_addr(i32 %idx) {
entry:
  %gep = getelementptr [100 x i32], ptr @global_arr, i32 0, i32 %idx
  ret ptr %gep
}

; Exercise the block address path.
; CHECK-LABEL: test_blockaddr:
; CHECK: ##.Ltmp
define ptr @test_blockaddr() {
entry:
  br label %target
target:
  ret ptr blockaddress(@test_blockaddr, %target)
}

