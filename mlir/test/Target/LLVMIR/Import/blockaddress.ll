; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

@g = private global ptr blockaddress(@fn, %bb1)
define void @fn() {
  br label %bb1
bb1:
  ret void
}

; CHECK:  llvm.mlir.global private @g()
; CHECK:     %[[ADDR:.*]] = llvm.blockaddress <function = @fn, tag = <id = [[ID:.*]]>> : !llvm.ptr
; CHECK:     llvm.return %[[ADDR]] : !llvm.ptr

; CHECK:   llvm.func @fn() {
; CHECK:     llvm.br ^[[RET_BB:.*]]
; CHECK:   ^[[RET_BB]]:
; CHECK:     llvm.blocktag <id = [[ID]]>
; CHECK:     llvm.return
; CHECK:   }

; // -----

; CHECK-LABEL: blockaddr0
define ptr @blockaddr0() {
  br label %bb1
  ; CHECK: %[[BLOCKADDR:.*]] = llvm.blockaddress <function = @blockaddr0, tag = <id = [[BLOCK_ID:.*]]>> : !llvm.ptr
  ; CHECK: llvm.br ^[[BB1:.*]]
bb1:
  ; CHECK: ^[[BB1]]:
  ; CHECK: llvm.blocktag <id = [[BLOCK_ID]]>
  ; CHECK-NEXT: llvm.return %[[BLOCKADDR]] : !llvm.ptr
  ret ptr blockaddress(@blockaddr0, %bb1)
}
