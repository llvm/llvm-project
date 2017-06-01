; RUN: llc -mtriple arm-linux-gnueabi -filetype asm -o - %s | FileCheck %s

declare i8* @llvm.thread.pointer()

define i8* @test() {
entry:
  %tmp1 = call i8* @llvm.thread.pointer()
  ret i8* %tmp1
}

; CHECK: bl __aeabi_read_tp

