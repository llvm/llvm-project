; RUN: llc -verify-machineinstrs -mtriple=mipsel-linux-gnu -mcpu=mips32r2 %s -o - | FileCheck %s

define void @fake_use(ptr %aaaa) {
; CHECK-LABEL: fake_use:
; CHECK:       # %bb.0:    # %entry
; CHECK-NEXT:              # fake_use: $a0 
; CHECK-NEXT:    jr $ra 
; CHECK-NEXT:    nop
entry:
  notail call void (...) @llvm.fake.use(ptr %aaaa)
  ret void
}

declare void @llvm.fake.use(...)

