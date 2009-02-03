; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test forward references and redefinitions of globals

@A = global i17* @B
@B = global i17 7

declare void @X()

declare void @X()

define void @X() {
  ret void
}

declare void @X()
