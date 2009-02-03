; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test forward references and redefinitions of globals

@A = global i32* @B             ; <i32**> [#uses=0]
@B = global i32 7               ; <i32*> [#uses=1]

declare void @X()

declare void @X()

define void @X() {
  ret void
}

declare void @X()
