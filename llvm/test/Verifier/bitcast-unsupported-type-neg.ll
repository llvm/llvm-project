; RUN: split-file %s %t
; RUN: not llvm-as -disable-output %t/token.ll 2>&1 | FileCheck %t/token.ll
; RUN: not llvm-as -disable-output %t/label.ll 2>&1 | FileCheck %t/label.ll
; RUN: not llvm-as -disable-output %t/metadata.ll 2>&1 | FileCheck %t/metadata.ll
; RUN: not llvm-as -disable-output %t/target-ext.ll 2>&1 | FileCheck %t/target-ext.ll
; RUN: not llvm-as -disable-output %t/vector-of-target-ext.ll 2>&1 | FileCheck %t/vector-of-target-ext.ll

;--- token.ll
; CHECK: invalid cast opcode for cast from 'token' to 'token'
define void @f(token %t) {
  %x = bitcast token %t to token
  ret void
}

;--- label.ll
; CHECK: invalid cast opcode for cast from 'label' to 'label'
define void @f() {
entry:
  %x = bitcast label %entry to label
  ret void
}

;--- metadata.ll
; CHECK: invalid cast opcode for cast from 'metadata' to 'metadata'
define void @f(metadata %m) {
  %x = bitcast metadata %m to metadata
  ret void
}

;--- target-ext.ll
; CHECK: invalid cast opcode for cast from 'target("foo")' to 'target("foo")'
define void @f(target("foo") %x) {
  %y = bitcast target("foo") %x to target("foo")
  ret void
}

;--- vector-of-target-ext.ll
; CHECK: invalid cast opcode for cast from '<2 x target("llvm.test.vectorelement")>' to '<2 x target("llvm.test.vectorelement")>'
define void @f(<2 x target("llvm.test.vectorelement")> %x) {
  %y = bitcast <2 x target("llvm.test.vectorelement")> %x to <2 x target("llvm.test.vectorelement")>
  ret void
}
