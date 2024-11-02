; RUN: llvm-reduce -write-tmp-files-as-bitcode --delta-passes=function-bodies,basic-blocks %s -o %t \
; RUN:     --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py --test-arg llvm-dis --test-arg FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: @callee(
; CHECK-FINAL: declare void @callee()
define void @callee() {
  ret void
}

; CHECK-ALL: define void @caller()
define void @caller() {
entry:
; CHECK-ALL: call void @callee()
; CHECK-ALL: ret void
  call void @callee()
  ret void
}
