; RUN: opt --thinlto-bc --thinlto-split-lto-unit %s -o %t0
; RUN: llvm-reduce -write-tmp-files-as-bitcode --delta-passes=function-bodies,basic-blocks %t0 -o %t1 \
; RUN:     --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py --test-arg llvm-dis --test-arg FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s
; RUN: llvm-dis < %t1* | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

@g = internal global i8 42, !type !0

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
define ptr @f() {
  ; CHECK-ALL: ret ptr @g.{{([0-9a-f]{32})}}
  ret ptr @g
}

!0 = !{i32 0, !"typeid"}
