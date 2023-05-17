; RUN: llvm-reduce --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS: define void @func(
; CHECK-FINAL: define void @func()
define void @func(i1 %arg) {
; CHECK-ALL: entry:
; CHECK-INTERESTINGNESS: call void @foo({{.*}}blockaddress
; CHECK-FINAL: call void @foo(ptr blockaddress(@func, %bb5))
entry:
  call void @foo(ptr blockaddress(@func, %bb5))
  ret void

; CHECK-ALL: bb5:
; CHECK-ALL: ret void
bb5:
  ret void
}

declare void @foo(ptr)
