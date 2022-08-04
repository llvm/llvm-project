; Intrinsic calls can't be uniformly replaced with undef without invalidating
; IR (eg: only intrinsic calls can have metadata arguments), so ensure they are
; not replaced. The whole call instruction can be removed by instruction
; reduction instead.

; RUN: llvm-reduce --delta-passes=functions,instructions --test FileCheck --test-arg --check-prefixes=ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2> %t.log
; RUN: FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-FINAL %s < %t

; Check that the call is removed by instruction reduction passes
; RUN: llvm-reduce --delta-passes=functions,instructions --test FileCheck --test-arg --check-prefix=ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-NOCALL %s < %t


declare ptr @llvm.sponentry.p0()
declare i8 @uninteresting()

; ALL-LABEL: define ptr @interesting(
define ptr @interesting() {
entry:
  ; CHECK-INTERESTINGNESS: call ptr
  ; CHECK-NOCALL-NOT: call i8

  ; CHECK-FINAL: %call = call ptr @llvm.sponentry.p0()
  ; CHECK-FINAL-NEXT: ret ptr %call
  %call = call ptr @llvm.sponentry.p0()
  call i8 @uninteresting()
  ret ptr %call
}
