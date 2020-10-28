; Intrinsic calls can't be uniformly replaced with undef without invalidating
; IR (eg: only intrinsic calls can have metadata arguments), so ensure they are
; not replaced. The whole call instruction can be removed by instruction
; reduction instead.

; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2> %t.log
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-FINAL %s

; Check that the call is removed by instruction reduction passes
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=ALL,CHECK-FUNC --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-NOCALL %s


declare i8* @llvm.sponentry.p0i8()
declare void @uninteresting()

; ALL-LABEL: define i8* @interesting(
define i8* @interesting() {
entry:
  ; CHECK-INTERESTINGNESS: call
  ; CHECK-NOCALL-NOT: call

  ; CHECK-FINAL: %call = call i8* @llvm.sponentry.p0i8()
  ; CHECK-FINAL-NEXT: ret i8* %call
  %call = call i8* @llvm.sponentry.p0i8()
  call void @uninteresting()
  ret i8* %call
}
