; It's invalid to have a non-intrinsic call with a metadata argument,
; so it's unproductive to replace the users of the intrinsic
; declaration with undef.

; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2> %t.log
; RUN: FileCheck -check-prefix=STDERR %s < %t.log
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-FINAL %s

; Check that the call is removed by instruction reduction passes
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=ALL,CHECK-FUNC --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck -implicit-check-not=uninteresting --check-prefixes=ALL,CHECK-NOCALL %s


; STDERR-NOT: Function has metadata parameter but isn't an intrinsic

declare i32 @llvm.amdgcn.reloc.constant(metadata)
declare void @uninteresting()

; ALL-LABEL: define i32 @interesting(
define i32 @interesting() {
entry:
  ; CHECK-INTERESTINGNESS: call
  ; CHECK-NOCALL-NOT: call

  ; CHECK-FINAL: %call = call i32 @llvm.amdgcn.reloc.constant(metadata !"arst")
  ; CHECK-FINAL-NEXT: ret i32 %call
  %call = call i32 @llvm.amdgcn.reloc.constant(metadata !"arst")
  call void @uninteresting()
  ret i32 %call
}
