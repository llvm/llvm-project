; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL --implicit-check-not=uninteresting %s

declare void @llvm.uninteresting()
declare void @uninteresting()

; CHECK-ALL: declare void @llvm.interesting()
; CHECK-ALL: declare void @interesting()
declare void @llvm.interesting()
declare void @interesting()

; CHECK-ALL: define void @main() {
; CHECK-ALL-NEXT:  call void @llvm.interesting()
; CHECK-ALL-NEXT:   call void @interesting()
; CHECK-ALL-NEXT:   ret void
; CHECK-ALL-NEXT: }
define void @main() {
  call void @llvm.interesting()
  call void @interesting()
  ret void
}
