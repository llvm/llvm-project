; RUN: llvm-reduce --test FileCheck --test-arg --check-prefix=CHECK-ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-ALL --implicit-check-not=uninteresting %s < %t

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
