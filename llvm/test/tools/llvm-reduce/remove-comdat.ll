; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-objects --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s --input-file=%t

; CHECK-INTERESTINGNESS: define void @f
; CHECK-INTERESTINGNESS: define void @g
; CHECK-INTERESTINGNESS: define void @i{{.*}} comdat

; CHECK-FINAL-NOT: $f
; CHECK-FINAL-NOT: $h
; CHECK-FINAL: $i = comdat
; CHECK-FINAL: define void @f() {
; CHECK-FINAL: define void @g() {
; CHECK-FINAL: define void @i() comdat {

$f = comdat any
$h = comdat any
$i = comdat any

define void @f() comdat {
  ret void
}

define void @g() comdat($h) {
  ret void
}

define void @i() comdat {
  ret void
}
