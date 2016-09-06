; RUN: llvm-link %s -S -o - | FileCheck %s

$c = comdat any
@a = alias void (), void ()* @f
define internal void @f() comdat($c) {
  ret void
}

; CHECK-DAG: $c = comdat any
; CHECK-DAG: @a = alias void (), void ()* @f
; CHECK-DAG: define internal void @f() comdat($c)

$f2 = comdat largest
define linkonce_odr void @f2() comdat($f2) {
  ret void
}
define void @f3() comdat($f2) {
  ret void
}

; CHECK-DAG: $f2 = comdat largest
; CHECK-DAG: define linkonce_odr void @f2()
