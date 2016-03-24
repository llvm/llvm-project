; RUN: llvm-link %s %p/Inputs/constructor-comdat.ll -S -o - 2>&1 | FileCheck %s
; RUN: llvm-link %p/Inputs/constructor-comdat.ll %s -S -o - 2>&1 | FileCheck --check-prefix=NOCOMDAT %s

$_ZN3fooIiEC5Ev = comdat any
; CHECK: $_ZN3fooIiEC5Ev = comdat any
; NOCOMDAT-NOT: comdat

@_ZN3fooIiEC1Ev = weak_odr alias void (), void ()* @_ZN3fooIiEC2Ev
; CHECK: @_ZN3fooIiEC1Ev = weak_odr alias void (), void ()* @_ZN3fooIiEC2Ev
; NOCOMDAT-DAG: define weak_odr void @_ZN3fooIiEC1Ev() {

; CHECK: define weak_odr void @_ZN3fooIiEC2Ev() comdat($_ZN3fooIiEC5Ev) {
; NOCOMDAT-DAG: define weak_odr void @_ZN3fooIiEC2Ev() {
define weak_odr void @_ZN3fooIiEC2Ev() comdat($_ZN3fooIiEC5Ev) {
  ret void
}
