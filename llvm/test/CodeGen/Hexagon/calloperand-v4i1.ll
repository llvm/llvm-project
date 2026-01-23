;RUN: llc -mtriple=hexagon  < %s -o - | FileCheck %s --check-prefix=CHECK
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length64b < %s -o - | FileCheck %s --check-prefix=CHECK
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-LABEL: compare_vectors
; CHECK: [[REG0:(p[0-9]+)]] = vcmph.eq([[REG1:(r[0-9]+):[0-9]]],[[REG2:(r[0-9]+):[0-9]]])
; CHECK: [[REG1:(r[0-9]+):[0-9]]] = CONST64(#281479271743489)
; CHECK: [[REG2:(r[0-9]+):[0-9]]] = mask([[REG0]])
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = and([[REG2]],[[REG1]])

define void @compare_vectors(<4 x i16> %a, <4 x i16> %b) {
entry:
  %result = icmp eq <4 x i16> %a, %b
  call i32 @f.1(<4 x i1> %result)
  ret void
}
; CHECK-LABEL: f.1:
; CHECK: [[REG3:(r[0-9]+)]] = and([[REG3]],##65537)
; CHECK: [[REG4:(r[0-9]+)]] = and([[REG4]],##65537)
define i32 @f.1(<4 x i1> %vec) {
  %element = extractelement <4 x i1> %vec, i32 2
  %is_true = icmp eq i1 %element, true
  br i1 %is_true, label %if_true, label %if_false

if_true:
  call void @action_if_true()
  br label %end

if_false:
  call void @action_if_false()
  br label %end

end:
  %result = phi i32 [1, %if_true], [0, %if_false]
  ret i32 %result
}

declare void @action_if_true()
declare void @action_if_false()
