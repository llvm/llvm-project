;RUN: llc -mtriple=hexagon  < %s -o - | FileCheck %s --check-prefix=CHECK
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length64b < %s -o - | FileCheck %s --check-prefix=CHECK
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s -o - | FileCheck %s --check-prefix=CHECK

; CHECK-LABEL: compare_vectors
; CHECK: [[REG0:(p[0-9]+)]] = vcmpb.eq([[REG1:(r[0-9]+):[0-9]]],[[REG2:(r[0-9]+):[0-9]]])
; CHECK: [[REG1:(r[0-9]+):[0-9]]] = CONST64(#72340172838076673)
; CHECK: [[REG2:(r[0-9]+):[0-9]]] = mask([[REG0]])
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = and([[REG2]],[[REG1]])

define void @compare_vectors(<8 x i8> %a, <8 x i8> %b) {
entry:
  %result = icmp eq <8 x i8> %a, %b
  call i32 @f.1(<8 x i1> %result)
  ret void
}
; CHECK-LABEL: f.1:
; CHECK: [[REG3:(r[0-9]+)]] = and([[REG3]],##16843009)
; CHECK: [[REG4:(r[0-9]+)]] = and([[REG4]],##16843009)
define i32 @f.1(<8 x i1> %vec) {
  %element = extractelement <8 x i1> %vec, i32 6
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
