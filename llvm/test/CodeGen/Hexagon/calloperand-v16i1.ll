;RUN: llc -mtriple=hexagon  < %s -o - | FileCheck %s --check-prefix=CHECK
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length64b < %s -o - | FileCheck %s --check-prefix=CHECK-HVX
;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s -o - | FileCheck %s --check-prefix=CHECK-HVX

; CHECK-LABEL: compare_vectors
; CHECK: [[REG5:(r[0-9]+):[0-9]]] = CONST64(#72340172838076673)
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = and(r{{[0-9]+}}:{{[0-9]+}},[[REG5]])
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = and(r{{[0-9]+}}:{{[0-9]+}},[[REG5]])

; CHECK-HVX: [[REG1:(q[0-9]+)]] = vcmp.eq(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-HVX: [[REG2:(r[0-9]+)]] = #-1
; CHECK-HVX: v0  = vand([[REG1]],[[REG2]])

define void @compare_vectors(<16 x i32> %a, <16 x i32> %b) {
entry:
  %result = icmp eq <16 x i32> %a, %b
  call i32 @f.1(<16 x i1> %result)
  ret void
}

; CHECK-LABEL: f.1:
; CHECK: [[REG3:(r[0-9]+)]] = and([[REG3]],##16843009)
; CHECK: [[REG4:(r[0-9]+)]] = and([[REG4]],##16843009)
; CHECK-HVX: [[REG3:(q[0-9]+)]] = vand(v0,r{{[0-9]+}})
; CHECK-HVX: [[REG4:(v[0-9]+)]] = vand([[REG3]],r{{[0-9]+}})
; CHECK-HVX: r{{[0-9]+}} = vextract([[REG4]],r{{[0-9]+}})

define i32 @f.1(<16 x i1> %vec) {
  %element = extractelement <16 x i1> %vec, i32 6
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
