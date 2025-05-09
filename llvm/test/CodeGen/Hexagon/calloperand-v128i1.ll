;RUN: llc -mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s -o - | FileCheck %s

; CHECK-LABEL: compare_vectors
; CHECK: [[REG1:(q[0-9]+)]] = vcmp.eq(v{{[0-9]+}}.b,v{{[0-9]+}}.b)
; CHECK: [[REG2:(r[0-9]+)]] = #-1
; CHECK: v0 = vand([[REG1]],[[REG2]])

define void @compare_vectors(<128 x i8> %a, <128 x i8> %b) {
entry:
  %result = icmp eq <128 x i8> %a, %b
  call i32 @f.1(<128 x i1> %result)
  ret void
}

; CHECK-LABEL: f.1:
; CHECK: [[REG3:(q[0-9]+)]] = vand(v0,r{{[0-9]+}})
; CHECK: [[REG4:(v[0-9]+)]] = vand([[REG3]],r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = vextract([[REG4]],r{{[0-9]+}})

define i32 @f.1(<128 x i1> %vec) {
  %element = extractelement <128 x i1> %vec, i32 6
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
