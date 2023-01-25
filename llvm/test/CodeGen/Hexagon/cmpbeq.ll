; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate 'cmpb.eq' instruction for a byte comparision.

@g0 = common global i8 0, align 1

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = xor i32 %a1, %a0
  %v1 = and i32 %v0, 255
  %v2 = icmp eq i32 %v1, 0
  br i1 %v2, label %b1, label %b2
; CHECK-NOT: xor(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-NOT: zxtb(r{{[0-9]+}})
; CHECK: cmpb.eq(r{{[0-9]+}},r{{[0-9]+}})

b1:                                               ; preds = %b0
  %v3 = trunc i32 %a0 to i8
  store i8 %v3, ptr @g0, align 1
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v4 = phi i32 [ 1, %b1 ], [ 0, %b0 ]
  ret i32 %v4
}

attributes #0 = { nounwind }
