; RUN: llc -march=hexagon -debug-only=isel < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that we generate 'cmpb.gtu' instruction for a byte comparision
; The "Optimized Lowered Selection" converts the "ugt with #40" to
; "ult with #41". The immediate value should be decremented to #40
; with the selected cmpb.gtu pattern
; CHECK: setcc{{.*}}41{{.*}}setult
; CHECK: A4_cmpbgtui{{.*}}40

@glob = common global i8 0, align 1

define i32 @cmpgtudec(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = xor i32 %a1, %a0
  %v4 = and i32 %v3, 255
  %v5 = icmp ugt i32 %v4, 40
  br i1 %v5, label %b6, label %b8

b6:                                               ; preds = %b2
  %v7 = trunc i32 %a0 to i8
  store i8 %v7, ptr @glob, align 1
  br label %b8

b8:                                               ; preds = %b6, %b2
  %v9 = phi i32 [ 1, %b6 ], [ 0, %b2 ]
  ret i32 %v9
}

attributes #0 = { nounwind }
