; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that we generate 'cmph.gtu' instruction.
; CHECK-LABEL: @cmphgtu
; CHECK: cmph.gtu

@glob = common global i8 0, align 1

define i32 @cmphgtu(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = xor i32 %a1, %a0
  %v4 = and i32 %v3, 65535
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

; With zxtb, we must not generate a cmph.gtu instruction.
; CHECK-LABEL: @nocmphgtu
; CHECK-NOT: cmph.gtu
define i32 @nocmphgtu(i32 %a0, i32 %a1) #0 {
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
