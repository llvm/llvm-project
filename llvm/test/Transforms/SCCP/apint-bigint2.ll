; RUN: opt < %s -sccp -S | FileCheck %s

@Y = constant [6 x i101] [ i101 12, i101 123456789000000, i101 -12,
                           i101 -123456789000000, i101 0,i101 9123456789000000]

; CHECK-LABEL: @array
; CHECK-NEXT: ret i101 123456789000000
define i101 @array() {
   %A = getelementptr [6 x i101], [6 x i101]* @Y, i32 0, i32 1
   %B = load i101, i101* %A
   %D = and i101 %B, 1
   %DD = or i101 %D, 1
   %E = trunc i101 %DD to i32
   %F = getelementptr [6 x i101], [6 x i101]* @Y, i32 0, i32 %E
   %G = load i101, i101* %F
 
   ret i101 %G
}

; CHECK-LABEL: @large_aggregate
; CHECK-NEXT: ret i101 undef
define i101 @large_aggregate() {
  %B = load i101, i101* undef
  %D = and i101 %B, 1
  %DD = or i101 %D, 1
  %F = getelementptr [6 x i101], [6 x i101]* @Y, i32 0, i32 5
  %G = getelementptr i101, i101* %F, i101 %DD
  %L3 = load i101, i101* %G
  ret i101 %L3
}
