; XFAIL: *
; Implement generic selection of a constant.

; RUN: llc -O2 -march=hexagon < %s | FileCheck %s --check-prefix=CHECK-TEST1
; RUN: llc -O2 -march=hexagon < %s | FileCheck %s --check-prefix=CHECK-TEST2
; RUN: llc -O2 -march=hexagon < %s | FileCheck %s --check-prefix=CHECK-TEST3
define i32 @main() #0 {
entry:
  %l = alloca [7 x i32], align 8
  %p_arrayidx45 = bitcast [7 x i32]* %l to i32*
  %vector_ptr = bitcast [7 x i32]* %l to <2 x i32>*
  store <2 x i32> <i32 3, i32 -2>, <2 x i32>* %vector_ptr, align 8
  %p_arrayidx.1 = getelementptr [7 x i32], [7 x i32]* %l, i32 0, i32 2
  %vector_ptr.1 = bitcast i32* %p_arrayidx.1 to <2 x i32>*
  store <2 x i32> <i32 -4, i32 6>, <2 x i32>* %vector_ptr.1, align 8
  %p_arrayidx.2 = getelementptr [7 x i32], [7 x i32]* %l, i32 0, i32 4
  %vector_ptr.2 = bitcast i32* %p_arrayidx.2 to <2 x i32>*
  store <2 x i32> <i32 -8, i32 -10>, <2 x i32>* %vector_ptr.2, align 8
  ret i32 0
}

; The instructions seem to be in a different order in the .s file than
; the corresponding values in the .ll file, so just run the test three
; times and each time test for a different instruction.
; CHECK-TEST1: combine(#-2,#3)
; CHECK-TEST2: combine(#6,#-4)
; CHECK-TEST3: combine(#-10,#-8)

attributes #0 = { "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

