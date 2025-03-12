; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that the elements of the constants have correct type.
; CHECK: .half 31

define void @fred(ptr %p) #0 {
  store <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19, i16 20, i16 21, i16 22, i16 23, i16 24, i16 25, i16 26, i16 27, i16 28, i16 29, i16 30, i16 31>, ptr %p, align 64
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

