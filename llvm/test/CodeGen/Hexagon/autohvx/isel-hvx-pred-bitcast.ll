; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: danny:
; CHECK: vrmpy
define i64 @danny(<64 x i8> %a0, <64 x i8> %a1) #0 {
  %v0 = icmp eq <64 x i8> %a0, %a1
  %v1 = bitcast <64 x i1> %v0 to i64
  ret i64 %v1
}

; CHECK-LABEL: sammy:
; CHECK: vrmpy
define i32 @sammy(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %v0 = icmp eq <32 x i16> %a0, %a1
  %v1 = bitcast <32 x i1> %v0 to i32
  ret i32 %v1
}

; CHECK-LABEL: kirby:
; CHECK: vrmpy
define i16 @kirby(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %v0 = icmp eq <16 x i32> %a0, %a1
  %v1 = bitcast <16 x i1> %v0 to i16
  ret i16 %v1
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+v66,+hvx,+hvxv66,+hvx-length64b" }
