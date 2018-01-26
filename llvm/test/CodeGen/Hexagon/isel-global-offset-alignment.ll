; RUN: llc -march=hexagon < %s | FileCheck %s

; This testcase checks that a valid offset is folded into a global address
; when it's used in a load or a store instruction. The store in this code
; is not really properly aligned (bugpoint output from a legal code), but
; what's important is that the offset on the store instructions is a multiple
; of the access size. In this case the actual address is @array+30, but that
; value is not a multiple of 8, so it cannot appear as an immediate in memd.
; Aside from the fact that @array+30 is not a valid address for memd, make
; sure that in a memd instruction the offset field is a multiple of 8.

; CHECK: r[[BASE:[0-9]+]] = #6
; CHECK-DAG: memd(r[[BASE]]+##array+24)
; CHECK-DAG: memd(r[[BASE]]+##array+32)


target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@array = external global [1000000 x i16], align 8

define void @fred() #0 {
b0:
  %v1 = add nsw i32 0, -1
  %v2 = getelementptr inbounds [1000000 x i16], [1000000 x i16]* @array, i32 0, i32 %v1
  %v3 = getelementptr i16, i16* %v2, i32 16
  %v4 = bitcast i16* %v3 to <8 x i16>*
  store <8 x i16> zeroinitializer, <8 x i16>* %v4, align 8
  ret void
}

attributes #0 = { norecurse nounwind }
