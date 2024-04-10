; RUN: llc < %s -mtriple=x86_64 -mattr=avx512bw | FileCheck %s

; Verify that the DAGCombiner doesn't change build_vector to concat_vectors if
; the vector element type is different than splat type. The example here:
;   v8i1 = build_vector (i8 (bitcast (v8i1 X))), ..., (i8 (bitcast (v8i1 X))))

; CHECK:      foo:
; CHECK:      # %bb.0: # %entry
; CHECK-NEXT: retq

define void @foo(<8 x i1> %mask.i1) {
entry:
  %0 = and <8 x i1> %mask.i1, <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>
  %1 = bitcast <8 x i1> %0 to i8
  %2 = icmp ne i8 %1, 0
  %insert54 = insertelement <8 x i1> zeroinitializer, i1 %2, i64 0
  %splat55 = shufflevector <8 x i1> %insert54, <8 x i1> zeroinitializer, <8 x i32> zeroinitializer
  %3 = and <8 x i1> %0, %splat55
  br label %end

end:                           ; preds = %entry
  %4 = select <8 x i1> %3, <8 x i1> zeroinitializer, <8 x i1> zeroinitializer
  ret void
}
