; The code in InstCombiner::FoldSelectOpOp was calling
; Type::getVectorNumElements without checking first if the type was a vector.

; RUN: opt < %s -instcombine -S

define i32 @vselect1(i32 %a.coerce, i32 %b.coerce, i32 %c.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %2 = bitcast i32 %c.coerce to <2 x i16>
  %cmp = icmp sge <2 x i16> %2, zeroinitializer
  %or = select <2 x i1> %cmp, <2 x i16> %0, <2 x i16> %1
  %3 = bitcast <2 x i16> %or to i32
  ret i32 %3
}
