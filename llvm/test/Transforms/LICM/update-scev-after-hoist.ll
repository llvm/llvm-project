; RUN: opt -S -passes='loop-unroll,loop-mssa(licm),print<scalar-evolution>' -unroll-count=4 -disable-output < %s 2>&1 | FileCheck %s --check-prefix=SCEV-EXPR

define i16 @main() {
; SCEV-EXPR:      Classifying expressions for: @main
; SCEV-EXPR-NEXT:  %mul = phi i16 [ 1, %entry ], [ %mul.n.3, %loop ]
; SCEV-EXPR-NEXT:  -->  %mul U: [0,-15) S: [-32768,32753)		Exits: 4096		LoopDispositions: { %loop: Variant }
; SCEV-EXPR-NEXT:  %div = phi i16 [ 32767, %entry ], [ %div.n.3, %loop ]
; SCEV-EXPR-NEXT:  -->  %div U: [-2048,-32768) S: [-2048,-32768)		Exits: 7		LoopDispositions: { %loop: Variant }
; SCEV-EXPR-NEXT:  %mul.n.reass.reass = mul i16 %mul, 8
; SCEV-EXPR-NEXT:  -->  (8 * %mul) U: [0,-7) S: [-32768,32761)		Exits: -32768		LoopDispositions: { %loop: Variant }
entry:
  br label %loop

loop:
  %mul = phi i16 [ 1, %entry ], [ %mul.n, %loop ]
  %div = phi i16 [ 32767, %entry ], [ %div.n, %loop ]
  %mul.n = mul i16 %mul, 2
  %div.n = sdiv i16 %div, 2
  %cmp = icmp sgt i16 %div, 0
  br i1 %cmp, label %loop, label %end

end:
  ret i16 %mul
}
