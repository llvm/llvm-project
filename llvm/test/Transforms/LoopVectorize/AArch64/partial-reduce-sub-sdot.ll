; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize                                                \
; RUN:     -scalable-vectorization=on -mattr=+sve2                               \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize       \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=COMMON,SVE

; RUN: opt -passes=loop-vectorize                                                \
; RUN:     -scalable-vectorization=off -mattr=+neon,+dotprod                     \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize       \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=COMMON,NEON

; COMMON: LV: Checking a loop in 'sub_reduction'
; SVE:  Cost of 1 for VF vscale x 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (mul (ir<%load1> sext to i32), (ir<%load2> sext to i32))
; NEON: Cost of 1 for VF 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (mul (ir<%load1> sext to i32), (ir<%load2> sext to i32))

; COMMON: LV: Checking a loop in 'add_sub_chained_reduction'
; SVE:  Cost of 1 for VF vscale x 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (mul (ir<%load1> sext to i32), (ir<%load2> sext to i32))
; SVE:  Cost of 9 for VF vscale x 16: EXPRESSION vp<{{.*}}> = vp<%9> + partial.reduce.add (sub (0, mul (ir<%load2> sext to i32), (ir<%load3> sext to i32)))
; NEON: Cost of 1 for VF 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (mul (ir<%load1> sext to i32), (ir<%load2> sext to i32))
; NEON: Cost of 9 for VF 16: EXPRESSION vp<{{.*}}> = vp<%9> + partial.reduce.add (sub (0, mul (ir<%load2> sext to i32), (ir<%load3> sext to i32)))

target triple = "aarch64"

; Test the cost of a SUB reduction, where the SUB is implemented outside the loop
; and therefore not part of the partial reduction.
define i32 @sub_reduction(ptr %arr1, ptr %arr2, i32 %init, i32 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ %init, %entry ], [ %sub, %loop ]
  %gep1 = getelementptr inbounds i8, ptr %arr1, i32 %iv
  %load1 = load i8, ptr %gep1
  %sext1 = sext i8 %load1 to i32
  %gep2 = getelementptr inbounds i8, ptr %arr2, i32 %iv
  %load2 = load i8, ptr %gep2
  %sext2 = sext i8 %load2 to i32
  %mul = mul i32 %sext1, %sext2
  %sub = sub i32 %acc, %mul
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret i32 %sub
}

; Test that the cost of a SUB that is part of an ADD-SUB reduction chain
; is high, because the negation happens inside the loop and cannot be
; folded into the SDOT instruction (because of the extend).
define i32 @add_sub_chained_reduction(ptr %arr1, ptr %arr2, ptr %arr3, i32 %init, i32 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ %init, %entry ], [ %sub, %loop ]
  %gep1 = getelementptr inbounds i8, ptr %arr1, i32 %iv
  %load1 = load i8, ptr %gep1
  %sext1 = sext i8 %load1 to i32
  %gep2 = getelementptr inbounds i8, ptr %arr2, i32 %iv
  %load2 = load i8, ptr %gep2
  %sext2 = sext i8 %load2 to i32
  %mul1 = mul i32 %sext1, %sext2
  %add = add i32 %acc, %mul1
  %gep3 = getelementptr inbounds i8, ptr %arr3, i32 %iv
  %load3 = load i8, ptr %gep3
  %sext3 = sext i8 %load3 to i32
  %mul2 = mul i32 %sext2, %sext3
  %sub = sub i32 %add, %mul2
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !0

exit:
  ret i32 %sub
}

attributes #0 = { vscale_range(1,16) }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.interleave.count", i32 1}
!2 = !{!"llvm.loop.vectorize.width", i32 16}
