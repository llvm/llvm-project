; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize         \
; RUN:     -mattr=+sve -scalable-vectorization=off                                 \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-FIXED-BASE
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize         \
; RUN:     -mattr=+sve2p1 -scalable-vectorization=off                              \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-FIXED
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize         \
; RUN:     -mattr=+sve2p1 -scalable-vectorization=on                               \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=8 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize         \
; RUN:     -mattr=+sve,+sme2 -scalable-vectorization=on                            \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE

; LV: Checking a loop in 'sext_reduction_i16_to_i32'
; CHECK-FIXED-BASE: Cost of 3 for VF 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> sext to i32)
; CHECK-FIXED: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> sext to i32)
; CHECK-SCALABLE: Cost of 1 for VF vscale x 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> sext to i32)

; LV: Checking a loop in 'zext_reduction_i16_to_i32'
; CHECK-FIXED-BASE: Cost of 3 for VF 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> zext to i32)
; CHECK-FIXED: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> zext to i32)
; CHECK-SCALABLE: Cost of 1 for VF vscale x 8: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> zext to i32)
target triple = "aarch64"

define i32 @sext_reduction_i16_to_i32(ptr %arr, i32 %n) vscale_range(1,16) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds i16, ptr %arr, i32 %iv
  %load = load i16, ptr %gep
  %sext = sext i16 %load to i32
  %add = add i32 %acc, %sext
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %add
}

define i32 @zext_reduction_i16_to_i32(ptr %arr, i32 %n) vscale_range(1,16) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds i16, ptr %arr, i32 %iv
  %load = load i16, ptr %gep
  %zext = zext i16 %load to i32
  %add = add i32 %acc, %zext
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %add
}
