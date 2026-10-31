; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=16 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize          \
; RUN:     -mattr=+sve -scalable-vectorization=on                                    \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE-BASE
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=16 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize          \
; RUN:     -mattr=+sve2p3 -scalable-vectorization=on                                \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=16 \
; RUN:     -enable-epilogue-vectorization=false -debug-only=loop-vectorize          \
; RUN:     -mattr=+sve,+sme2p3 -scalable-vectorization=on                           \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE

; LV: Checking a loop in 'sext_reduction_i8_to_i16'
; CHECK-SCALABLE-BASE-NOT: Cost of {{.*}} for VF vscale x 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (ir<%load> sext to i16)
; CHECK-SCALABLE: Cost of 1 for VF vscale x 16: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> sext to i16)

; LV: Checking a loop in 'zext_reduction_i8_to_i16'
; CHECK-SCALABLE-BASE-NOT: Cost of {{.*}} for VF vscale x 16: EXPRESSION vp<{{.*}}> = ir<%acc> + partial.reduce.add (ir<%load> sext to i16)
; CHECK-SCALABLE: Cost of 1 for VF vscale x 16: EXPRESSION vp<%8> = ir<%acc> + partial.reduce.add (ir<%load> zext to i16)

target triple = "aarch64"

define i16 @sext_reduction_i8_to_i16(ptr %arr, i32 %n) vscale_range(1,16) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds i8, ptr %arr, i32 %iv
  %load = load i8, ptr %gep
  %sext = sext i8 %load to i16
  %add = add i16 %acc, %sext
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i16 %add
}

define i16 @zext_reduction_i8_to_i16(ptr %arr, i32 %n) vscale_range(1,16) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi i16 [ 0, %entry ], [ %add, %loop ]
  %gep = getelementptr inbounds i8, ptr %arr, i32 %iv
  %load = load i8, ptr %gep
  %zext = zext i8 %load to i16
  %add = add i16 %acc, %zext
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i16 %add
}
