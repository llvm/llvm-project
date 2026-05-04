; REQUIRES: asserts

; IC=1, VF=4, epilogue VF=4: dead epilogue.
; RUN: opt < %s -passes='loop-vectorize' -force-vector-width=4 \
; RUN:   -force-vector-interleave=1 -enable-epilogue-vectorization \
; RUN:   -epilogue-vectorization-force-VF=4 --debug-only=loop-vectorize \
; RUN:   --disable-output -S 2>&1 | FileCheck %s --check-prefix=DEAD-IC1

; IC=1, VF=4, epilogue VF=8: dead epilogue.
; RUN: opt < %s -passes='loop-vectorize' -force-vector-width=4 \
; RUN:   -force-vector-interleave=1 -enable-epilogue-vectorization \
; RUN:   -epilogue-vectorization-force-VF=8 --debug-only=loop-vectorize \
; RUN:   --disable-output -S 2>&1 | FileCheck %s --check-prefix=DEAD-IC1

; IC=2, VF=4, epilogue VF=4: epilogue is NOT dead.
; RUN: opt < %s -passes='loop-vectorize' -force-vector-width=4 \
; RUN:   -force-vector-interleave=2 -enable-epilogue-vectorization \
; RUN:   -epilogue-vectorization-force-VF=4 --debug-only=loop-vectorize \
; RUN:   --disable-output -S 2>&1 | FileCheck %s --check-prefix=NOT-DEAD-IC2

; IC=2, VF=4, epilogue VF=8: dead epilogue.
; RUN: opt < %s -passes='loop-vectorize' -force-vector-width=4 \
; RUN:   -force-vector-interleave=2 -enable-epilogue-vectorization \
; RUN:   -epilogue-vectorization-force-VF=8 --debug-only=loop-vectorize \
; RUN:   --disable-output -S 2>&1 | FileCheck %s --check-prefix=DEAD-IC2

; DEAD-IC1: LV: Checking a loop in 'simple_memset'
; DEAD-IC1: LEV: Forced epilogue VF results in dead epilogue vector loop, skipping vectorizing epilogue.

; NOT-DEAD-IC2: LV: Checking a loop in 'simple_memset'
; NOT-DEAD-IC2: LEV: Epilogue vectorization factor is forced.

; DEAD-IC2: LV: Checking a loop in 'simple_memset'
; DEAD-IC2: LEV: Forced epilogue VF results in dead epilogue vector loop, skipping vectorizing epilogue.

define void @simple_memset(ptr %dst, i64 %N) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 0, ptr %gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
