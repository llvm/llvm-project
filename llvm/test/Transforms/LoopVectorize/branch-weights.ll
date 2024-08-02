; RUN: opt < %s -S -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4  -enable-epilogue-vectorization -epilogue-vectorization-force-VF=4 | FileCheck %s

; CHECK-LABEL: @f0(
;
; CHECK: entry:
; CHECK:   br i1 %cmp.entry, label %iter.check, label %exit, !prof [[PROF_F0_ENTRY:![0-9]+]]
;
; CHECK: iter.check:
; CHECK:   br i1 %min.iters.check, label %vec.epilog.scalar.ph, label %vector.scevcheck, !prof [[PROF_F0_UNLIKELY:![0-9]+]]
;
; CHECK: vector.scevcheck:
; CHECK:   br i1 %4, label %vec.epilog.scalar.ph, label %vector.main.loop.iter.check, !prof [[PROF_F0_UNLIKELY]]
;
; CHECK: vector.main.loop.iter.check:
; CHECK:   br i1 %min.iters.check1, label %vec.epilog.ph, label %vector.ph, !prof [[PROF_F0_UNLIKELY]]
;
; CHECK: vector.ph:
; CHECK:   br label %vector.body
;
; CHECK: vector.body:
; CHECK:   br i1 %8, label %middle.block, label %vector.body, !prof [[PROF_F0_VECTOR_BODY:![0-9]+]]
;
; CHECK: middle.block:
; CHECK:   br i1 %cmp.n, label %exit.loopexit, label %vec.epilog.iter.check, !prof [[PROF_F0_MIDDLE_BLOCKS:![0-9]+]]
;
; CHECK: vec.epilog.iter.check:
; CHECK:   br i1 %min.epilog.iters.check, label %vec.epilog.scalar.ph, label %vec.epilog.ph, !prof [[PROF_F0_VEC_EPILOGUE_SKIP:![0-9]+]]
;
; CHECK: vec.epilog.ph:
; CHECK:   br label %vec.epilog.vector.body
;
; CHECK: vec.epilog.vector.body:
; CHECK:   br i1 %12, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !prof [[PROF_F0_VEC_EPILOG_VECTOR_BODY:![0-9]+]]
;
; CHECK: vec.epilog.middle.block:
; CHECK:   br i1 %cmp.n12, label %exit.loopexit, label %vec.epilog.scalar.ph, !prof [[PROF_F0_MIDDLE_BLOCKS:![0-9]+]]
;
; CHECK: vec.epilog.scalar.ph:
; CHECK:   br label %loop
;
; CHECK: loop:
; CHECK:   br i1 %cmp.loop, label %loop, label %exit.loopexit, !prof [[PROF_F0_LOOP:![0-9]+]]
;
; CHECK: exit.loopexit:
; CHECK:   br label %exit
;
; CHECK: exit:
; CHECK:   ret void

define void @f0(i8 %n, i32 %len, ptr %p) !prof !0 {
entry:
  %cmp.entry = icmp sgt i32 %len, 0
  br i1 %cmp.entry, label %loop, label %exit, !prof !1

loop:
  %i8 = phi i8 [0, %entry], [%i8.inc, %loop]
  %i32 = phi i32 [0, %entry], [%i32.inc, %loop]

  %ptr = getelementptr inbounds i32, ptr %p, i8 %i8
  store i32 %i32, ptr %ptr

  %i8.inc = add i8 %i8, 1
  %i32.inc = add i32 %i32, 1

  %cmp.loop = icmp ult i32 %i32, %len
  br i1 %cmp.loop, label %loop, label %exit, !prof !2

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 13}
!1 = !{!"branch_weights", i32 12, i32 1}
!2 = !{!"branch_weights", i32 1234, i32 1}

; CHECK: [[PROF_F0_ENTRY]] = !{!"branch_weights", i32 12, i32 1}
; CHECK: [[PROF_F0_UNLIKELY]] = !{!"branch_weights", i32 1, i32 127}
; CEHCK: [[PROF_F0_VECTOR_BODY]] = !{!"branch_weights", i32 1, i32 307}
; CHECK: [[PROF_F0_MIDDLE_BLOCKS]] =  !{!"branch_weights", i32 1, i32 3}
; CHECK: [[PROF_F0_VEC_EPILOGUE_SKIP]] = !{!"branch_weights", i32 4, i32 0}
; CHECK: [[PROF_F0_VEC_EPILOG_VECTOR_BODY]] = !{!"branch_weights", i32 0, i32 0}
; CEHCK: [[PROF_F0_LOOP]] = !{!"branch_weights", i32 2, i32 1}
