; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=FORCED-VF

; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 -scalable-vectorization=off \
; RUN:   -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN:   < %s 2>&1  | FileCheck %s --check-prefix=SCALABLE-OFF

; RUN: opt -S -passes=loop-vectorize \
; RUN:   -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=WIDTH-HINT

; Forcing a fixed-width VF does not by itself disable scalable vectorization:
; only an explicit request (-scalable-vectorization=off or the
; llvm.loop.vectorize.scalable.disable loop hint) does. The forced fixed VF is
; still the one used.

; FORCED-VF: Scalable vectorization is explicitly disabled
; FORCED-VF: remark: <unknown>:0:0: vectorized loop (vectorization width: 4, interleaved count: {{[0-9]+}})

; SCALABLE-OFF: remark: <unknown>:0:0: Scalable vectorization is explicitly disabled
; SCALABLE-OFF: remark: <unknown>:0:0: vectorized loop (vectorization width: 4, interleaved count: {{[0-9]+}})

target triple = "aarch64-unknown-linux-gnu"

define void @test_force_vec_width(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %lb = load i32, ptr %gep.b, align 4
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %iv
  %lc = load i32, ptr %gep.c, align 4
  %add = add nsw i32 %lb, %lc
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; WIDTH-HINT: Scalable vectorization is explicitly disabled
; WIDTH-HINT: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: {{[0-9]+}})

; SCALABLE-OFF: remark: <unknown>:0:0: Scalable vectorization is explicitly disabled
; SCALABLE-OFF: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: {{[0-9]+}})
define void @test_width_md(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %n) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %lb = load i32, ptr %gep.b, align 4
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %iv
  %lc = load i32, ptr %gep.c, align 4
  %add = add nsw i32 %lb, %lc
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

attributes #0 = { "target-features"="+sve"  }

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", i32 8}