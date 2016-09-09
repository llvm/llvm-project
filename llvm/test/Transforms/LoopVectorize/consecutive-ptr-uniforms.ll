; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses -enable-cond-stores-vec -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s --check-prefix=INTER

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

%pair = type { i32, i32 }

; CHECK-LABEL: consecutive_ptr_forward
;
; Check that a forward consecutive pointer is recognized as uniform and remains
; uniform after vectorization.
;
; CHECK:     LV: Found uniform instruction: %tmp1 = getelementptr inbounds i32, i32* %a, i64 %i
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NOT:   getelementptr
; CHECK:       getelementptr inbounds i32, i32* %a, i64 %index
; CHECK-NOT:   getelementptr
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @consecutive_ptr_forward(i32* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %tmp3 = add i32 %tmp0, %tmp2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  ret i32 %tmp4
}

; CHECK-LABEL: consecutive_ptr_reverse
;
; Check that a reverse consecutive pointer is recognized as uniform and remains
; uniform after vectorization.
;
; CHECK:     LV: Found uniform instruction: %tmp1 = getelementptr inbounds i32, i32* %a, i64 %i
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %offset.idx = sub i64 %n, %index
; CHECK-NOT:   getelementptr
; CHECK:       %[[G0:.+]] = getelementptr inbounds i32, i32* %a, i64 %offset.idx
; CHECK:       getelementptr i32, i32* %[[G0]], i64 -3
; CHECK-NOT:   getelementptr
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @consecutive_ptr_reverse(i32* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ %n, %entry ]
  %tmp0 = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp2 = load i32, i32* %tmp1, align 8
  %tmp3 = add i32 %tmp0, %tmp2
  %i.next = add nuw nsw i64 %i, -1
  %cond = icmp sgt i64 %i.next, 0
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  ret i32 %tmp4
}

; CHECK-LABEL: interleaved_access_forward
; INTER-LABEL: interleaved_access_forward
;
; Check that a consecutive-like pointer used by a forward interleaved group is
; recognized as uniform and remains uniform after vectorization. When
; interleaved memory accesses aren't enabled, the pointer should not be
; recognized as uniform, and it should not be uniform after vectorization.
;
; CHECK-NOT: LV: Found uniform instruction: %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
; CHECK-NOT: LV: Found uniform instruction: %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %[[I1:.+]] = or i64 %index, 1
; CHECK:       %[[I2:.+]] = or i64 %index, 2
; CHECK:       %[[I3:.+]] = or i64 %index, 3
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %index, i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %index, i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 1
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
; INTER:     LV: Found uniform instruction: %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
; INTER:     LV: Found uniform instruction: %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
; INTER:     vector.body
; INTER:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; INTER-NOT:   getelementptr
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %index, i32 0
; INTER-NOT:   getelementptr
; INTER:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @interleaved_access_forward(%pair* %p, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = phi i32 [ %tmp6, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
  %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
  %tmp3 = load i32, i32* %tmp1, align 8
  %tmp4 = load i32, i32* %tmp2, align 8
  %tmp5 = add i32 %tmp3, %tmp4
  %tmp6 = add i32 %tmp0, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp14 = phi i32 [ %tmp6, %for.body ]
  ret i32 %tmp14
}

; CHECK-LABEL: interleaved_access_reverse
; INTER-LABEL: interleaved_access_reverse
;
; Check that a consecutive-like pointer used by a reverse interleaved group is
; recognized as uniform and remains uniform after vectorization. When
; interleaved memory accesses aren't enabled, the pointer should not be
; recognized as uniform, and it should not be uniform after vectorization.
;
; recognized as uniform, and it should not be uniform after vectorization.
; CHECK-NOT: LV: Found uniform instruction: %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
; CHECK-NOT: LV: Found uniform instruction: %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %offset.idx = sub i64 %n, %index
; CHECK:       %[[I1:.+]] = add i64 %offset.idx, -1
; CHECK:       %[[I2:.+]] = add i64 %offset.idx, -2
; CHECK:       %[[I3:.+]] = add i64 %offset.idx, -3
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %offset.idx, i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 0
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %offset.idx, i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 1
; CHECK:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 1
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
; INTER:     LV: Found uniform instruction: %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
; INTER:     LV: Found uniform instruction: %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
; INTER:     vector.body
; INTER:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; INTER:       %offset.idx = sub i64 %n, %index
; INTER-NOT:   getelementptr
; INTER:       %[[G0:.+]] = getelementptr inbounds %pair, %pair* %p, i64 %offset.idx, i32 0
; INTER:       getelementptr i32, i32* %[[G0]], i64 -6
; INTER-NOT:   getelementptr
; INTER:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @interleaved_access_reverse(%pair* %p, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ %n, %entry ]
  %tmp0 = phi i32 [ %tmp6, %for.body ], [ 0, %entry ]
  %tmp1 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
  %tmp2 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 1
  %tmp3 = load i32, i32* %tmp1, align 8
  %tmp4 = load i32, i32* %tmp2, align 8
  %tmp5 = add i32 %tmp3, %tmp4
  %tmp6 = add i32 %tmp0, %tmp5
  %i.next = add nuw nsw i64 %i, -1
  %cond = icmp sgt i64 %i.next, 0
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp14 = phi i32 [ %tmp6, %for.body ]
  ret i32 %tmp14
}

; INTER-LABEL: predicated_store
;
; Check that a consecutive-like pointer used by a forward interleaved group and
; scalarized store is not recognized as uniform and is not uniform after
; vectorization. The store is scalarized because it's in a predicated block.
; Even though the load in this example is vectorized and only uses the pointer
; as if it were uniform, the store is scalarized, making the pointer
; non-uniform.
;
; INTER-NOT: LV: Found uniform instruction: %tmp0 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
; INTER:     vector.body
; INTER:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, {{.*}} ]
; INTER:       %[[I1:.+]] = or i64 %index, 1
; INTER:       %[[I2:.+]] = or i64 %index, 2
; INTER:       %[[I3:.+]] = or i64 %index, 3
; INTER:       %[[G0:.+]] = getelementptr inbounds %pair, %pair* %p, i64 %index, i32 0
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 0
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 0
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 0
; INTER:       %[[B0:.+]] = bitcast i32* %[[G0]] to <8 x i32>*
; INTER:       %wide.vec = load <8 x i32>, <8 x i32>* %[[B0]], align 8
; INTER:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @predicated_store(%pair *%p, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i  = phi i64 [ %i.next, %if.merge ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds %pair, %pair* %p, i64 %i, i32 0
  %tmp1 = load i32, i32* %tmp0, align 8
  %tmp2 = icmp eq i32 %tmp1, %x
  br i1 %tmp2, label %if.then, label %if.merge

if.then:
  store i32 %tmp1, i32* %tmp0, align 8
  br label %if.merge

if.merge:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: irregular_type
;
; Check that a consecutive pointer used by a scalarized store is not recognized
; as uniform and is not uniform after vectorization. The store is scalarized
; because the stored type may required padding.
;
; CHECK-NOT: LV: Found uniform instruction: %tmp1 = getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %i
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %[[I1:.+]] = or i64 %index, 1
; CHECK:       %[[I2:.+]] = or i64 %index, 2
; CHECK:       %[[I3:.+]] = or i64 %index, 3
; CHECK:       getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %index
; CHECK:       getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %[[I1]]
; CHECK:       getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %[[I2]]
; CHECK:       getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %[[I3]]
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @irregular_type(x86_fp80* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = sitofp i32 1 to x86_fp80
  %tmp1 = getelementptr inbounds x86_fp80, x86_fp80* %a, i64 %i
  store x86_fp80 %tmp0, x86_fp80* %tmp1, align 16
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
