; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine -enable-new-pm=0 2>&1 | FileCheck %s --check-prefix=INTER
; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s --check-prefix=INTER

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
; CHECK:       %[[G0:.+]] = getelementptr inbounds i32, i32* %a, i64 -3
; CHECK:       getelementptr inbounds i32, i32* %[[G0]], i64 %offset.idx
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
  %i.next = add nsw i64 %i, -1
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
; INTER:       getelementptr inbounds i32, i32* %[[G0]], i64 -6
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
  %i.next = add nsw i64 %i, -1
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
; INTER:       %[[G0:.+]] = getelementptr inbounds %pair, %pair* %p, i64 %index, i32 0
; INTER:       %[[B0:.+]] = bitcast i32* %[[G0]] to <8 x i32>*
; INTER:       %wide.vec = load <8 x i32>, <8 x i32>* %[[B0]], align 8
; INTER:       %[[I1:.+]] = or i64 %index, 1
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I1]], i32 0
; INTER:       %[[I2:.+]] = or i64 %index, 2
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I2]], i32 0
; INTER:       %[[I3:.+]] = or i64 %index, 3
; INTER:       getelementptr inbounds %pair, %pair* %p, i64 %[[I3]], i32 0
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

; CHECK-LABEL: pointer_iv_uniform
;
; Check that a pointer induction variable is recognized as uniform and remains
; uniform after vectorization.
;
; CHECK:     LV: Found uniform instruction: %p = phi i32* [ %tmp03, %for.body ], [ %a, %entry ]
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NOT:   getelementptr
; CHECK:       %next.gep = getelementptr i32, i32* %a, i64 %index
; CHECK-NOT:   getelementptr
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @pointer_iv_uniform(i32* %a, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %p = phi i32* [ %tmp03, %for.body ], [ %a, %entry ]
  store i32 %x, i32* %p, align 8
  %tmp03 = getelementptr inbounds i32, i32* %p, i32 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; INTER-LABEL: pointer_iv_non_uniform_0
;
; Check that a pointer induction variable with a non-uniform user is not
; recognized as uniform and is not uniform after vectorization. The pointer
; induction variable is used by getelementptr instructions that are non-uniform
; due to scalarization of the stores.
;
; INTER-NOT: LV: Found uniform instruction: %p = phi i32* [ %tmp03, %for.body ], [ %a, %entry ]
; INTER:     vector.body
; INTER:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; INTER:       %[[I0:.+]] = shl i64 %index, 2
; INTER:       %next.gep = getelementptr i32, i32* %a, i64 %[[I0]]
; INTER:       %[[S1:.+]] = shl i64 %index, 2
; INTER:       %[[I1:.+]] = or i64 %[[S1]], 4
; INTER:       %next.gep2 = getelementptr i32, i32* %a, i64 %[[I1]]
; INTER:       %[[S2:.+]] = shl i64 %index, 2
; INTER:       %[[I2:.+]] = or i64 %[[S2]], 8
; INTER:       %next.gep3 = getelementptr i32, i32* %a, i64 %[[I2]]
; INTER:       %[[S3:.+]] = shl i64 %index, 2
; INTER:       %[[I3:.+]] = or i64 %[[S3]], 12
; INTER:       %next.gep4 = getelementptr i32, i32* %a, i64 %[[I3]]
; INTER:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @pointer_iv_non_uniform_0(i32* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %p = phi i32* [ %tmp03, %for.body ], [ %a, %entry ]
  %tmp00 = load i32, i32* %p, align 8
  %tmp01 = getelementptr inbounds i32, i32* %p, i32 1
  %tmp02 = load i32, i32* %tmp01, align 8
  %tmp03 = getelementptr inbounds i32, i32* %p, i32 4
  %tmp04 = load i32, i32* %tmp03, align 8
  %tmp05 = getelementptr inbounds i32, i32* %p, i32 5
  %tmp06 = load i32, i32* %tmp05, align 8
  %tmp07 = sub i32 %tmp04, %tmp00
  %tmp08 = sub i32 %tmp02, %tmp02
  %tmp09 = getelementptr inbounds i32, i32* %p, i32 2
  store i32 %tmp07, i32* %tmp09, align 8
  %tmp10 = getelementptr inbounds i32, i32* %p, i32 3
  store i32 %tmp08, i32* %tmp10, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: pointer_iv_non_uniform_1
;
; Check that a pointer induction variable with a non-uniform user is not
; recognized as uniform and is not uniform after vectorization. The pointer
; induction variable is used by a store that will be scalarized.
;
; CHECK-NOT: LV: Found uniform instruction: %p = phi x86_fp80* [%tmp1, %for.body], [%a, %entry]
; CHECK:     vector.body
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %next.gep = getelementptr x86_fp80, x86_fp80* %a, i64 %index
; CHECK:       %[[I1:.+]] = or i64 %index, 1
; CHECK:       %next.gep2 = getelementptr x86_fp80, x86_fp80* %a, i64 %[[I1]]
; CHECK:       %[[I2:.+]] = or i64 %index, 2
; CHECK:       %next.gep3 = getelementptr x86_fp80, x86_fp80* %a, i64 %[[I2]]
; CHECK:       %[[I3:.+]] = or i64 %index, 3
; CHECK:       %next.gep4 = getelementptr x86_fp80, x86_fp80* %a, i64 %[[I3]]
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @pointer_iv_non_uniform_1(x86_fp80* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %p = phi x86_fp80* [%tmp1, %for.body], [%a, %entry]
  %tmp0 = sitofp i32 1 to x86_fp80
  store x86_fp80 %tmp0, x86_fp80* %p, align 16
  %tmp1 = getelementptr inbounds x86_fp80, x86_fp80* %p, i32 1
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: pointer_iv_mixed
;
; Check multiple pointer induction variables where only one is recognized as
; uniform and remains uniform after vectorization. The other pointer induction
; variable is not recognized as uniform and is not uniform after vectorization
; because it is stored to memory.
;
; CHECK-NOT: LV: Found uniform instruction: %p = phi i32* [ %tmp3, %for.body ], [ %a, %entry ]
; CHECK:     LV: Found uniform instruction: %q = phi i32** [ %tmp4, %for.body ], [ %b, %entry ]
; CHECK:     vector.body
; CHECK:       %pointer.phi = phi i32* [ %a, %vector.ph ], [ %ptr.ind, %vector.body ]
; CHECK:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:       %[[PTRVEC:.+]] = getelementptr i32, i32* %pointer.phi, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
; CHECK:       %next.gep = getelementptr i32*, i32** %b, i64 %index
; CHECK:       %[[NEXTGEPBC:.+]] = bitcast i32** %next.gep to <4 x i32*>*
; CHECK:       store <4 x i32*> %[[PTRVEC]], <4 x i32*>* %[[NEXTGEPBC]], align 8
; CHECK:       %ptr.ind = getelementptr i32, i32* %pointer.phi, i64 4
; CHECK:       br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @pointer_iv_mixed(i32* %a, i32** %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %p = phi i32* [ %tmp3, %for.body ], [ %a, %entry ]
  %q = phi i32** [ %tmp4, %for.body ], [ %b, %entry ]
  %tmp0 = phi i32 [ %tmp2, %for.body ], [ 0, %entry ]
  %tmp1 = load i32, i32* %p, align 8
  %tmp2 = add i32 %tmp1, %tmp0
  store i32* %p, i32** %q, align 8
  %tmp3 = getelementptr inbounds i32, i32* %p, i32 1
  %tmp4 = getelementptr inbounds i32*, i32** %q, i32 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp5 = phi i32 [ %tmp2, %for.body ]
  ret i32 %tmp5
}

; INTER-LABEL: bitcast_pointer_operand
;
; Check that a pointer operand having a user other than a memory access is
; recognized as uniform after vectorization. In this test case, %tmp1 is a
; bitcast that is used by a load and a getelementptr instruction (%tmp2). Once
; %tmp2 is marked uniform, %tmp1 should be marked uniform as well.
;
; INTER:       LV: Found uniform instruction: %cond = icmp slt i64 %i.next, %n
; INTER-NEXT:  LV: Found uniform instruction: %tmp2 = getelementptr inbounds i8, i8* %tmp1, i64 3
; INTER-NEXT:  LV: Found uniform instruction: %tmp6 = getelementptr inbounds i8, i8* %B, i64 %i
; INTER-NEXT:  LV: Found uniform instruction: %tmp1 = bitcast i64* %tmp0 to i8*
; INTER-NEXT:  LV: Found uniform instruction: %tmp0 = getelementptr inbounds i64, i64* %A, i64 %i
; INTER-NEXT:  LV: Found uniform instruction: %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
; INTER-NEXT:  LV: Found uniform instruction: %i.next = add nuw nsw i64 %i, 1
; INTER:       define void @bitcast_pointer_operand(
; INTER:       vector.body:
; INTER-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; INTER-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i64, i64* %A, i64 [[INDEX]]
; INTER-NEXT:    [[TMP5:%.*]] = bitcast i64* [[TMP4]] to <32 x i8>*
; INTER-NEXT:    [[WIDE_VEC:%.*]] = load <32 x i8>, <32 x i8>* [[TMP5]], align 1
; INTER-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <32 x i8> [[WIDE_VEC]], <32 x i8> poison, <4 x i32> <i32 0, i32 8, i32 16, i32 24>
; INTER-NEXT:    [[STRIDED_VEC5:%.*]] = shufflevector <32 x i8> [[WIDE_VEC]], <32 x i8> poison, <4 x i32> <i32 3, i32 11, i32 19, i32 27>
; INTER-NEXT:    [[TMP6:%.*]] = xor <4 x i8> [[STRIDED_VEC5]], [[STRIDED_VEC]]
; INTER-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i8, i8* %B, i64 [[INDEX]]
; INTER-NEXT:    [[TMP8:%.*]] = bitcast i8* [[TMP7]] to <4 x i8>*
; INTER-NEXT:    store <4 x i8> [[TMP6]], <4 x i8>* [[TMP8]], align 1
; INTER-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; INTER:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @bitcast_pointer_operand(i64* %A, i8* %B, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds i64, i64* %A, i64 %i
  %tmp1 = bitcast i64* %tmp0 to i8*
  %tmp2 = getelementptr inbounds i8, i8* %tmp1, i64 3
  %tmp3 = load i8, i8* %tmp2, align 1
  %tmp4 = load i8, i8* %tmp1, align 1
  %tmp5 = xor i8 %tmp3, %tmp4
  %tmp6 = getelementptr inbounds i8, i8* %B, i64 %i
  store i8 %tmp5, i8* %tmp6
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
