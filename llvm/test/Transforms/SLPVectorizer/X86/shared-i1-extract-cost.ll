; NOTE: Do not autogenerate. Check SLP decisions and selected lowering instructions.
; RUN: opt -passes=slp-vectorizer -S < %s | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -S < %s | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -passes=slp-vectorizer -mcpu=alderlake -S < %s | FileCheck %s --check-prefixes=CHECK,OTHER

; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -mattr=+sse3 -S < %s | FileCheck %s --check-prefixes=CHECK,OTHER
; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -slp-threshold=-100 -S < %s | FileCheck %s --check-prefix=FORCE

; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 < %s | llc -mcpu=x86-64 | FileCheck %s --check-prefix=ASM

; https://github.com/llvm/llvm-project/issues/228871

target triple = "x86_64-unknown-linux-gnu"

; The separately consumed i8 comparisons share mask materialization on SSE2.
; ASM-LABEL: separate_lanes:
; ASM-NOT: pcmpeqb
; ASM-NOT: movmsk
; ASM: retq
; SSE2-LABEL: @separate_lanes(
; SSE2-NOT: icmp eq <2 x i8>
; SSE2-COUNT-2: icmp eq i8
; SSE2: ret i1
; OTHER-LABEL: @separate_lanes(
; OTHER: icmp eq <2 x i8>
; OTHER: extractelement <2 x i1>
; OTHER: ret i1
define zeroext i1 @separate_lanes(ptr noalias noundef readonly align 8 dereferenceable(72) %l,
                          ptr noalias noundef readonly align 8 dereferenceable(72) %r) noinline nounwind {
entry:
  %lp2 = getelementptr inbounds i8, ptr %l, i64 56
  %l2 = load i8, ptr %lp2, align 8
  %lp5 = getelementptr inbounds i8, ptr %l, i64 16
  %l5 = load i64, ptr %lp5, align 8
  %lp6 = getelementptr inbounds i8, ptr %l, i64 57
  %l6 = load i8, ptr %lp6, align 1
  %rp2 = getelementptr inbounds i8, ptr %r, i64 56
  %r2 = load i8, ptr %rp2, align 8
  %rp5 = getelementptr inbounds i8, ptr %r, i64 16
  %r5 = load i64, ptr %rp5, align 8
  %rp6 = getelementptr inbounds i8, ptr %r, i64 57
  %r6 = load i8, ptr %rp6, align 1
  %eq2 = icmp eq i8 %l2, %r2
  %eq5 = icmp eq i64 %l5, %r5
  %take5 = and i1 %eq2, %eq5
  %eq6 = icmp eq i8 %l6, %r6
  %take6 = and i1 %take5, %eq6
  ret i1 %take6
}

; CHECK-LABEL: @whole_mask(
; CHECK: icmp eq <2 x i8>
; CHECK: call i1 @llvm.vector.reduce.and.v2i1
; CHECK: ret i1
define zeroext i1 @whole_mask(ptr noalias noundef readonly align 8 dereferenceable(72) %l,
                          ptr noalias noundef readonly align 8 dereferenceable(72) %r) noinline nounwind {
entry:
  %lp2 = getelementptr inbounds i8, ptr %l, i64 56
  %l2 = load i8, ptr %lp2, align 8
  %lp5 = getelementptr inbounds i8, ptr %l, i64 16
  %l5 = load i64, ptr %lp5, align 8
  %lp6 = getelementptr inbounds i8, ptr %l, i64 57
  %l6 = load i8, ptr %lp6, align 1
  %rp2 = getelementptr inbounds i8, ptr %r, i64 56
  %r2 = load i8, ptr %rp2, align 8
  %rp5 = getelementptr inbounds i8, ptr %r, i64 16
  %r5 = load i64, ptr %rp5, align 8
  %rp6 = getelementptr inbounds i8, ptr %r, i64 57
  %r6 = load i8, ptr %rp6, align 1
  %eq2 = icmp eq i8 %l2, %r2
  %eq6 = icmp eq i8 %l6, %r6
  %take6 = and i1 %eq2, %eq6
  %eq5 = icmp eq i64 %l5, %r5
  %take5 = and i1 %take6, %eq5
  ret i1 %take5
}

; The same decision applies when optimizing for size.
; SSE2-LABEL: @byte_separate_minsize(
; SSE2-NOT: icmp eq <2 x i8>
; SSE2-COUNT-2: icmp eq i8
; SSE2: ret i1
define zeroext i1 @byte_separate_minsize(ptr noalias noundef readonly align 8 dereferenceable(72) %l,
                          ptr noalias noundef readonly align 8 dereferenceable(72) %r) noinline nounwind optsize minsize {
entry:
  %lp2 = getelementptr inbounds i8, ptr %l, i64 56
  %l2 = load i8, ptr %lp2, align 8
  %lp5 = getelementptr inbounds i8, ptr %l, i64 16
  %l5 = load i64, ptr %lp5, align 8
  %lp6 = getelementptr inbounds i8, ptr %l, i64 57
  %l6 = load i8, ptr %lp6, align 1
  %rp2 = getelementptr inbounds i8, ptr %r, i64 56
  %r2 = load i8, ptr %rp2, align 8
  %rp5 = getelementptr inbounds i8, ptr %r, i64 16
  %r5 = load i64, ptr %rp5, align 8
  %rp6 = getelementptr inbounds i8, ptr %r, i64 57
  %r6 = load i8, ptr %rp6, align 1
  %eq2 = icmp eq i8 %l2, %r2
  %eq5 = icmp eq i64 %l5, %r5
  %take5 = and i1 %eq2, %eq5
  %eq6 = icmp eq i8 %l6, %r6
  %take6 = and i1 %take5, %eq6
  ret i1 %take6
}

; CHECK-LABEL: @word16_separate(
; CHECK: icmp eq <2 x i16>
; CHECK-COUNT-2: extractelement <2 x i1>
; CHECK: ret i1
define zeroext i1 @word16_separate(ptr noalias noundef readonly align 8 dereferenceable(72) %l,
                          ptr noalias noundef readonly align 8 dereferenceable(72) %r) noinline nounwind {
entry:
  %lp3 = getelementptr inbounds i8, ptr %l, i64 52
  %l3 = load i16, ptr %lp3, align 4
  %lp5 = getelementptr inbounds i8, ptr %l, i64 16
  %l5 = load i64, ptr %lp5, align 8
  %lp7 = getelementptr inbounds i8, ptr %l, i64 54
  %l7 = load i16, ptr %lp7, align 2
  %rp3 = getelementptr inbounds i8, ptr %r, i64 52
  %r3 = load i16, ptr %rp3, align 4
  %rp5 = getelementptr inbounds i8, ptr %r, i64 16
  %r5 = load i64, ptr %rp5, align 8
  %rp7 = getelementptr inbounds i8, ptr %r, i64 54
  %r7 = load i16, ptr %rp7, align 2
  %eq3 = icmp eq i16 %l3, %r3
  %eq5 = icmp eq i64 %l5, %r5
  %take5 = and i1 %eq3, %eq5
  %eq7 = icmp eq i16 %l7, %r7
  %take7 = and i1 %take5, %eq7
  ret i1 %take7
}

; CHECK-LABEL: @word32_separate(
; CHECK: icmp eq <2 x i32>
; CHECK-COUNT-2: extractelement <2 x i1>
; CHECK: ret i1
define zeroext i1 @word32_separate(ptr noalias noundef readonly align 8 dereferenceable(72) %l,
                          ptr noalias noundef readonly align 8 dereferenceable(72) %r) noinline nounwind {
entry:
  %lp4 = getelementptr inbounds i8, ptr %l, i64 40
  %l4 = load i32, ptr %lp4, align 8
  %lp5 = getelementptr inbounds i8, ptr %l, i64 16
  %l5 = load i64, ptr %lp5, align 8
  %lp8 = getelementptr inbounds i8, ptr %l, i64 44
  %l8 = load i32, ptr %lp8, align 4
  %rp4 = getelementptr inbounds i8, ptr %r, i64 40
  %r4 = load i32, ptr %rp4, align 8
  %rp5 = getelementptr inbounds i8, ptr %r, i64 16
  %r5 = load i64, ptr %rp5, align 8
  %rp8 = getelementptr inbounds i8, ptr %r, i64 44
  %r8 = load i32, ptr %rp8, align 4
  %eq4 = icmp eq i32 %l4, %r4
  %eq5 = icmp eq i64 %l5, %r5
  %take5 = and i1 %eq4, %eq5
  %eq8 = icmp eq i32 %l8, %r8
  %take8 = and i1 %take5, %eq8
  ret i1 %take8
}

; Force tree formation for these controls, even when the old cost is
; unprofitable. Their surviving extracts exercise the fallback paths.

; FORCE-LABEL: @single_lane(
; FORCE: [[MASK:%.*]] = icmp eq <2 x i8>
; FORCE: icmp eq <2 x i8>
; FORCE: [[OUT:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; FORCE-NEXT: call void @observe_one(i1 [[OUT]])
; FORCE-NEXT: ret void
define void @single_lane(ptr noalias %out, ptr noalias %a, ptr noalias %b,
                         ptr noalias %c, ptr noalias %d,
                         i32 %v, i32 %w, i32 %z) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %c1 = getelementptr inbounds i8, ptr %c, i64 1
  %d1 = getelementptr inbounds i8, ptr %d, i64 1
  %out1 = getelementptr inbounds i32, ptr %out, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %c0v = load i8, ptr %c, align 1
  %c1v = load i8, ptr %c1, align 1
  %d0v = load i8, ptr %d, align 1
  %d1v = load i8, ptr %d1, align 1
  %x0 = icmp eq i8 %a0v, %b0v
  %x1 = icmp eq i8 %a1v, %b1v
  %y0 = icmp eq i8 %c0v, %d0v
  %y1 = icmp eq i8 %c1v, %d1v
  %s0 = select i1 %x0, i32 %v, i32 %w
  %s1 = select i1 %x1, i32 %v, i32 %w
  %r0 = select i1 %y0, i32 %z, i32 %s0
  %r1 = select i1 %y1, i32 %z, i32 %s1
  store i32 %r0, ptr %out, align 4
  store i32 %r1, ptr %out1, align 4
  call void @observe_one(i1 %x1)
  ret void
}

; FORCE-LABEL: @repeated_lane(
; FORCE: [[MASK:%.*]] = icmp eq <2 x i8>
; FORCE: [[LO:%.*]] = extractelement <2 x i1> [[MASK]], i64 0
; FORCE-NEXT: [[HI:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; FORCE-NEXT: call void @observe_three(i1 [[HI]], i1 [[LO]], i1 [[HI]])
declare void @observe_three(i1, i1, i1)

define void @repeated_lane(ptr noalias %c, ptr noalias %d) {
entry:
  %c1 = getelementptr inbounds i8, ptr %c, i64 1
  %d1 = getelementptr inbounds i8, ptr %d, i64 1
  %c0v = load i8, ptr %c, align 1
  %c1v = load i8, ptr %c1, align 1
  %d0v = load i8, ptr %d, align 1
  %d1v = load i8, ptr %d1, align 1
  %y0 = icmp eq i8 %c0v, %d0v
  %x1 = icmp eq i8 %c1v, %d1v
  call void @observe_three(i1 %x1, i1 %y0, i1 %x1)
  ret void
}

; FORCE-LABEL: @separate_producers(
; FORCE: [[FIRST:%.*]] = icmp eq <2 x i8>
; FORCE-NEXT: [[SECOND:%.*]] = icmp eq <2 x i8>
; FORCE: [[LO:%.*]] = extractelement <2 x i1> [[SECOND]], i64 0
; FORCE-NEXT: [[HI:%.*]] = extractelement <2 x i1> [[FIRST]], i64 1
; FORCE-NEXT: call void @observe_two(i1 [[HI]], i1 [[LO]])
declare void @observe_two(i1, i1)

define void @separate_producers(ptr noalias %out, ptr noalias %a, ptr noalias %b,
                         ptr noalias %c, ptr noalias %d,
                         i32 %v, i32 %w, i32 %z) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %c1 = getelementptr inbounds i8, ptr %c, i64 1
  %d1 = getelementptr inbounds i8, ptr %d, i64 1
  %out1 = getelementptr inbounds i32, ptr %out, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %c0v = load i8, ptr %c, align 1
  %c1v = load i8, ptr %c1, align 1
  %d0v = load i8, ptr %d, align 1
  %d1v = load i8, ptr %d1, align 1
  %x0 = icmp eq i8 %a0v, %b0v
  %x1 = icmp eq i8 %a1v, %b1v
  %y0 = icmp eq i8 %c0v, %d0v
  %y1 = icmp eq i8 %c1v, %d1v
  %s0 = select i1 %x0, i32 %v, i32 %w
  %s1 = select i1 %x1, i32 %v, i32 %w
  %r0 = select i1 %y0, i32 %z, i32 %s0
  %r1 = select i1 %y1, i32 %z, i32 %s1
  store i32 %r0, ptr %out, align 4
  store i32 %r1, ptr %out1, align 4
  call void @observe_two(i1 %x1, i1 %y0)
  ret void
}

; FORCE-LABEL: @ephemeral_user(
; FORCE: icmp eq <2 x i8>
; FORCE-NEXT: [[MASK:%.*]] = icmp eq <2 x i8>
; FORCE: [[HI:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; FORCE-NEXT: call void @observe_one(i1 [[HI]])
; FORCE-NEXT: [[LO:%.*]] = extractelement <2 x i1> [[MASK]], i64 0
; FORCE-NEXT: call void @llvm.assume(i1 [[LO]])
declare void @observe_one(i1)
declare void @llvm.assume(i1)

define void @ephemeral_user(ptr noalias %out, ptr noalias %a, ptr noalias %b,
                         ptr noalias %c, ptr noalias %d,
                         i32 %v, i32 %w, i32 %z) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %c1 = getelementptr inbounds i8, ptr %c, i64 1
  %d1 = getelementptr inbounds i8, ptr %d, i64 1
  %out1 = getelementptr inbounds i32, ptr %out, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %c0v = load i8, ptr %c, align 1
  %c1v = load i8, ptr %c1, align 1
  %d0v = load i8, ptr %d, align 1
  %d1v = load i8, ptr %d1, align 1
  %x0 = icmp eq i8 %a0v, %b0v
  %x1 = icmp eq i8 %a1v, %b1v
  %y0 = icmp eq i8 %c0v, %d0v
  %y1 = icmp eq i8 %c1v, %d1v
  %s0 = select i1 %x0, i32 %v, i32 %w
  %s1 = select i1 %x1, i32 %v, i32 %w
  %r0 = select i1 %y0, i32 %z, i32 %s0
  %r1 = select i1 %y1, i32 %z, i32 %s1
  store i32 %r0, ptr %out, align 4
  store i32 %r1, ptr %out1, align 4
  call void @observe_one(i1 %y1)
  call void @llvm.assume(i1 %y0)
  ret void
}

; FORCE-LABEL: @cross_block(
; FORCE: [[MASK:%.*]] = icmp eq <2 x i8>
; FORCE: left:
; FORCE-NEXT: [[LO:%.*]] = extractelement <2 x i1> [[MASK]], i64 0
; FORCE-NEXT: call void @observe_left(i1 [[LO]])
; FORCE: right:
; FORCE-NEXT: [[HI:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; FORCE-NEXT: call void @observe_right(i1 [[HI]])
declare void @observe_left(i1)
declare void @observe_right(i1)

define void @cross_block(ptr noalias %a, ptr noalias %b, i1 %take_left) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  br i1 %take_left, label %left, label %right

left:
  call void @observe_left(i1 %lane0)
  ret void

right:
  call void @observe_right(i1 %lane1)
  ret void
}

; FORCE-LABEL: @phi_user(
; FORCE: [[MASK:%.*]] = icmp eq <2 x i8>
; FORCE: left:
; FORCE-NEXT: [[LO:%.*]] = extractelement <2 x i1> [[MASK]], i64 0
; FORCE-NEXT: call void @observe_left(i1 [[LO]])
; FORCE: right:
; FORCE-NEXT: [[HI:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; FORCE: merge:
; FORCE-NEXT: phi i1 [ false, %left ], [ [[HI]], %right ]
define void @phi_user(ptr noalias %a, ptr noalias %b, i1 %take_left) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  br i1 %take_left, label %left, label %right

left:
  call void @observe_left(i1 %lane0)
  br label %merge

right:
  br label %merge

merge:
  %chosen = phi i1 [ false, %left ], [ %lane1, %right ]
  call void @observe_right(i1 %chosen)
  ret void
}
