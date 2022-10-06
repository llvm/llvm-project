; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,INLINE"
; RUN: opt < %s -msan-check-access-address=0 -msan-poison-stack-with-call=1 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,CALL"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,ORIGIN"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,ORIGIN"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -msan-print-stack-names=false -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,ORIGIN-LEAN"
; RUN: opt < %s -S -passes="msan<kernel>" 2>&1 | FileCheck %s "--check-prefixes=CHECK,KMSAN"
; RUN: opt < %s -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,KMSAN"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; ORIGIN: [[IDPTR:@[0-9]+]] = private global i32 0
; ORIGIN-LEAN: [[IDPTR:@[0-9]+]] = private global i32 0
; ORIGIN: [[DESCR:@[0-9]+]] = private constant [9 x i8] c"unique_x\00"

define void @static() sanitize_memory {
entry:
  %unique_x = alloca i32, align 4
  ret void
}

; CHECK-LABEL: define void @static(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4, i8* {{.*}} [[IDPTR]] {{.*}}, i8* {{.*}} [[DESCR]],
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4, i8* {{.*}} [[IDPTR]]
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: ret void


define void @dynamic() sanitize_memory {
entry:
  br label %l
l:
  %x = alloca i32, align 4
  ret void
}

; CHECK-LABEL: define void @dynamic(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: ret void

define void @array() sanitize_memory {
entry:
  %x = alloca i32, i64 5, align 4
  ret void
}

; CHECK-LABEL: define void @array(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 20, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 20)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 20,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 20,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 20,
; CHECK: ret void

define void @array32() sanitize_memory {
entry:
  %x = alloca i32, i32 5, align 4
  ret void
}

; CHECK-LABEL: define void @array32(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 20, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 20)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 20,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 20,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 20,
; CHECK: ret void

define void @array_non_const(i64 %cnt) sanitize_memory {
entry:
  %x = alloca i32, i64 %cnt, align 4
  ret void
}

; CHECK-LABEL: define void @array_non_const(
; CHECK: %[[A:.*]] = mul i64 4, %cnt
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 %[[A]], i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 %[[A]])
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 %[[A]],
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 %[[A]],
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 %[[A]],
; CHECK: ret void

define void @array_non_const32(i32 %cnt) sanitize_memory {
entry:
  %x = alloca i32, i32 %cnt, align 4
  ret void
}

; CHECK-LABEL: define void @array_non_const32(
; CHECK: %[[Z:.*]] = zext i32 %cnt to i64
; CHECK: %[[A:.*]] = mul i64 4, %[[Z]]
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 %[[A]], i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 %[[A]])
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 %[[A]],
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 %[[A]],
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 %[[A]],
; CHECK: ret void

; Check that the local is unpoisoned in the absence of sanitize_memory
define void @unpoison_local() {
entry:
  %x = alloca i32, i64 5, align 4
  ret void
}

; CHECK-LABEL: define void @unpoison_local(
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 0, i64 20, i1 false)
; CALL: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 0, i64 20, i1 false)
; ORIGIN-NOT: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 20,
; ORIGIN-LEAN-NOT: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 20,
; KMSAN: call void @__msan_unpoison_alloca(i8* {{.*}}, i64 20)
; CHECK: ret void

; Check that every llvm.lifetime.start() causes poisoning of locals.
define void @lifetime_start() sanitize_memory {
entry:
  %x = alloca i32, align 4
  %c = bitcast i32* %x to i8*
  br label %another_bb

another_bb:
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %c)
  store i32 7, i32* %x
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %c)
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %c)
  store i32 8, i32* %x
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %c)
  ret void
}

; CHECK-LABEL: define void @lifetime_start(
; CHECK-LABEL: entry:
; CHECK: %x = alloca i32
; CHECK-LABEL: another_bb:

; CHECK: call void @llvm.lifetime.start
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,

; CHECK: call void @llvm.lifetime.start
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: ret void

; Make sure variable-length arrays are handled correctly.
define void @lifetime_start_var(i64 %cnt) sanitize_memory {
entry:
  %x = alloca i32, i64 %cnt, align 4
  %c = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* nonnull %c)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* nonnull %c)
  ret void
}

; CHECK-LABEL: define void @lifetime_start_var(
; CHECK-LABEL: entry:
; CHECK: %x = alloca i32, i64 %cnt
; CHECK: call void @llvm.lifetime.start
; CHECK: %[[A:.*]] = mul i64 4, %cnt
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 %[[A]], i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 %[[A]])
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 %[[A]],
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 %[[A]],
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 %[[A]],
; CHECK: call void @llvm.lifetime.end
; CHECK: ret void


; If we can't trace one of the lifetime markers to a single alloca, fall back
; to poisoning allocas at the beginning of the function.
; Each alloca must be poisoned only once.
define void @lifetime_no_alloca(i8 %v) sanitize_memory {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %cx = bitcast i32* %x to i8*
  %cy = bitcast i32* %y to i8*
  %cz = bitcast i32* %z to i8*
  %tobool = icmp eq i8 %v, 0
  %xy = select i1 %tobool, i32* %x, i32* %y
  %cxcy = select i1 %tobool, i8* %cx, i8* %cy
  br label %another_bb

another_bb:
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %cz)
  store i32 7, i32* %z
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %cz)
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %cz)
  store i32 7, i32* %z
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %cz)
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %cxcy)
  store i32 8, i32* %xy
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %cxcy)
  ret void
}

; CHECK-LABEL: define void @lifetime_no_alloca(
; CHECK-LABEL: entry:
; CHECK: %x = alloca i32
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: %y = alloca i32
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: %z = alloca i32
; INLINE: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,

; There're two lifetime intrinsics for %z, but we must instrument it only once.
; INLINE-NOT: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL-NOT: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN-NOT: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN-NOT: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN-NOT: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK-LABEL: another_bb:

; CHECK: call void @llvm.lifetime.start
; INLINE-NOT: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL-NOT: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN-NOT: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN-NOT: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN-NOT: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: call void @llvm.lifetime.end
; CHECK: call void @llvm.lifetime.start
; INLINE-NOT: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 -1, i64 4, i1 false)
; CALL-NOT: call void @__msan_poison_stack(i8* {{.*}}, i64 4)
; ORIGIN-NOT: call void @__msan_set_alloca_origin_with_descr(i8* {{.*}}, i64 4,
; ORIGIN-LEAN-NOT: call void @__msan_set_alloca_origin_no_descr(i8* {{.*}}, i64 4,
; KMSAN-NOT: call void @__msan_poison_alloca(i8* {{.*}}, i64 4,
; CHECK: call void @llvm.lifetime.end



declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
