; RUN: llc < %s -mtriple=arm64-eabi -mattr=+mte | FileCheck %s

; test create_tag
define ptr @create_tag(ptr %ptr, i32 %m) {
entry:
; CHECK-LABEL: create_tag:
  %0 = zext i32 %m to i64
  %1 = tail call ptr @llvm.aarch64.irg(ptr %ptr, i64 %0)
  ret ptr %1
;CHECK: irg x0, x0, {{x[0-9]+}}
}

; *********** __arm_mte_increment_tag  *************
; test increment_tag1
define ptr @increment_tag1(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag1:
  %0 = tail call ptr @llvm.aarch64.addg(ptr %ptr, i64 7)
  ret ptr %0
; CHECK: addg x0, x0, #0, #7
}

%struct.S2K = type { [512 x i32] }
define ptr @increment_tag1stack(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag1stack:
  %s = alloca %struct.S2K, align 4
  call void @llvm.lifetime.start.p0(i64 2048, ptr nonnull %s)
  %0 = call ptr @llvm.aarch64.addg(ptr nonnull %s, i64 7)
  call void @llvm.lifetime.end.p0(i64 2048, ptr nonnull %s)
  ret ptr %0
; CHECK: addg x0, sp, #0, #7
}


define ptr @increment_tag2(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag2:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 4
  %0 = tail call ptr @llvm.aarch64.addg(ptr nonnull %add.ptr, i64 7)
  ret ptr %0
; CHECK: addg x0, x0, #16, #7
}

define ptr @increment_tag2stack(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag2stack:
  %s = alloca %struct.S2K, align 4
  call void @llvm.lifetime.start.p0(i64 2048, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S2K, ptr %s, i64 0, i32 0, i64 4
  %0 = call ptr @llvm.aarch64.addg(ptr nonnull %arrayidx, i64 7)
  call void @llvm.lifetime.end.p0(i64 2048, ptr nonnull %s)
  ret ptr %0
; CHECK: addg x0, sp, #16, #7
}

define ptr @increment_tag3(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag3:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 252
  %0 = tail call ptr @llvm.aarch64.addg(ptr nonnull %add.ptr, i64 7)
  ret ptr %0
; CHECK: addg x0, x0, #1008, #7
}

define ptr @increment_tag3stack(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag3stack:
  %s = alloca %struct.S2K, align 4
  call void @llvm.lifetime.start.p0(i64 2048, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S2K, ptr %s, i64 0, i32 0, i64 252
  %0 = call ptr @llvm.aarch64.addg(ptr nonnull %arrayidx, i64 7)
  call void @llvm.lifetime.end.p0(i64 2048, ptr nonnull %s)
  ret ptr %0
; CHECK: addg x0, sp, #1008, #7
}


define ptr @increment_tag4(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag4:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 256
  %0 = tail call ptr @llvm.aarch64.addg(ptr nonnull %add.ptr, i64 7)
  ret ptr %0
; CHECK: add [[T0:x[0-9]+]], x0, #1024
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}

define ptr @increment_tag4stack(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag4stack:
  %s = alloca %struct.S2K, align 4
  call void @llvm.lifetime.start.p0(i64 2048, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S2K, ptr %s, i64 0, i32 0, i64 256
  %0 = call ptr @llvm.aarch64.addg(ptr nonnull %arrayidx, i64 7)
  call void @llvm.lifetime.end.p0(i64 2048, ptr nonnull %s)
  ret ptr %0
; CHECK: add [[T0:x[0-9]+]], {{.*}}, #1024
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}


define ptr @increment_tag5(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag5:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 5
  %0 = tail call ptr @llvm.aarch64.addg(ptr nonnull %add.ptr, i64 7)
  ret ptr %0
; CHECK: add [[T0:x[0-9]+]], x0, #20
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}

define ptr @increment_tag5stack(ptr %ptr) {
entry:
; CHECK-LABEL: increment_tag5stack:
  %s = alloca %struct.S2K, align 4
  call void @llvm.lifetime.start.p0(i64 2048, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S2K, ptr %s, i64 0, i32 0, i64 5
  %0 = call ptr @llvm.aarch64.addg(ptr nonnull %arrayidx, i64 7)
  call void @llvm.lifetime.end.p0(i64 2048, ptr nonnull %s)
  ret ptr %0
; CHECK: add [[T0:x[0-9]+]], {{.*}}, #20
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}


; *********** __arm_mte_exclude_tag  *************
; test exclude_tag
define i32 @exclude_tag(ptr %ptr, i32 %m) local_unnamed_addr #0 {
entry:
;CHECK-LABEL: exclude_tag:
  %0 = zext i32 %m to i64
  %1 = tail call i64 @llvm.aarch64.gmi(ptr %ptr, i64 %0)
  %conv = trunc i64 %1 to i32
  ret i32 %conv
; CHECK: gmi	x0, x0, {{x[0-9]+}}
}


; *********** __arm_mte_get_tag *************
%struct.S8K = type { [2048 x i32] }
define ptr @get_tag1(ptr %ptr) {
entry:
; CHECK-LABEL: get_tag1:
  %0 = tail call ptr @llvm.aarch64.ldg(ptr %ptr, ptr %ptr)
  ret ptr %0
; CHECK: ldg x0, [x0]
}

define ptr @get_tag1_two_parm(ptr %ret_ptr, ptr %ptr) {
entry:
; CHECK-LABEL: get_tag1_two_parm:
  %0 = tail call ptr @llvm.aarch64.ldg(ptr %ret_ptr, ptr %ptr)
  ret ptr %0
; CHECK: ldg x0, [x1]
}

define ptr @get_tag1stack() {
entry:
; CHECK-LABEL: get_tag1stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %s, ptr nonnull %s)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: ldg [[T0]], [sp]
}

define ptr @get_tag1stack_two_param(ptr %ret_ptr) {
entry:
; CHECK-LABEL: get_tag1stack_two_param:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %ret_ptr, ptr nonnull %s)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK-NOT: mov {{.*}}, sp
; CHECK: ldg x0, [sp]
}


define ptr @get_tag2(ptr %ptr) {
entry:
; CHECK-LABEL: get_tag2:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 4
  %0 = tail call ptr @llvm.aarch64.ldg(ptr nonnull %add.ptr, ptr nonnull %add.ptr)
  ret ptr %0
; CHECK: add  [[T0:x[0-9]+]], x0, #16
; CHECK: ldg  [[T0]], [x0, #16]
}

define ptr @get_tag2stack() {
entry:
; CHECK-LABEL: get_tag2stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 4
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %arrayidx, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x0, [[T0]], #16
; CHECK: ldg x0, [sp, #16]
}


define ptr @get_tag3(ptr %ptr) {
entry:
; CHECK-LABEL: get_tag3:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 1020
  %0 = tail call ptr @llvm.aarch64.ldg(ptr nonnull %add.ptr, ptr nonnull %add.ptr)
  ret ptr %0
; CHECK: add [[T0:x[0-8]+]], x0, #4080
; CHECK: ldg [[T0]], [x0, #4080]
}

define ptr @get_tag3stack() {
entry:
; CHECK-LABEL: get_tag3stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 1020
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %arrayidx, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x0, [[T0]], #4080
; CHECK: ldg x0, [sp, #4080]
}


define ptr @get_tag4(ptr %ptr) {
entry:
; CHECK-LABEL: get_tag4:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 1024
  %0 = tail call ptr @llvm.aarch64.ldg(ptr nonnull %add.ptr, ptr nonnull %add.ptr)
  ret ptr %0
; CHECK: add x0, x0, #1, lsl #12
; CHECK-NEXT: ldg x0, [x0]
}

define ptr @get_tag4stack() {
entry:
; CHECK-LABEL: get_tag4stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 1024
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %arrayidx, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK-NEXT: add x[[T1:[0-9]+]], [[T0]], #1, lsl #12
; CHECK-NEXT: ldg x[[T1]], [x[[T1]]]
}

define ptr @get_tag5(ptr %ptr) {
entry:
; CHECK-LABEL: get_tag5:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 5
  %0 = tail call ptr @llvm.aarch64.ldg(ptr nonnull %add.ptr, ptr nonnull %add.ptr)
  ret ptr %0
; CHECK: add x0, x0, #20
; CHECK-NEXT: ldg x0, [x0]
}

define ptr @get_tag5stack() {
entry:
; CHECK-LABEL: get_tag5stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 5
  %0 = call ptr @llvm.aarch64.ldg(ptr nonnull %arrayidx, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret ptr %0
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x[[T1:[0-9]+]], [[T0]], #20
; CHECK-NEXT: ldg x[[T1]], [x[[T1]]]
}


; *********** __arm_mte_set_tag  *************
define void @set_tag1(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag1:
  tail call void @llvm.aarch64.stg(ptr %tag, ptr %ptr)
  ret void
; CHECK: stg x0, [x1]
}

define void @set_tag1stack(ptr %tag) {
entry:
; CHECK-LABEL: set_tag1stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  call void @llvm.aarch64.stg(ptr %tag, ptr nonnull %s)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %tag)
  ret void
; CHECK: stg x0, [sp]
}


define void @set_tag2(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag2:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 4
  tail call void @llvm.aarch64.stg(ptr %tag, ptr %add.ptr)
  ret void
; CHECK: stg x0, [x1, #16]
}

define void @set_tag2stack(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag2stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 4
  call void @llvm.aarch64.stg(ptr %tag, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret void
; CHECK: stg x0, [sp, #16]
}



define void @set_tag3(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag3:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 1020
  tail call void @llvm.aarch64.stg(ptr %tag, ptr %add.ptr)
  ret void
; CHECK: stg x0, [x1, #4080]
}

define void @set_tag3stack(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag3stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 1020
  call void @llvm.aarch64.stg(ptr %tag, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret void
; CHECK: stg x0, [sp, #4080]
}



define void @set_tag4(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag4:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 1024
  tail call void @llvm.aarch64.stg(ptr %tag, ptr %add.ptr)
  ret void
; CHECK: add x[[T0:[0-9]+]], x1, #1, lsl #12
; CHECK-NEXT: stg x0, [x[[T0]]]
}

define void @set_tag4stack(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag4stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 1024
  call void @llvm.aarch64.stg(ptr %tag, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret void
; CHECK: add x[[T0:[0-9]+]], {{.*}}, #1, lsl #12
; CHECK-NEXT: stg x0, [x[[T0]]]
}


define void @set_tag5(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag5:
  %add.ptr = getelementptr inbounds i32, ptr %ptr, i64 5
  tail call void @llvm.aarch64.stg(ptr %tag, ptr %add.ptr)
  ret void
; CHECK: add x[[T0:[0-9]+]], x1, #20
; CHECK-NEXT: stg x0, [x[[T0]]]
}

define void @set_tag5stack(ptr %tag, ptr %ptr) {
entry:
; CHECK-LABEL: set_tag5stack:
  %s = alloca %struct.S8K, align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr nonnull %s)
  %arrayidx = getelementptr inbounds %struct.S8K, ptr %s, i64 0, i32 0, i64 5
  call void @llvm.aarch64.stg(ptr %tag, ptr nonnull %arrayidx)
  call void @llvm.lifetime.end.p0(i64 8192, ptr nonnull %s)
  ret void
; CHECK: add x[[T0:[0-9]+]], {{.*}}, #20
; CHECK-NEXT: stg x0, [x[[T0]]]
}


; *********** __arm_mte_ptrdiff  *************
define i64 @subtract_pointers(ptr %ptra, ptr %ptrb) {
entry:
; CHECK-LABEL: subtract_pointers:
  %0 = tail call i64 @llvm.aarch64.subp(ptr %ptra, ptr %ptrb)
  ret i64 %0
; CHECK: subp x0, x0, x1
}

declare ptr @llvm.aarch64.irg(ptr, i64)
declare ptr @llvm.aarch64.addg(ptr, i64)
declare i64 @llvm.aarch64.gmi(ptr, i64)
declare ptr @llvm.aarch64.ldg(ptr, ptr)
declare void @llvm.aarch64.stg(ptr, ptr)
declare i64 @llvm.aarch64.subp(ptr, ptr)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
