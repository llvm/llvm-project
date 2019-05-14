; RUN: llc < %s -mtriple=arm64-eabi -mattr=+mte | FileCheck %s

; test create_tag
define i32* @create_tag(i32* %ptr, i32 %m) {
entry:
; CHECK-LABEL: create_tag:
  %0 = bitcast i32* %ptr to i8*
  %1 = zext i32 %m to i64
  %2 = tail call i8* @llvm.aarch64.irg(i8* %0, i64 %1)
  %3 = bitcast i8* %2 to i32*
  ret i32* %3
;CHECK: irg x0, x0, {{x[0-9]+}}
}

; *********** __arm_mte_increment_tag  *************
; test increment_tag1
define i32* @increment_tag1(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag1:
  %0 = bitcast i32* %ptr to i8*
  %1 = tail call i8* @llvm.aarch64.addg(i8* %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: addg x0, x0, #0, #7
}

%struct.S2K = type { [512 x i32] }
define i32* @increment_tag1stack(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag1stack:
  %s = alloca %struct.S2K, align 4
  %0 = bitcast %struct.S2K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %0)
  %1 = call i8* @llvm.aarch64.addg(i8* nonnull %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %0)
  ret i32* %2
; CHECK: addg x0, sp, #0, #7
}


define i32* @increment_tag2(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag2:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 4
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.addg(i8* nonnull %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: addg x0, x0, #16, #7
}

define i32* @increment_tag2stack(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag2stack:
  %s = alloca %struct.S2K, align 4
  %0 = bitcast %struct.S2K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S2K, %struct.S2K* %s, i64 0, i32 0, i64 4
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.addg(i8* nonnull %1, i64 7)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %0)
  ret i32* %3
; CHECK: addg x0, sp, #16, #7
}

define i32* @increment_tag3(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag3:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 252
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.addg(i8* nonnull %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: addg x0, x0, #1008, #7
}

define i32* @increment_tag3stack(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag3stack:
  %s = alloca %struct.S2K, align 4
  %0 = bitcast %struct.S2K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S2K, %struct.S2K* %s, i64 0, i32 0, i64 252
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.addg(i8* nonnull %1, i64 7)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %0)
  ret i32* %3
; CHECK: addg x0, sp, #1008, #7
}


define i32* @increment_tag4(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag4:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 256
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.addg(i8* nonnull %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add [[T0:x[0-9]+]], x0, #1024
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}

define i32* @increment_tag4stack(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag4stack:
  %s = alloca %struct.S2K, align 4
  %0 = bitcast %struct.S2K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S2K, %struct.S2K* %s, i64 0, i32 0, i64 256
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.addg(i8* nonnull %1, i64 7)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %0)
  ret i32* %3
; CHECK: add [[T0:x[0-9]+]], {{.*}}, #1024
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}


define i32* @increment_tag5(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag5:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 5
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.addg(i8* nonnull %0, i64 7)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add [[T0:x[0-9]+]], x0, #20
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}

define i32* @increment_tag5stack(i32* %ptr) {
entry:
; CHECK-LABEL: increment_tag5stack:
  %s = alloca %struct.S2K, align 4
  %0 = bitcast %struct.S2K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S2K, %struct.S2K* %s, i64 0, i32 0, i64 5
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.addg(i8* nonnull %1, i64 7)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %0)
  ret i32* %3
; CHECK: add [[T0:x[0-9]+]], {{.*}}, #20
; CHECK-NEXT: addg x0, [[T0]], #0, #7
}


; *********** __arm_mte_exclude_tag  *************
; test exclude_tag
define i32 @exclude_tag(i32* %ptr, i32 %m) local_unnamed_addr #0 {
entry:
;CHECK-LABEL: exclude_tag:
  %0 = zext i32 %m to i64
  %1 = bitcast i32* %ptr to i8*
  %2 = tail call i64 @llvm.aarch64.gmi(i8* %1, i64 %0)
  %conv = trunc i64 %2 to i32
  ret i32 %conv
; CHECK: gmi	x0, x0, {{x[0-9]+}}
}


; *********** __arm_mte_get_tag *************
%struct.S8K = type { [2048 x i32] }
define i32* @get_tag1(i32* %ptr) {
entry:
; CHECK-LABEL: get_tag1:
  %0 = bitcast i32* %ptr to i8*
  %1 = tail call i8* @llvm.aarch64.ldg(i8* %0, i8* %0)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK ldg x0, [x0]
}

define i32* @get_tag1_two_parm(i32* %ret_ptr, i32* %ptr) {
entry:
; CHECK-LABEL: get_tag1_two_parm:
  %0 = bitcast i32* %ret_ptr to i8*
  %1 = bitcast i32* %ptr to i8*
  %2 = tail call i8* @llvm.aarch64.ldg(i8* %0, i8* %1)
  %3 = bitcast i8* %2 to i32*
  ret i32* %3
; CHECK ldg x0, [x1]
}

define i32* @get_tag1stack() {
entry:
; CHECK-LABEL: get_tag1stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %1 = call i8* @llvm.aarch64.ldg(i8* nonnull %0, i8* nonnull %0)
  %2 = bitcast i8* %1 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %2
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: ldg [[T0]], [sp]
}

define i32* @get_tag1stack_two_param(i32* %ret_ptr) {
entry:
; CHECK-LABEL: get_tag1stack_two_param:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  %1 = bitcast i32*  %ret_ptr to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %2 = call i8* @llvm.aarch64.ldg(i8* nonnull %1, i8* nonnull %0)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %3
; CHECK-NOT: mov {{.*}}, sp
; CHECK: ldg x0, [sp]
}


define i32* @get_tag2(i32* %ptr) {
entry:
; CHECK-LABEL: get_tag2:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 4
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.ldg(i8* nonnull %0, i8* nonnull %0)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add  [[T0:x[0-9]+]], x0, #16
; CHECK: ldg  [[T0]], [x0, #16]
}

define i32* @get_tag2stack() {
entry:
; CHECK-LABEL: get_tag2stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 4
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.ldg(i8* nonnull %1, i8* nonnull %1)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %3
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x0, [[T0]], #16
; CHECK: ldg x0, [sp, #16]
}


define i32* @get_tag3(i32* %ptr) {
entry:
; CHECK-LABEL: get_tag3:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 1020
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.ldg(i8* nonnull %0, i8* nonnull %0)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add [[T0:x[0-8]+]], x0, #4080
; CHECK: ldg [[T0]], [x0, #4080]
}

define i32* @get_tag3stack() {
entry:
; CHECK-LABEL: get_tag3stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 1020
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.ldg(i8* nonnull %1, i8* nonnull %1)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %3
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x0, [[T0]], #4080
; CHECK: ldg x0, [sp, #4080]
}


define i32* @get_tag4(i32* %ptr) {
entry:
; CHECK-LABEL: get_tag4:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 1024
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.ldg(i8* nonnull %0, i8* nonnull %0)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add x0, x0, #1, lsl #12
; CHECK-NEXT: ldg x0, [x0]
}

define i32* @get_tag4stack() {
entry:
; CHECK-LABEL: get_tag4stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 1024
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.ldg(i8* nonnull %1, i8* nonnull %1)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %3
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK-NEXT: add x[[T1:[0-9]+]], [[T0]], #1, lsl #12
; CHECK-NEXT: ldg x[[T1]], [x[[T1]]]
}

define i32* @get_tag5(i32* %ptr) {
entry:
; CHECK-LABEL: get_tag5:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 5
  %0 = bitcast i32* %add.ptr to i8*
  %1 = tail call i8* @llvm.aarch64.ldg(i8* nonnull %0, i8* nonnull %0)
  %2 = bitcast i8* %1 to i32*
  ret i32* %2
; CHECK: add x0, x0, #20
; CHECK-NEXT: ldg x0, [x0]
}

define i32* @get_tag5stack() {
entry:
; CHECK-LABEL: get_tag5stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 5
  %1 = bitcast i32* %arrayidx to i8*
  %2 = call i8* @llvm.aarch64.ldg(i8* nonnull %1, i8* nonnull %1)
  %3 = bitcast i8* %2 to i32*
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret i32* %3
; CHECK: mov [[T0:x[0-9]+]], sp
; CHECK: add x[[T1:[0-9]+]], [[T0]], #20
; CHECK-NEXT: ldg x[[T1]], [x[[T1]]]
}


; *********** __arm_mte_set_tag  *************
define void @set_tag1(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag1:
  %0 = bitcast i32* %tag to i8*
  %1 = bitcast i32* %ptr to i8*
  tail call void @llvm.aarch64.stg(i8* %0, i8* %1)
  ret void
; CHECK: stg x0, [x1]
}

define void @set_tag1stack(i32* %tag) {
entry:
; CHECK-LABEL: set_tag1stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast i32* %tag to i8*
  %1 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %1)
  call void @llvm.aarch64.stg(i8* %0, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret void
; CHECK: stg x0, [sp]
}


define void @set_tag2(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag2:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 4
  %0 = bitcast i32* %tag to i8*
  %1 = bitcast i32* %add.ptr to i8*
  tail call void @llvm.aarch64.stg(i8* %0, i8* %1)
  ret void
; CHECK: stg x0, [x1, #16]
}

define void @set_tag2stack(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag2stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 4
  %1 = bitcast i32* %arrayidx to i8*
  %2 = bitcast i32* %tag to i8*
  call void @llvm.aarch64.stg(i8* %2, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret void
; CHECK: stg x0, [sp, #16]
}



define void @set_tag3(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag3:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 1020
  %0 = bitcast i32* %add.ptr to i8*
  %1 = bitcast i32* %tag to i8*
  tail call void @llvm.aarch64.stg(i8* %1, i8* %0)
  ret void
; CHECK: stg x0, [x1, #4080]
}

define void @set_tag3stack(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag3stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 1020
  %1 = bitcast i32* %arrayidx to i8*
  %2 = bitcast i32* %tag to i8*
  call void @llvm.aarch64.stg(i8* %2, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret void
; CHECK: stg x0, [sp, #4080]
}



define void @set_tag4(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag4:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 1024
  %0 = bitcast i32* %add.ptr to i8*
  %1 = bitcast i32* %tag to i8*
  tail call void @llvm.aarch64.stg(i8* %1, i8* %0)
  ret void
; CHECK: add x[[T0:[0-9]+]], x1, #1, lsl #12
; CHECK-NEXT: stg x0, [x[[T0]]]
}

define void @set_tag4stack(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag4stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 1024
  %1 = bitcast i32* %arrayidx to i8*
  %2 = bitcast i32* %tag to i8*
  call void @llvm.aarch64.stg(i8* %2, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret void
; CHECK: add x[[T0:[0-9]+]], {{.*}}, #1, lsl #12
; CHECK-NEXT: stg x0, [x[[T0]]]
}


define void @set_tag5(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag5:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 5
  %0 = bitcast i32* %add.ptr to i8*
  %1 = bitcast i32* %tag to i8*
  tail call void @llvm.aarch64.stg(i8* %1, i8* %0)
  ret void
; CHECK: add x[[T0:[0-9]+]], x1, #20
; CHECK-NEXT: stg x0, [x[[T0]]]
}

define void @set_tag5stack(i32* %tag, i32* %ptr) {
entry:
; CHECK-LABEL: set_tag5stack:
  %s = alloca %struct.S8K, align 4
  %0 = bitcast %struct.S8K* %s to i8*
  call void @llvm.lifetime.start.p0i8(i64 8192, i8* nonnull %0)
  %arrayidx = getelementptr inbounds %struct.S8K, %struct.S8K* %s, i64 0, i32 0, i64 5
  %1 = bitcast i32* %arrayidx to i8*
  %2 = bitcast i32* %tag to i8*
  call void @llvm.aarch64.stg(i8* %2, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 8192, i8* nonnull %0)
  ret void
; CHECK: add x[[T0:[0-9]+]], {{.*}}, #20
; CHECK-NEXT: stg x0, [x[[T0]]]
}


; *********** __arm_mte_ptrdiff  *************
define i64 @subtract_pointers(i32* %ptra, i32* %ptrb) {
entry:
; CHECK-LABEL: subtract_pointers:
  %0 = bitcast i32* %ptra to i8*
  %1 = bitcast i32* %ptrb to i8*
  %2 = tail call i64 @llvm.aarch64.subp(i8* %0, i8* %1)
  ret i64 %2
; CHECK: subp x0, x0, x1
}

declare i8* @llvm.aarch64.irg(i8*, i64)
declare i8* @llvm.aarch64.addg(i8*, i64)
declare i64 @llvm.aarch64.gmi(i8*, i64)
declare i8* @llvm.aarch64.ldg(i8*, i8*)
declare void @llvm.aarch64.stg(i8*, i8*)
declare i64 @llvm.aarch64.subp(i8*, i8*)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
