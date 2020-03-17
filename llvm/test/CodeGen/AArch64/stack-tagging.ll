; RUN: opt < %s -stack-tagging -S -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use8(i8*)
declare void @use32(i32*)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define void @OneVar() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @use32(i32* %x)
  ret void
}

; CHECK-LABEL: define void @OneVar(
; CHECK:  [[BASE:%.*]] = call i8* @llvm.aarch64.irg.sp(i64 0)
; CHECK:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; CHECK:  [[TX:%.*]] = call { i32, [12 x i8] }* @llvm.aarch64.tagp.{{.*}}({ i32, [12 x i8] }* [[X]], i8* [[BASE]], i64 0)
; CHECK:  [[TX8:%.*]] = bitcast { i32, [12 x i8] }* [[TX]] to i8*
; CHECK:  call void @llvm.aarch64.settag(i8* [[TX8]], i64 16)
; CHECK:  [[GEP32:%.*]] = bitcast { i32, [12 x i8] }* [[TX]] to i32*
; CHECK:  call void @use32(i32* [[GEP32]])
; CHECK:  [[GEP8:%.*]] = bitcast { i32, [12 x i8] }* [[X]] to i8*
; CHECK:  call void @llvm.aarch64.settag(i8* [[GEP8]], i64 16)
; CHECK:  ret void


define void @ManyVars() sanitize_memtag {
entry:
  %x1 = alloca i32, align 4
  %x2 = alloca i8, align 4
  %x3 = alloca i32, i32 11, align 4
  %x4 = alloca i32, align 4, !stack-safe !0
  call void @use32(i32* %x1)
  call void @use8(i8* %x2)
  call void @use32(i32* %x3)
  ret void
}

; CHECK-LABEL: define void @ManyVars(
; CHECK:  alloca { i32, [12 x i8] }, align 16
; CHECK:  call { i32, [12 x i8] }* @llvm.aarch64.tagp.{{.*}}({ i32, [12 x i8] }* {{.*}}, i64 0)
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK:  alloca { i8, [15 x i8] }, align 16
; CHECK:  call { i8, [15 x i8] }* @llvm.aarch64.tagp.{{.*}}({ i8, [15 x i8] }* {{.*}}, i64 1)
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK:  alloca { [11 x i32], [4 x i8] }, align 16
; CHECK:  call { [11 x i32], [4 x i8] }* @llvm.aarch64.tagp.{{.*}}({ [11 x i32], [4 x i8] }* {{.*}}, i64 2)
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 48)
; CHECK:  alloca i32, align 4
; CHECK-NOT: @llvm.aarch64.tagp
; CHECK-NOT: @llvm.aarch64.settag

; CHECK:  call void @use32(
; CHECK:  call void @use8(
; CHECK:  call void @use32(

; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK:  call void @llvm.aarch64.settag(i8* {{.*}}, i64 48)
; CHECK-NEXT:  ret void


define void @Scope(i32 %b) sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  call void @use8(i8* %0) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: define void @Scope(
; CHECK:  br i1
; CHECK:  call void @llvm.lifetime.start.p0i8(
; CHECK:  call void @llvm.aarch64.settag(
; CHECK:  call void @use8(
; CHECK:  call void @llvm.aarch64.settag(
; CHECK:  call void @llvm.lifetime.end.p0i8(
; CHECK:  br label
; CHECK:  ret void


; Spooked by the multiple lifetime ranges, StackTagging remove all of them and sets tags on entry and exit.
define void @BadScope(i32 %b) sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  call void @use8(i8* %0) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)

  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  call void @use8(i8* %0) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: define void @BadScope(
; CHECK:       call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK:       br i1
; CHECK:       call void @use8(i8*
; CHECK-NEXT:  call void @use8(i8*
; CHECK:       br label
; CHECK:       call void @llvm.aarch64.settag(i8* {{.*}}, i64 16)
; CHECK-NEXT:  ret void

define void @DynamicAllocas(i32 %cnt) sanitize_memtag {
entry:
  %x = alloca i32, i32 %cnt, align 4
  br label %l
l:
  %y = alloca i32, align 4
  call void @use32(i32* %x)
  call void @use32(i32* %y)
  ret void
}

; CHECK-LABEL: define void @DynamicAllocas(
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     %x = alloca i32, i32 %cnt, align 4
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     alloca i32, align 4
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     ret void

; If we can't trace one of the lifetime markers to a single alloca, fall back
; to poisoning all allocas at the beginning of the function.
; Each alloca must be poisoned only once.
define void @UnrecognizedLifetime(i8 %v) sanitize_memtag {
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

; CHECK-LABEL: define void @UnrecognizedLifetime(
; CHECK: call i8* @llvm.aarch64.irg.sp(i64 0)
; CHECK: alloca { i32, [12 x i8] }, align 16
; CHECK: call { i32, [12 x i8] }* @llvm.aarch64.tagp
; CHECK: call void @llvm.aarch64.settag(
; CHECK: alloca { i32, [12 x i8] }, align 16
; CHECK: call { i32, [12 x i8] }* @llvm.aarch64.tagp
; CHECK: call void @llvm.aarch64.settag(
; CHECK: alloca { i32, [12 x i8] }, align 16
; CHECK: call { i32, [12 x i8] }* @llvm.aarch64.tagp
; CHECK: call void @llvm.aarch64.settag(
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: call void @llvm.aarch64.settag(
; CHECK: call void @llvm.aarch64.settag(
; CHECK: call void @llvm.aarch64.settag(
; CHECK: ret void

!0 = !{}