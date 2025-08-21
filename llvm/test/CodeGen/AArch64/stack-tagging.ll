; RUN: opt < %s -aarch64-stack-tagging -S -o - | FileCheck %s --check-prefixes=CHECK,SSI
; RUN: opt < %s -aarch64-stack-tagging -stack-tagging-use-stack-safety=0 -S -o - | FileCheck %s --check-prefixes=CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use8(ptr)
declare void @use32(ptr)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

define dso_local void @noUse32(ptr) sanitize_memtag {
entry:
  ret void
}

define void @OneVar() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @use32(ptr %x)
  ret void
}

; CHECK-LABEL: define void @OneVar(
; CHECK:  [[BASE:%.*]] = call ptr @llvm.aarch64.irg.sp(i64 0)
; CHECK:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; CHECK:  [[TX:%.*]] = call ptr @llvm.aarch64.tagp.{{.*}}(ptr [[X]], ptr [[BASE]], i64 0)
; CHECK:  call void @llvm.aarch64.settag(ptr [[TX]], i64 16)
; CHECK:  call void @use32(ptr [[TX]])
; CHECK:  call void @llvm.aarch64.settag(ptr [[X]], i64 16)
; CHECK:  ret void


define void @ManyVars() sanitize_memtag {
entry:
  %x1 = alloca i32, align 4
  %x2 = alloca i8, align 4
  %x3 = alloca i32, i32 11, align 4
  %x4 = alloca i32, align 4
  call void @use32(ptr %x1)
  call void @use8(ptr %x2)
  call void @use32(ptr %x3)
  ret void
}

; CHECK-LABEL: define void @ManyVars(
; CHECK:  alloca { i32, [12 x i8] }, align 16
; CHECK:  call ptr @llvm.aarch64.tagp.{{.*}}(ptr {{.*}}, i64 0)
; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK:  alloca { i8, [15 x i8] }, align 16
; CHECK:  call ptr @llvm.aarch64.tagp.{{.*}}(ptr {{.*}}, i64 1)
; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK:  alloca { [11 x i32], [4 x i8] }, align 16
; CHECK:  call ptr @llvm.aarch64.tagp.{{.*}}(ptr {{.*}}, i64 2)
; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 48)
; CHECK:  alloca i32, align 4
; SSI-NOT: @llvm.aarch64.tagp
; SSI-NOT: @llvm.aarch64.settag

; CHECK:  call void @use32(
; CHECK:  call void @use8(
; CHECK:  call void @use32(

; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK:  call void @llvm.aarch64.settag(ptr {{.*}}, i64 48)
; CHECK-NEXT:  ret void


define void @Scope(i32 %b) sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %x)
  call void @use8(ptr %x) #3
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %x)
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: define void @Scope(
; CHECK:  br i1
; CHECK:  call void @llvm.lifetime.start.p0(
; CHECK:  call void @llvm.aarch64.settag(
; CHECK:  call void @use8(
; CHECK:  call void @llvm.aarch64.settag(
; CHECK:  call void @llvm.lifetime.end.p0(
; CHECK:  br label
; CHECK:  ret void


; Spooked by the multiple lifetime ranges, StackTagging remove all of them and sets tags on entry and exit.
define void @BadScope(i32 %b) sanitize_memtag {
entry:
  %x = alloca i32, align 4
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %x)
  call void @use8(ptr %x) #3
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %x)

  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %x)
  call void @use8(ptr %x) #3
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %x)
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: define void @BadScope(
; CHECK:       call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK:       br i1
; CHECK:       call void @use8(ptr
; CHECK-NEXT:  call void @use8(ptr
; CHECK:       br label
; CHECK:       call void @llvm.aarch64.settag(ptr {{.*}}, i64 16)
; CHECK-NEXT:  ret void

define void @DynamicAllocas(i32 %cnt) sanitize_memtag {
entry:
  %x = alloca i32, i32 %cnt, align 4
  br label %l
l:
  %y = alloca i32, align 4
  call void @use32(ptr %x)
  call void @use32(ptr %y)
  ret void
}

; CHECK-LABEL: define void @DynamicAllocas(
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     %x = alloca i32, i32 %cnt, align 4
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     alloca i32, align 4
; CHECK-NOT: @llvm.aarch64.irg.sp
; CHECK:     ret void

!0 = !{}
