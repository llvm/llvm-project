; RUN: opt -S -passes=early-cse -earlycse-debug-hash < %s | FileCheck %s
; RUN: opt -S -passes=gvn < %s | FileCheck %s
; RUN: opt -S -passes=newgvn < %s | FileCheck %s

; These tests checks if passes with CSE functionality can do CSE on
; launder.invariant.group, that is prohibited if there is a memory clobber
; between barriers call.

; CHECK-LABEL: define i8 @optimizable()
define i8 @optimizable() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
; CHECK: call ptr @llvm.launder.invariant.group.p0
    %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; FIXME: This one could be CSE
; CHECK: call ptr @llvm.launder.invariant.group
    %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: call void @clobber(ptr {{.*}}%ptr)
    call void @clobber(ptr %ptr)

; CHECK: call void @use(ptr {{.*}}%ptr2)
    call void @use(ptr %ptr2)
; CHECK: call void @use(ptr {{.*}}%ptr3)
    call void @use(ptr %ptr3)
; CHECK: load i8, ptr %ptr3, {{.*}}!invariant.group
    %v = load i8, ptr %ptr3, !invariant.group !0

    ret i8 %v
}

; CHECK-LABEL: define i8 @unoptimizable()
define i8 @unoptimizable() {
entry:
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
; CHECK: call ptr @llvm.launder.invariant.group.p0
    %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
    call void @clobber(ptr %ptr)
; CHECK: call ptr @llvm.launder.invariant.group.p0
    %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: call void @clobber(ptr {{.*}}%ptr)
    call void @clobber(ptr %ptr)
; CHECK: call void @use(ptr {{.*}}%ptr2)
    call void @use(ptr %ptr2)
; CHECK: call void @use(ptr {{.*}}%ptr3)
    call void @use(ptr %ptr3)
; CHECK: load i8, ptr %ptr3, {{.*}}!invariant.group
    %v = load i8, ptr %ptr3, !invariant.group !0

    ret i8 %v
}

; CHECK-LABEL: define i8 @unoptimizable2()
define i8 @unoptimizable2() {
    %ptr = alloca i8
    store i8 42, ptr %ptr, !invariant.group !0
; CHECK: call ptr @llvm.launder.invariant.group
    %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
    store i8 43, ptr %ptr
; CHECK: call ptr @llvm.launder.invariant.group
    %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: call void @clobber(ptr {{.*}}%ptr)
    call void @clobber(ptr %ptr)
; CHECK: call void @use(ptr {{.*}}%ptr2)
    call void @use(ptr %ptr2)
; CHECK: call void @use(ptr {{.*}}%ptr3)
    call void @use(ptr %ptr3)
; CHECK: load i8, ptr %ptr3, {{.*}}!invariant.group
    %v = load i8, ptr %ptr3, !invariant.group !0
    ret i8 %v
}

; This test check if optimizer is not proving equality based on mustalias
; CHECK-LABEL: define void @dontProveEquality(ptr %a)
define void @dontProveEquality(ptr %a) {
  %b = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %r = icmp eq ptr %b, %a
; CHECK: call void @useBool(i1 %r)
  call void @useBool(i1 %r)

  %b2 = call ptr @llvm.strip.invariant.group.p0(ptr %a)
  %r2 = icmp eq ptr %b2, %a
; CHECK: call void @useBool(i1 %r2)
  call void @useBool(i1 %r2)

  ret void
}

declare void @use(ptr readonly)
declare void @useBool(i1)

declare void @clobber(ptr)
; CHECK: Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite){{$}}
; CHECK-NEXT: declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.launder.invariant.group.p0(ptr)

; CHECK: Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none){{$}}
; CHECK-NEXT: declare ptr @llvm.strip.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)


!0 = !{}
