; RUN: opt -S -passes=globalopt < %s | FileCheck %s

; CHECK: @llvm.global_ctors = appending global [1 x {{.*}}@_GLOBAL__I_c
@llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_b, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_c, ptr null }]

; CHECK: @tmp = local_unnamed_addr global i32 42
; CHECK: @tmp2 = local_unnamed_addr global i32 42
; CHECK: @tmp3 = global i32 42
@tmp = global i32 0
@tmp2 = global i32 0
@tmp3 = global i32 0
@ptrToTmp3 = global ptr null

define i32 @TheAnswerToLifeTheUniverseAndEverything() {
  ret i32 42
}

define void @_GLOBAL__I_a() {
enter:
  call void @_optimizable()
  call void @_not_optimizable()
  ret void
}

define void @_optimizable() {
enter:
  %valptr = alloca i32

  %val = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  store i32 %val, ptr @tmp
  store i32 %val, ptr %valptr

  %barr = call ptr @llvm.launder.invariant.group(ptr %valptr)

  %val2 = load i32, ptr %barr
  store i32 %val2, ptr @tmp2
  ret void
}

; We can't step through launder.invariant.group here, because that would change
; this load in @usage_of_globals()
; %val = load i32, ptr %ptrVal, !invariant.group !0
; into
; %val = load i32, ptr @tmp3, !invariant.group !0
; and then we could assume that %val and %val2 to be the same, which coud be
; false, because @changeTmp3ValAndCallBarrierInside() may change the value
; of @tmp3.
define void @_not_optimizable() {
enter:
  store i32 13, ptr @tmp3, !invariant.group !0

  %barr = call ptr @llvm.launder.invariant.group(ptr @tmp3)

  store ptr %barr, ptr @ptrToTmp3
  store i32 42, ptr %barr, !invariant.group !0

  ret void
}

define void @usage_of_globals() {
entry:
  %ptrVal = load ptr, ptr @ptrToTmp3
  %val = load i32, ptr %ptrVal, !invariant.group !0

  call void @changeTmp3ValAndCallBarrierInside()
  %val2 = load i32, ptr @tmp3, !invariant.group !0
  ret void;
}

@tmp4 = global i32 0

define void @_GLOBAL__I_b() {
enter:
  %val = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  %p2 = call ptr @llvm.strip.invariant.group.p0(ptr @tmp4)
  store i32 %val, ptr %p2
  ret void
}

@tmp5 = global i32 0
@tmp6 = global ptr null
; CHECK: @tmp6 = local_unnamed_addr global ptr null

define ptr @_dont_return_param(ptr %p) {
  %p2 = call ptr @llvm.launder.invariant.group(ptr %p)
  ret ptr %p2
}

; We should bail out if we return any pointers derived via invariant.group intrinsics at any point.
define void @_GLOBAL__I_c() {
enter:
  %tmp5 = call ptr @_dont_return_param(ptr @tmp5)
  store ptr %tmp5, ptr @tmp6
  ret void
}


declare void @changeTmp3ValAndCallBarrierInside()

declare ptr @llvm.launder.invariant.group(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)

!0 = !{}
