; RUN: opt -S -globalopt < %s | FileCheck %s

; This test is hint, what could globalOpt optimize and what it can't
; FIXME: @tmp and @tmp2 can be safely set to 42
; CHECK: @tmp = local_unnamed_addr global i32 0
; CHECK: @tmp2 = local_unnamed_addr global i32 0
; CHECK: @tmp3 = global i32 0

@tmp = global i32 0
@tmp2 = global i32 0
@tmp3 = global i32 0
@ptrToTmp3 = global i32* null

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

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
  store i32 %val, i32* @tmp
  store i32 %val, i32* %valptr
  
  %0 = bitcast i32* %valptr to i8*
  %barr = call i8* @llvm.invariant.group.barrier(i8* %0)
  %1 = bitcast i8* %barr to i32*
  
  %val2 = load i32, i32* %1
  store i32 %val2, i32* @tmp2
  ret void
}

; We can't step through invariant.group.barrier here, because that would change
; this load in @usage_of_globals()
; val = load i32, i32* %ptrVal, !invariant.group !0 
; into 
; %val = load i32, i32* @tmp3, !invariant.group !0
; and then we could assume that %val and %val2 to be the same, which coud be 
; false, because @changeTmp3ValAndCallBarrierInside() may change the value
; of @tmp3.
define void @_not_optimizable() {
enter:
  store i32 13, i32* @tmp3, !invariant.group !0
  
  %0 = bitcast i32* @tmp3 to i8*
  %barr = call i8* @llvm.invariant.group.barrier(i8* %0)
  %1 = bitcast i8* %barr to i32*
  
  store i32* %1, i32** @ptrToTmp3
  store i32 42, i32* %1, !invariant.group !0
  
  ret void
}
define void @usage_of_globals() {
entry:
  %ptrVal = load i32*, i32** @ptrToTmp3
  %val = load i32, i32* %ptrVal, !invariant.group !0
  
  call void @changeTmp3ValAndCallBarrierInside()
  %val2 = load i32, i32* @tmp3, !invariant.group !0
  ret void;
}

declare void @changeTmp3ValAndCallBarrierInside()

declare i8* @llvm.invariant.group.barrier(i8*)

!0 = !{!"something"}
