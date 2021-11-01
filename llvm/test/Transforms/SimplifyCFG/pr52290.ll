; RUN: opt < %s -simplifycfg -S | FileCheck %s
; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; XFAIL: *
; REQUIRES: asserts
; FIXME: Fails due to infinite loop in iterativelySimplifyCFG.

; ModuleID = 'test/Transforms/SimplifyCFG/pr-new.ll'
source_filename = "test/Transforms/SimplifyCFG/pr-new.ll"

define i32 @test(float %arg) gc "statepoint-example" personality i32* ()* @blam {
; CHECK-LABEL: @test
bb:
  %tmp = call i1 @llvm.experimental.widenable.condition()
  br i1 %tmp, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb7, label %bb5

bb2:                                              ; preds = %bb
  %tmp3 = getelementptr inbounds i8, i8 addrspace(1)* undef, i64 16
  br i1 undef, label %bb6, label %bb4

bb4:                                              ; preds = %bb2
  call void @snork() [ "deopt"() ]
  unreachable

bb5:                                              ; preds = %bb1
  ret i32 0

bb6:                                              ; preds = %bb2
  br label %bb7

bb7:                                              ; preds = %bb6, %bb1
  %tmp8 = call i32 (...) @llvm.experimental.deoptimize.i32(i32 10) [ "deopt"() ]
  ret i32 %tmp8
}

declare i32* @blam()

declare void @snork()

declare i32 @llvm.experimental.deoptimize.i32(...)

; Function Attrs: inaccessiblememonly nofree nosync nounwind speculatable willreturn
declare i1 @llvm.experimental.widenable.condition() #0

attributes #0 = { inaccessiblememonly nofree nosync nounwind speculatable willreturn }

