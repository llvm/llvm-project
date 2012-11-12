; RUN: opt -objc-arc -S < %s | FileCheck %s

; rdar://10803830
; The optimizer should be able to prove that the block does not
; "escape", so the retainBlock+release pair can be eliminated.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.__block_descriptor = type { i64, i64 }

@_NSConcreteStackBlock = external global i8*
@__block_descriptor_tmp = external global { i64, i64, i8*, i8* }

; CHECK: define void @test() {
; CHECK-NOT: @objc
; CHECK: declare i8* @objc_retainBlock(i8*)
; CHECK: declare void @objc_release(i8*)

define void @test() {
entry:
  %block = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>, align 8
  %block.isa = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 0
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %block.isa, align 8
  %block.flags = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 1
  store i32 1073741824, i32* %block.flags, align 8
  %block.reserved = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 2
  store i32 0, i32* %block.reserved, align 4
  %block.invoke = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 3
  store i8* bitcast (i32 (i8*)* @__test_block_invoke_0 to i8*), i8** %block.invoke, align 8
  %block.descriptor = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 4
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %block.descriptor, align 8
  %block.captured = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block, i64 0, i32 5
  store i32 4, i32* %block.captured, align 8
  %tmp = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32 }>* %block to i8*
  %tmp1 = call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  %tmp2 = getelementptr inbounds i8* %tmp1, i64 16
  %tmp3 = bitcast i8* %tmp2 to i8**
  %tmp4 = load i8** %tmp3, align 8
  %tmp5 = bitcast i8* %tmp4 to i32 (i8*)*
  %call = call i32 %tmp5(i8* %tmp1)
  call void @objc_release(i8* %tmp1) nounwind, !clang.imprecise_release !0
  ret void
}

declare i32 @__test_block_invoke_0(i8* nocapture %.block_descriptor) nounwind readonly

declare i8* @objc_retainBlock(i8*)

declare void @objc_release(i8*)

!0 = metadata !{}
