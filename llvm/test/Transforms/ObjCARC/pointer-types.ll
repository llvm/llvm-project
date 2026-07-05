; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

; Don't hoist @llvm.objc.release past a use of its pointer, even
; if the use has function type, because clang uses function types
; in dubious ways.
; rdar://10551239

; CHECK-LABEL: define void @test0(
; CHECK: %otherBlock = phi ptr [ %b1, %if.then ], [ null, %entry ]
; CHECK-NEXT: call void @use_fptr(ptr %otherBlock)
; CHECK-NEXT: call void @llvm.objc.release(ptr %otherBlock)

define void @test0(i1 %tobool, ptr %b1) {
entry:
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %otherBlock = phi ptr [ %b1, %if.then ], [ null, %entry ]
  call void @use_fptr(ptr %otherBlock)
  call void @llvm.objc.release(ptr %otherBlock) nounwind
  ret void
}

declare void @use_fptr(ptr)
declare void @llvm.objc.release(ptr)

