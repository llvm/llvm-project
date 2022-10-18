; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

; Don't hoist @llvm.objc.release past a use of its pointer, even
; if the use has function type, because clang uses function types
; in dubious ways.
; rdar://10551239

; CHECK-LABEL: define void @test0(
; CHECK: %otherBlock = phi void ()* [ %b1, %if.then ], [ null, %entry ]
; CHECK-NEXT: call void @use_fptr(void ()* %otherBlock)
; CHECK-NEXT: %tmp11 = bitcast void ()* %otherBlock to i8*
; CHECK-NEXT: call void @llvm.objc.release(i8* %tmp11)

define void @test0(i1 %tobool, void ()* %b1) {
entry:
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %otherBlock = phi void ()* [ %b1, %if.then ], [ null, %entry ]
  call void @use_fptr(void ()* %otherBlock)
  %tmp11 = bitcast void ()* %otherBlock to i8*
  call void @llvm.objc.release(i8* %tmp11) nounwind
  ret void
}

declare void @use_fptr(void ()*)
declare void @llvm.objc.release(i8*)

