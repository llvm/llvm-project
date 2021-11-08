; RUN: llc < %s | FileCheck %s
target triple = "armv6kz-unknown-unknown-gnueabihf"

; Make sure this doesn't crash, and we actually emit a tail call.
; Unfortunately, this test is sort of fragile... the original issue only
; shows up if scheduling happens in a very specific order. But including
; it anyway just to demonstrate the issue.
; CHECK: pop {r{{[0-9]+}}, lr}

@e = external local_unnamed_addr constant [0 x i32 (i32, i32)*], align 4

; Function Attrs: nounwind sspstrong
define i32 @AVI_ChunkRead_p_chk(i32 %g) nounwind sspstrong "target-cpu"="arm1176jzf-s" {
entry:
  %b = alloca i8, align 1
  %tobool = icmp eq i32 %g, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %add = add nsw i32 %g, 1
  %arrayidx = getelementptr inbounds [0 x i32 (i32, i32)*], [0 x i32 (i32, i32)*]* @e, i32 0, i32 %add
  %0 = load i32 (i32, i32)*, i32 (i32, i32)** %arrayidx, align 4
  %call = tail call i32 %0(i32 0, i32 0) #3
  br label %return

if.end:                                           ; preds = %entry
  call void @c(i8* nonnull %b)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ 0, %if.end ]
  ret i32 %retval.0
}

declare void @c(i8*)
