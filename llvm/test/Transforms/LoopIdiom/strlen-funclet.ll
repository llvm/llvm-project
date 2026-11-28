; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s

; A strlen formed in a Windows EH funclet needs the funclet bundle, or
; WinEHPrepare removes the call and makes the rest of the funclet
; unreachable.

target triple = "x86_64-pc-windows-msvc"

declare void @thrower()
declare ptr @what()
declare i32 @__CxxFrameHandler3(...)

define i64 @in_catch() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: @in_catch(
; CHECK:         [[CP:%.*]] = catchpad within
; CHECK:         call i64 @strlen(ptr {{.*}}) [ "funclet"(token [[CP]]) ]
entry:
  invoke void @thrower() to label %done unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  %p = call ptr @what() [ "funclet"(token %cp) ]
  br label %loop

loop:
  %n = phi i64 [ 0, %catch ], [ %n.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %n
  %c = load i8, ptr %gep, align 1
  %nz = icmp ne i8 %c, 0
  %n.next = add nsw i64 %n, 1
  br i1 %nz, label %loop, label %exit

exit:
  catchret from %cp to label %caught

caught:
  ret i64 %n

done:
  ret i64 0
}

; Outside of a funclet the call gets no bundle.
define i64 @in_parent(ptr %p) personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: @in_parent(
; CHECK:         call i64 @strlen(ptr %p){{$}}
entry:
  br label %loop

loop:
  %n = phi i64 [ 0, %entry ], [ %n.next, %loop ]
  %gep = getelementptr inbounds i8, ptr %p, i64 %n
  %c = load i8, ptr %gep, align 1
  %nz = icmp ne i8 %c, 0
  %n.next = add nsw i64 %n, 1
  br i1 %nz, label %loop, label %exit

exit:
  invoke void @thrower() to label %done unwind label %dispatch

dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %done

done:
  ret i64 %n
}
