; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Iter = type { i32, i32, i32 }

%frame.reverse = type { %Iter, %Iter }

declare i32 @pers(...)
declare void @llvm.stackrestore(i8*)
declare i8* @llvm.stacksave()
declare void @begin(%Iter* sret)
declare void @plus(%Iter* sret, %Iter*, i32)
declare void @reverse(%frame.reverse* inalloca align 4)

define i32 @main() personality i32 (...)* @pers {
  %temp.lvalue = alloca %Iter
  br label %blah

blah:
  %inalloca.save = call i8* @llvm.stacksave()
  %rev_args = alloca inalloca %frame.reverse, align 4
  %beg = getelementptr %frame.reverse, %frame.reverse* %rev_args, i32 0, i32 0
  %end = getelementptr %frame.reverse, %frame.reverse* %rev_args, i32 0, i32 1

; CHECK:  calll   __chkstk
; CHECK:  movl %esp, %[[beg:[^ ]*]]
; CHECK:  leal 12(%[[beg]]), %[[end:[^ ]*]]

  call void @begin(%Iter* sret %temp.lvalue)
; CHECK:  calll _begin

  invoke void @plus(%Iter* sret %end, %Iter* %temp.lvalue, i32 4)
          to label %invoke.cont unwind label %lpad

;  Uses end as sret param.
; CHECK:  pushl %[[end]]
; CHECK:  calll _plus

invoke.cont:
  call void @begin(%Iter* sret %beg)

; CHECK:  pushl %[[beg]]
; CHECK:  calll _begin

  invoke void @reverse(%frame.reverse* inalloca align 4 %rev_args)
          to label %invoke.cont5 unwind label %lpad

invoke.cont5:                                     ; preds = %invoke.cont
  call void @llvm.stackrestore(i8* %inalloca.save)
  ret i32 0

lpad:                                             ; preds = %invoke.cont, %entry
  %lp = landingpad { i8*, i32 }
          cleanup
  unreachable
}
