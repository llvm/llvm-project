; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

%Iter = type { i32, i32, i32 }

%frame.reverse = type { %Iter, %Iter }

declare i32 @pers(...)
declare void @llvm.stackrestore(ptr)
declare ptr @llvm.stacksave()
declare void @begin(ptr sret(%Iter))
declare void @plus(ptr sret(%Iter), ptr, i32)
declare void @reverse(ptr inalloca(%frame.reverse) align 4)

define i32 @main() personality ptr @pers {
  %temp.lvalue = alloca %Iter
  br label %blah

blah:
  %inalloca.save = call ptr @llvm.stacksave()
  %rev_args = alloca inalloca %frame.reverse, align 4
  %end = getelementptr %frame.reverse, ptr %rev_args, i32 0, i32 1

; CHECK:  pushl   %eax
; CHECK:  subl    $20, %esp
; CHECK:  movl %esp, %[[beg:[^ ]*]]

  call void @begin(ptr sret(%Iter) %temp.lvalue)
; CHECK:  calll _begin

  invoke void @plus(ptr sret(%Iter) %end, ptr %temp.lvalue, i32 4)
          to label %invoke.cont unwind label %lpad

;  Uses end as sret param.
; CHECK:  leal 12(%[[beg]]), %[[end:[^ ]*]]
; CHECK:  pushl %[[end]]
; CHECK:  calll _plus

invoke.cont:
  call void @begin(ptr sret(%Iter) %rev_args)

; CHECK:  pushl %[[beg]]
; CHECK:  calll _begin

  invoke void @reverse(ptr inalloca(%frame.reverse) align 4 %rev_args)
          to label %invoke.cont5 unwind label %lpad

invoke.cont5:                                     ; preds = %invoke.cont
  call void @llvm.stackrestore(ptr %inalloca.save)
  ret i32 0

lpad:                                             ; preds = %invoke.cont, %entry
  %lp = landingpad { ptr, i32 }
          cleanup
  unreachable
}
