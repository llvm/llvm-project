; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s -check-prefixes=X64
; Control Flow Guard is currently only available on Windows

; funclets only supported in MSVC env.

; Test that Control Flow Guard Checks are added well for targets in try-catch.


declare i32 @target_func()


%eh.ThrowInfo = type { i32, i8*, i8*, i8* }

declare i32 @__CxxFrameHandler3(...)
declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

define i32 @func_cf_exception() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  invoke void @_CxxThrowException(i8* null, %eh.ThrowInfo* null) #11
          to label %unreachable unwind label %ehcleanup

ehcleanup:
  %0 = cleanuppad within none []
  %isnull = icmp eq i32 ()** %func_ptr, null
  br i1 %isnull, label %exit, label %callfn

callfn:
  %1 = load i32 ()*, i32 ()** %func_ptr, align 8
  %2 = call i32 %1() #9 [ "funclet"(token %0) ]
  br label %exit

exit:
  cleanupret from %0 unwind label %catch.dispatch

unreachable:
  unreachable

catch.dispatch:
  %3 = catchswitch within none [label %catch] unwind to caller

catch:
  %4 = catchpad within %3 [i8* null, i32 64, i8* null]
  catchret from %4 to label %try.cont

try.cont:
  ret i32 0

  ; X32-LABEL: func_cf_exception
	; X32: 	     calll *___guard_check_icall_fptr
	; X32-NEXT:  calll *%ecx

  ; X64-LABEL: func_cf_exception
  ; X64:       callq *__guard_dispatch_icall_fptr(%rip)
  ; X64-NOT:   callq
}


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
