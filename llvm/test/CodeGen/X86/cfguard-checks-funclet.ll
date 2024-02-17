; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s -check-prefixes=X64
; Control Flow Guard is currently only available on Windows

; funclets only supported in MSVC env.

; Test that Control Flow Guard Checks are added well for targets in try-catch.


declare i32 @target_func()


%eh.ThrowInfo = type { i32, ptr, ptr, ptr }

declare i32 @__CxxFrameHandler3(...)
declare void @_CxxThrowException(ptr, ptr)

define i32 @func_cf_exception() personality ptr @__CxxFrameHandler3 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  invoke void @_CxxThrowException(ptr null, ptr null) #11
          to label %unreachable unwind label %ehcleanup

ehcleanup:
  %0 = cleanuppad within none []
  %isnull = icmp eq ptr %func_ptr, null
  br i1 %isnull, label %exit, label %callfn

callfn:
  %1 = load ptr, ptr %func_ptr, align 8
  %2 = call i32 %1() #9 [ "funclet"(token %0) ]
  br label %exit

exit:
  cleanupret from %0 unwind label %catch.dispatch

unreachable:
  unreachable

catch.dispatch:
  %3 = catchswitch within none [label %catch] unwind to caller

catch:
  %4 = catchpad within %3 [ptr null, i32 64, ptr null]
  catchret from %4 to label %try.cont

try.cont:
  ret i32 0

  ; X86-LABEL: func_cf_exception
  ; X86:         calll *___guard_check_icall_fptr
  ; X86-NEXT:    calll *%ecx

  ; X64-LABEL: func_cf_exception
  ; X64:       callq *__guard_dispatch_icall_fptr(%rip)
  ; X64-NOT:   callq
}


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
