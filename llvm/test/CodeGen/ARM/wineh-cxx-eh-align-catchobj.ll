; RUN: not llc -mtriple=thumbv7-pc-windows-msvc < %s -o /dev/null 2>&1 | FileCheck %s

; The frame is not realigned, so an over-aligned catch object or local passed to
; llvm.localescape is an error.

; CHECK: error: {{.*}}in function catch_overaligned{{.*}}Windows C++ exception handling on ARM: the catch object is 16-byte aligned, but the stack is only 8-byte aligned
; CHECK: error: {{.*}}in function esc{{.*}}Windows C++ exception handling on ARM: a local passed to llvm.localescape is 16-byte aligned, but the stack is only 8-byte aligned

%struct.A = type { [4 x i32] }
@ti = external global i8
declare void @g()
declare void @use(ptr)
declare i32 @__CxxFrameHandler3(...)

define void @catch_overaligned() personality ptr @__CxxFrameHandler3 {
entry:
  %e = alloca %struct.A, align 16
  invoke void @g() to label %ret unwind label %dispatch
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr @ti, i32 8, ptr %e]
  call void @use(ptr %e) [ "funclet"(token %cp) ]
  catchret from %cp to label %ret
ret:
  ret void
}

declare void @llvm.localescape(...)
define void @esc() personality ptr @__CxxFrameHandler3 {
entry:
  %a = alloca [16 x i8], align 16
  call void (...) @llvm.localescape(ptr %a)
  invoke void @g() to label %cont unwind label %cleanup
cont:
  ret void
cleanup:
  %p = cleanuppad within none []
  call void @use(ptr null) [ "funclet"(token %p) ]
  cleanupret from %p unwind to caller
}
