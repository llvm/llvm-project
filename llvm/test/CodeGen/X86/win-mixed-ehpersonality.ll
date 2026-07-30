; RUN: llc -mtriple x86_64-pc-windows-msvc < %s | FileCheck %s

declare void @maybe_throw()

@_ZTIi = external constant ptr
@g = external global i32

declare i32 @__C_specific_handler(...)
declare i32 @__gxx_personality_seh0(...)
declare i32 @llvm.eh.typeid.for(ptr) readnone nounwind

define i32 @use_seh() personality ptr @__C_specific_handler {
entry:
  invoke void @maybe_throw()
      to label %cont unwind label %lpad

cont:
  ret i32 0

lpad:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %cs [ptr @filt_g]
  catchret from %p to label %ret1

ret1:
  ret i32 1
}

define internal i32 @filt_g(ptr, ptr) {
  %g = load i32, ptr @g
  ret i32 %g
}

; CHECK-LABEL: use_seh:
; CHECK: callq maybe_throw
; CHECK: xorl %eax, %eax
; CHECK: .LBB0_[[epilogue:[0-9]+]]
; CHECK: retq
; CHECK: # %catch{{$}}
; CHECK: movl $1, %eax
; CHECK: jmp .LBB0_[[epilogue]]

; A MinGW64-ish EH style. It could happen if a binary uses both MSVC CRT and
; mingw CRT and is linked with LTO.
define i32 @use_gcc() personality ptr @__gxx_personality_seh0 {
entry:
  invoke void @maybe_throw()
      to label %cont unwind label %lpad

cont:
  ret i32 0

lpad:
  %ehvals = landingpad { ptr, i32 }
      cleanup
      catch ptr @_ZTIi
  %ehsel = extractvalue { ptr, i32 } %ehvals, 1
  %filt_g_sel = call i32 @llvm.eh.typeid.for(ptr @filt_g)
  %matches = icmp eq i32 %ehsel, %filt_g_sel
  br i1 %matches, label %ret1, label %eh.resume

ret1:
  ret i32 1

eh.resume:
  resume { ptr, i32 } %ehvals
}

; CHECK-LABEL: use_gcc:
; CHECK: callq maybe_throw
; CHECK: xorl %eax, %eax
;
; CHECK: # %lpad
; CHECK: cmpl $2, %edx
; CHECK: jne
;
; CHECK: # %ret1
; CHECK: movl $1, %eax
;
; CHECK: callq _Unwind_Resume
