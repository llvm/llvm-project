; RUN: llc -mtriple=aarch64-linux-gnu -relocation-model=pic %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -relocation-model=pic %s -o - -fast-isel | FileCheck %s

; Test that -fno-plt (RtLibUseGOT module flag) routes all external function
; calls through GOT instead of PLT. This covers regular external functions,
; functions with NonLazyBind attribute, and C++ exception handling library
; functions (__cxa_allocate_exception, __cxa_begin_catch, _Unwind_Resume).

declare void @external_func()

define void @caller_external() nounwind {
; CHECK-LABEL: caller_external:
; CHECK:         adrp x8, :got:external_func
; CHECK:         ldr x8, [x8, :got_lo12:external_func]
; CHECK:         blr x8
  call void @external_func()
  ret void
}

; Function with NonLazyBind attribute — should also use GOT.
declare void @external_func_nonlazy() #0

define void @caller_nonlazy() nounwind {
; CHECK-LABEL: caller_nonlazy:
; CHECK:         adrp x8, :got:external_func_nonlazy
; CHECK:         ldr x8, [x8, :got_lo12:external_func_nonlazy]
; CHECK:         blr x8
  call void @external_func_nonlazy()
  ret void
}

; __cxa_allocate_exception — registered in RuntimeLibcalls.td, should use GOT.
declare ptr @__cxa_allocate_exception(i64)

define void @caller_throw() nounwind {
; CHECK-LABEL: caller_throw:
; CHECK:         adrp [[ADRP:x[0-9]+]], :got:__cxa_allocate_exception
; CHECK:         ldr [[LD:x[0-9]+]], [[[ADRP]], :got_lo12:__cxa_allocate_exception]
; CHECK:         blr [[LD]]
  %p = call ptr @__cxa_allocate_exception(i64 4)
  ret void
}

; __cxa_begin_catch — registered in RuntimeLibcalls.td, should use GOT.
declare ptr @__cxa_begin_catch(ptr)

define void @caller_catch() nounwind {
; CHECK-LABEL: caller_catch:
; CHECK:         adrp [[ADRP:x[0-9]+]], :got:__cxa_begin_catch
; CHECK:         ldr [[LD:x[0-9]+]], [[[ADRP]], :got_lo12:__cxa_begin_catch]
; CHECK:         blr [[LD]]
  %p = call ptr @__cxa_begin_catch(ptr null)
  ret void
}

; _Unwind_Resume — inserted by DwarfEHPrepare from a resume instruction.
; With -fno-plt, DwarfEHPrepare marks it NonLazyBind, and
; classifyGlobalFunctionReference routes it through GOT.
declare i32 @__gxx_personality_v0(...)
declare void @cleanup_func()

define void @caller_resume() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: caller_resume:
; CHECK:         adrp [[ADRP:x[0-9]+]], :got:_Unwind_Resume
; CHECK:         ldr [[LD:x[0-9]+]], [[[ADRP]], :got_lo12:_Unwind_Resume]
; CHECK:         blr [[LD]]
entry:
  invoke void @external_func() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lpi = landingpad { ptr, i32 } cleanup
  call void @cleanup_func()
  resume { ptr, i32 } %lpi
}

attributes #0 = { nonlazybind }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"RtLibUseGOT", i32 1}
