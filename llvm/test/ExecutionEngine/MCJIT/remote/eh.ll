; REQUIRES: cxx-shared-library
; RUN: %lli -jit-kind=mcjit -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s
; XFAIL: target=arm{{.*}}, target={{.*-(cygwin|windows-msvc|windows-gnu)}}
; UNSUPPORTED: powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed
; FIXME: Something hangs here.
; UNSUPPORTED: use_msan_with_origins
declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)
declare i32 @__gxx_personality_v0(...)
declare void @__cxa_end_catch()
declare ptr @__cxa_begin_catch(ptr)

@_ZTIi = external constant ptr

define void @throwException() {
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
  unreachable
}

define i32 @main() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @throwException()
          to label %try.cont unwind label %lpad

lpad:
  %p = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %e = extractvalue { ptr, i32 } %p, 0
  call ptr @__cxa_begin_catch(ptr %e)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0
}
