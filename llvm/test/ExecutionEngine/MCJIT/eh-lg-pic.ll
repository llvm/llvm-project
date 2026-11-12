; REQUIRES: cxx-shared-library
; RUN: %lli -jit-kind=mcjit -relocation-model=pic -code-model=large %s
; XFAIL: target={{.*-(cygwin|windows-msvc|windows-gnu)}}
; XFAIL: target={{(mips|mipsel)-.*}}, target={{(i686|i386).*}}, target={{(aarch64|arm).*}}
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

; Make an internal function so we exercise R_X86_64_GOTOFF64 relocations.
define internal dso_local void @use_gotoff() {
  ret void
}

define i32 @main() personality ptr @__gxx_personality_v0 {
entry:
  call void @use_gotoff()
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
