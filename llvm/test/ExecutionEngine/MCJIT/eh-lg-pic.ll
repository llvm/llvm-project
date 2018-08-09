; REQUIRES: cxx-shared-library
; RUN: %lli -relocation-model=pic -code-model=large %s
; XFAIL: cygwin, windows-msvc, windows-gnu, mips-, mipsel-, i686, i386, aarch64, arm
declare i8* @__cxa_allocate_exception(i64)
declare void @__cxa_throw(i8*, i8*, i8*)
declare i32 @__gxx_personality_v0(...)
declare void @__cxa_end_catch()
declare i8* @__cxa_begin_catch(i8*)

@_ZTIi = external constant i8*

define void @throwException() {
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable
}

; Make an internal function so we exercise R_X86_64_GOTOFF64 relocations.
define internal dso_local void @use_gotoff() {
  ret void
}

define i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @use_gotoff()
  invoke void @throwException()
          to label %try.cont unwind label %lpad

lpad:
  %p = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %e = extractvalue { i8*, i32 } %p, 0
  call i8* @__cxa_begin_catch(i8* %e)
  call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0
}
