; RUN: llc -mtriple x86_64-apple-darwin %s -o -   | FileCheck %s   --check-prefix=APPLE
; RUN: llc -mtriple x86_64-pc-windows-gnu %s -o - | FileCheck %s   --check-prefix=MINGW64
; RUN: llc -mtriple i686-pc-windows-gnu %s -o -   | FileCheck %s   --check-prefix=MINGW32
@_ZTIi = external constant i8*

define i32 @main() uwtable optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; APPLE: .cfi_startproc
; APPLE: .cfi_personality 155, ___gxx_personality_v0
; APPLE: .cfi_lsda 16, Lexception0
; APPLE: .cfi_def_cfa_offset 16
; APPLE: callq __Unwind_Resume
; APPLE: .cfi_endproc
; APPLE: GCC_except_table0:
; APPLE: Lexception0:

; MINGW64: .seh_proc
; MINGW64: .seh_handler __gxx_personality_v0
; MINGW64: .seh_setframe %rbp, 32
; MINGW64: callq _Unwind_Resume
; MINGW64: .seh_handlerdata
; MINGW64: .seh_endproc
; MINGW64: GCC_except_table0:
; MINGW64: Lexception0:

; MINGW32: .cfi_startproc
; MINGW32: .cfi_personality 0, ___gxx_personality_v0
; MINGW32: .cfi_lsda 0, Lexception0
; MINGW32: .cfi_def_cfa_offset 8
; MINGW32: calll __Unwind_Resume
; MINGW32: .cfi_endproc
; MINGW32: GCC_except_table0:
; MINGW32: Lexception0:

entry:
  invoke void @_Z1fv() optsize
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume

try.cont:
  ret i32 0

eh.resume:
  resume { i8*, i32 } %0
}

declare void @_Z1fv() optsize

declare i32 @__gxx_personality_v0(...)
