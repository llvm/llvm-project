; RUN: llc -mtriple x86_64-apple-darwin %s -o - | FileCheck %s
@_ZTIi = external constant i8*

define i32 @main() uwtable optsize ssp {
entry:
  invoke void @_Z1fv() optsize
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume

try.cont:
  ret i32 0

eh.resume:
  resume { i8*, i32 } %0
}

declare void @_Z1fv() optsize

declare i32 @__gxx_personality_v0(...)

; CHECK: Leh_func_end0:
; CHECK: GCC_except_table0
; CHECK: = Leh_func_end0-
