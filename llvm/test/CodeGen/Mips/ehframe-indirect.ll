; RUN: llc -mtriple=mipsel-linux-gnu < %s | FileCheck -check-prefix=ALL -check-prefix=O32 %s
; RUN: llc -mtriple=mipsel-linux-android < %s | FileCheck -check-prefix=ALL -check-prefix=O32 %s
; RUN: llc -mtriple=mips64el-linux-gnu < %s | FileCheck -check-prefix=ALL -check-prefix=N64 %s
; RUN: llc -mtriple=mips64el-linux-android < %s | FileCheck -check-prefix=ALL -check-prefix=N64 %s

define i32 @main() {
; ALL: .cfi_startproc
; ALL: .cfi_personality 128, DW.ref.__gxx_personality_v0

entry:
  invoke void @foo() to label %cont unwind label %lpad
; ALL: foo
; ALL: jalr

lpad:
  %0 = landingpad { i8*, i32 } personality i8*
    bitcast (i32 (...)* @__gxx_personality_v0 to i8*) catch i8* null
  ret i32 0

cont:
  ret i32 0
}
; ALL: .cfi_endproc

declare i32 @__gxx_personality_v0(...)

declare void @foo()

; ALL: .hidden DW.ref.__gxx_personality_v0
; ALL: .weak DW.ref.__gxx_personality_v0
; ALL: .section .data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
; O32: .align 2
; N64: .align 3
; ALL: .type DW.ref.__gxx_personality_v0,@object
; O32: .size DW.ref.__gxx_personality_v0, 4
; N64: .size DW.ref.__gxx_personality_v0, 8
; ALL: DW.ref.__gxx_personality_v0:
; O32: .4byte __gxx_personality_v0
; N64: .8byte __gxx_personality_v0
