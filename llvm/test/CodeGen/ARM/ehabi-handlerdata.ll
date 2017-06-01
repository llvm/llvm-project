; ARM EHABI test for the handlerdata.

; This test case checks whether the handlerdata for exception
; handling is generated properly.
;
; (1) The handlerdata must not be empty.
; (2) LPStartEncoding == DW_EH_PE_omit
; (3) TTypeEncoding == DW_EH_PE_absptr
; (4) CallSiteEncoding == DW_EH_PE_udata4

; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype=asm -o - %s \
; RUN:   | FileCheck %s

; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype=asm -o - %s \
; RUN:     -relocation-model=pic \
; RUN:   | FileCheck %s

declare void @throw_exception()

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

define void @test1() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @throw_exception() to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

; CHECK:   .globl test1
; CHECK:   .p2align 2
; CHECK:   .type test1,%function
; CHECK-LABEL: test1:
; CHECK:   .fnstart
; CHECK:   .personality __gxx_personality_v0
; CHECK:   .handlerdata
; CHECK:   .p2align 2
; CHECK-LABEL: GCC_except_table0:
; CHECK-LABEL: .Lexception0:
; CHECK:   .byte 255                     @ @LPStart Encoding = omit
; CHECK:   .byte 0                       @ @TType Encoding = absptr
; CHECK:   .asciz
; CHECK:   .byte 3                       @ Call site Encoding = udata4
; CHECK:   .long
; CHECK:   .long
; CHECK:   .long
; CHECK:   .fnend
