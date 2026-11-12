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

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

define void @test1() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @throw_exception() to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
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
; CHECK:   .uleb128 .Lttbase
; CHECK:   .byte 1                       @ Call site Encoding = uleb128
; CHECK:   .uleb128 .Lcst_end
; CHECK:   .uleb128
; CHECK:   .uleb128
; CHECK:   .uleb128
; CHECK:   .fnend
