; Test for handlerdata when the function has landingpad and nounwind.

; This test case checks whether the handlerdata is generated for the function
; with landingpad instruction, even if the function has "nounwind" atttribute.
;
; For example, although the following function never throws any exception,
; however, it is still required to generate LSDA, otherwise, we can't catch
; the exception properly.
;
; void test1() noexcept {
;   try {
;     throw_exception();
;   } catch (...) {
;   }
; }

; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype=asm -o - %s \
; RUN:   | FileCheck %s

declare void @throw_exception()

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

define void @test1() nounwind personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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

; CHECK-NOT: .cantunwind

; CHECK:   .personality __gxx_personality_v0
; CHECK:   .handlerdata
; CHECK:   .p2align 2
; CHECK-LABEL: GCC_except_table0:
; CHECK-LABEL: .Lexception0:
; CHECK:   .byte 255                     @ @LPStart Encoding = omit
; CHECK:   .byte 0                       @ @TType Encoding = absptr
; CHECK:   .uleb128 .Lttbase
; CHECK:   .byte 1                       @ Call site Encoding = uleb128
; CHECK:   .fnend
