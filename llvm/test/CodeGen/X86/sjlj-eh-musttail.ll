; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39439.
; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s

declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @_callee();

define void @_Z8functionv() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  invoke void @_Z20function_that_throwsv()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  musttail call void @_callee();
  ret void
}

; CHECK-LABEL: __Z8functionv:
; CHECK:         calll   __Unwind_SjLj_Unregister
; CHECK-NOT:     {{.*}}:
; CHECK:         jmp     __callee
