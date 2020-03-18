; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=i386-unknown-unknown < %s | FileCheck %s

;There should be 2 endbr* instruction at entry and catch pad.
;CHECK-COUNT-2: endbr

declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()

define void @test8() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
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
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
