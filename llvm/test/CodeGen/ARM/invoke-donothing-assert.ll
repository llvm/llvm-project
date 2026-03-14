; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s
; This testcase makes sure we can handle invoke @llvm.donothing without
; assertion failure.
; <rdar://problem/13228754> & <rdar://problem/13316637>

; CHECK: .globl  _foo
define void @foo() personality ptr @__gxx_personality_sj0 {
invoke.cont:
  invoke void @callA() 
          to label %invoke.cont25 unwind label %lpad2
invoke.cont25:
  invoke void @llvm.donothing()
          to label %invoke.cont27 unwind label %lpad15

invoke.cont27:
  invoke void @callB()
          to label %invoke.cont75 unwind label %lpad15

invoke.cont75:
  ret void

lpad2:
  %0 = landingpad { ptr, i32 }
          cleanup
  br label %eh.resume

lpad15:
  %1 = landingpad { ptr, i32 }
          cleanup
  br label %eh.resume

eh.resume:
  resume { ptr, i32 } zeroinitializer
}

; CHECK: .globl _bar
define linkonce_odr void @bar(ptr %a) personality ptr @__gxx_personality_sj0 {
if.end.i.i.i:
  invoke void @llvm.donothing()
          to label %call.i.i.i.noexc unwind label %eh.resume

call.i.i.i.noexc:
  br i1 false, label %cleanup, label %new.notnull.i.i

new.notnull.i.i:
  br label %cleanup

cleanup:
  %0 = load i32, ptr %a, align 4
  %inc294 = add nsw i32 %0, 4
  store i32 %inc294, ptr %a, align 4
  br i1 false, label %_ZN3lol5ArrayIivvvvvvvED1Ev.exit, label %delete.notnull.i.i.i1409

delete.notnull.i.i.i1409:
  br label %_ZN3lol5ArrayIivvvvvvvED1Ev.exit

_ZN3lol5ArrayIivvvvvvvED1Ev.exit:
  ret void

eh.resume:
  %1 = landingpad { ptr, i32 }
          cleanup
  %2 = extractvalue { ptr, i32 } %1, 0
  %3 = extractvalue { ptr, i32 } %1, 1
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %2, 0
  %lpad.val395 = insertvalue { ptr, i32 } %lpad.val, i32 %3, 1
  resume { ptr, i32 } %lpad.val395
}

declare void @callA()
declare void @callB()
declare void @llvm.donothing() nounwind readnone
declare i32 @__gxx_personality_sj0(...)
