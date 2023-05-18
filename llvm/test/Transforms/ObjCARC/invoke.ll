; RUN: opt -S -passes=objc-arc < %s | FileCheck %s

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @objc_msgSend(ptr, ptr, ...)
declare void @use_pointer(ptr)
declare void @callee()
declare ptr @returner()

; ARCOpt shouldn't try to move the releases to the block containing the invoke.

; CHECK-LABEL: define void @test0(
; CHECK: invoke.cont:
; CHECK:   call void @llvm.objc.release(ptr %zipFile) [[NUW:#[0-9]+]], !clang.imprecise_release !0
; CHECK:   ret void
; CHECK: lpad:
; CHECK:   call void @llvm.objc.release(ptr %zipFile) [[NUW]], !clang.imprecise_release !0
; CHECK:   ret void
; CHECK-NEXT: }
define void @test0(ptr %zipFile) personality ptr @__gxx_personality_v0 {
entry:
  call ptr @llvm.objc.retain(ptr %zipFile) nounwind
  call void @use_pointer(ptr %zipFile)
  invoke void @objc_msgSend(ptr %zipFile) 
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @llvm.objc.release(ptr %zipFile) nounwind, !clang.imprecise_release !0
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {ptr, i32}
           cleanup
  call void @llvm.objc.release(ptr %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

; ARCOpt should move the release before the callee calls.

; CHECK-LABEL: define void @test1(
; CHECK: invoke.cont:
; CHECK:   call void @llvm.objc.release(ptr %zipFile) [[NUW]], !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: lpad:
; CHECK:   call void @llvm.objc.release(ptr %zipFile) [[NUW]], !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: done:
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test1(ptr %zipFile) personality ptr @__gxx_personality_v0 {
entry:
  call ptr @llvm.objc.retain(ptr %zipFile) nounwind
  call void @use_pointer(ptr %zipFile)
  invoke void @objc_msgSend(ptr %zipFile)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @callee()
  br label %done

lpad:                                             ; preds = %entry
  %exn = landingpad {ptr, i32}
           cleanup
  call void @callee()
  br label %done

done:
  call void @llvm.objc.release(ptr %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

; The optimizer should ignore invoke unwind paths consistently.
; PR12265

; CHECK: define void @test2() personality ptr @__objc_personality_v0 {
; CHECK: invoke.cont:
; CHECK-NEXT: call ptr @llvm.objc.retain
; CHECK-NOT: @llvm.objc.r
; CHECK: finally.cont:
; CHECK-NEXT: call void @llvm.objc.release
; CHECK-NOT: @objc
; CHECK: finally.rethrow:
; CHECK-NOT: @objc
; CHECK: }
define void @test2() personality ptr @__objc_personality_v0 {
entry:
  %call = invoke ptr @objc_msgSend()
          to label %invoke.cont unwind label %finally.rethrow, !clang.arc.no_objc_arc_exceptions !0

invoke.cont:                                      ; preds = %entry
  %tmp1 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call) nounwind
  call void @objc_msgSend(), !clang.arc.no_objc_arc_exceptions !0
  invoke void @use_pointer(ptr %call)
          to label %finally.cont unwind label %finally.rethrow, !clang.arc.no_objc_arc_exceptions !0

finally.cont:                                     ; preds = %invoke.cont
  tail call void @llvm.objc.release(ptr %call) nounwind, !clang.imprecise_release !0
  ret void

finally.rethrow:                                  ; preds = %invoke.cont, %entry
  %tmp2 = landingpad { ptr, i32 }
          catch ptr null
  unreachable
}

; Don't try to place code on invoke critical edges.

; CHECK-LABEL: define void @test3(
; CHECK: if.end:
; CHECK-NEXT: call void @llvm.objc.release(ptr %p) [[NUW]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test3(ptr %p, i1 %b) personality ptr @__objc_personality_v0 {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  br i1 %b, label %if.else, label %if.then

if.then:
  invoke void @use_pointer(ptr %p)
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

if.else:
  invoke void @use_pointer(ptr %p)
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r = landingpad { ptr, i32 }
       cleanup
  ret void

if.end:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Like test3, but with ARC-relevant exception handling.

; CHECK-LABEL: define void @test4(
; CHECK: lpad:
; CHECK-NEXT: %r = landingpad { ptr, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call void @llvm.objc.release(ptr %p) [[NUW]]
; CHECK-NEXT: ret void
; CHECK: if.end:
; CHECK-NEXT: call void @llvm.objc.release(ptr %p) [[NUW]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test4(ptr %p, i1 %b) personality ptr @__objc_personality_v0 {
entry:
  %0 = call ptr @llvm.objc.retain(ptr %p)
  call void @callee()
  br i1 %b, label %if.else, label %if.then

if.then:
  invoke void @use_pointer(ptr %p)
          to label %if.end unwind label %lpad

if.else:
  invoke void @use_pointer(ptr %p)
          to label %if.end unwind label %lpad

lpad:
  %r = landingpad { ptr, i32 }
       cleanup
  call void @llvm.objc.release(ptr %p)
  ret void

if.end:
  call void @llvm.objc.release(ptr %p)
  ret void
}

; Don't turn the retainAutoreleaseReturnValue into retain, because it's
; for an invoke which we can assume codegen will put immediately prior.

; CHECK-LABEL: define void @test5(
; CHECK: call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z)
; CHECK: }
define void @test5() personality ptr @__objc_personality_v0 {
entry:
  %z = invoke ptr @returner()
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r13 = landingpad { ptr, i32 }
          cleanup
  ret void

if.end:
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z)
  ret void
}

; Like test5, but there's intervening code.

; CHECK-LABEL: define void @test6(
; CHECK: call ptr @llvm.objc.retain(ptr %z)
; CHECK: }
define void @test6() personality ptr @__objc_personality_v0 {
entry:
  %z = invoke ptr @returner()
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r13 = landingpad { ptr, i32 }
          cleanup
  ret void

if.end:
  call void @callee()
  call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %z)
  ret void
}

declare i32 @__gxx_personality_v0(...)
declare i32 @__objc_personality_v0(...)

; CHECK: attributes [[NUW]] = { nounwind }

!0 = !{}
