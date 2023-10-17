; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=pseudo-probe -function-sections -S -o - | FileCheck %s

;; Check the generation of pseudoprobe intrinsic call for non-EH blocks only.

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(ptr) nounwind
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @bar()

@_ZTIi = external constant ptr

define void @foo() uwtable ssp personality ptr @__gxx_personality_v0 {
entry:
; CHECK: call void @llvm.pseudoprobe
  invoke void @bar()
          to label %ret unwind label %lpad

ret:
; CHECK: call void @llvm.pseudoprobe
  ret void

lpad:                                             ; preds = %entry
; CHECK-NOT: call void @llvm.pseudoprobe
  %exn = landingpad {ptr, i32}
            catch ptr @_ZTIi
  %eh.exc = extractvalue { ptr, i32 } %exn, 0
  %eh.selector = extractvalue { ptr, i32 } %exn, 1
  %0 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  %1 = icmp eq i32 %eh.selector, %0
  br i1 %1, label %catch, label %eh.resume

catch:
; CHECK-NOT: call void @llvm.pseudoprobe
  %ignored = call ptr @__cxa_begin_catch(ptr %eh.exc) nounwind
  call void @__cxa_end_catch() nounwind
  br label %ret

eh.resume:
; CHECK-NOT: call void @llvm.pseudoprobe
  resume { ptr, i32 } %exn
}
