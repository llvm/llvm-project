; RUN: opt --dwarfehprepare %s -S -o - -mtriple=aarch64-linux-gnu | FileCheck %s

define void @call() sspreq personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @call()
; CHECK:     invoke void @bar()
; CHECK:             to label %[[CONT:.*]] unwind label %[[CLEANUP:.*]]
; CHECK: [[CONT]]:
; CHECK:     ret void
; CHECK: [[CLEANUP]]:
; CHECK:     [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK:             cleanup
; CHECK:     [[EXN:%.*]] = extractvalue { ptr, i32 } [[LP]], 0
; CHECK:     call void @_Unwind_Resume(ptr [[EXN]])
; CHECK:     unreachable
  call void @bar()
  ret void
}

define void @invoke_no_cleanup() sspreq personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @invoke_no_cleanup
; CHECK:   invoke void @bar()
; CHECK:           to label %done unwind label %catch

; CHECK: catch:
; CHECK:   [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK:           cleanup
; CHECK:           catch ptr null
; CHECK:   [[SEL:%.*]] = extractvalue { ptr, i32 } [[LP]], 1
; CHECK:   [[CMP:%.*]] = icmp eq i32 [[SEL]], 0
; CHECK:   br i1 [[CMP]], label %[[RESUME:.*]], label %[[SPLIT:.*]]

; CHECK: [[SPLIT]]:
; CHECK:   br label %done

; CHECK: done:
; CHECK:   ret void

; CHECK: [[RESUME]]:
; CHECK:   [[EXN:%.*]] = extractvalue { ptr, i32 } [[LP]], 0
; CHECK:   call void @_Unwind_Resume(ptr [[EXN]])
; CHECK:   unreachable
  invoke void @bar() to label %done unwind label %catch

catch:
  %lp = landingpad { ptr, i32 }
          catch ptr null
  br label %done

done:
  ret void
}

define void @invoke_no_cleanup_catches() sspreq personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @invoke_no_cleanup_catches
; CHECK:   invoke void @bar()
; CHECK:           to label %done unwind label %catch

; CHECK: catch:
; CHECK:   [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK:           cleanup
; CHECK:           catch ptr null
; CHECK:   [[SEL:%.*]] = extractvalue { ptr, i32 } [[LP]], 1
; CEHCK:   [[CMP:%.*]] = icmp eq i32 [[SEL]], 0
; CEHCK:   br i1 [[CMP]], label %[[RESUME:.*]], label %[[SPLIT:.*]]

; CHECK: [[SPLIT]]:
; CHECK:   %exn = extractvalue { ptr, i32 } %lp, 0
; CHECK:   invoke ptr @__cxa_begin_catch(ptr %exn)
; CHECK:            to label %[[SPLIT2:.*]] unwind label %[[CLEANUP_RESUME:.*]]

; CHECK: [[SPLIT2]]:
; CHECK:   invoke void @__cxa_end_catch()
; CHECK:            to label  %[[SPLIT3:.*]] unwind label %[[CLEANUP_RESUME:.*]]

; CHECK: [[SPLIT3]]:
; CHECK:   br label %done

; CHECK: done:
; CHECK:   ret void

; CHECK: [[RESUME]]:
; CHECK:   [[EXN1:%.*]] = extractvalue { ptr, i32 } [[LP]], 0
; CHECK:   br label %[[RESUME_MERGE:.*]]

; CHECK: [[CLEANUP_RESUME]]:
; CHECK:   [[LP:%.*]] = landingpad { ptr, i32 }
; CHECK:           cleanup
; CHECK:   [[EXN2:%.*]] = extractvalue { ptr, i32 } [[LP]], 0
; CHECK:   br label %[[RESUME_MERGE]]

; CHECK: [[RESUME_MERGE]]:
; CHECK:   [[EXN_PHI:%.*]] = phi ptr [ [[EXN1]], %[[RESUME]] ], [ [[EXN2]], %[[CLEANUP_RESUME]] ]
; CHECK:   call void @_Unwind_Resume(ptr [[EXN_PHI]])
; CHECK:   unreachable
  invoke void @bar() to label %done unwind label %catch

catch:
  %lp = landingpad { ptr, i32 }
          catch ptr null
  %exn = extractvalue { ptr, i32 } %lp, 0
  call ptr @__cxa_begin_catch(ptr %exn)
  call void @__cxa_end_catch()
  br label %done

done:
  ret void
}

; Don't try to invoke any intrinsics.
define ptr @call_intrinsic() sspreq personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define ptr @call_intrinsic
; CHECK: call ptr @llvm.frameaddress.p0(i32 0)
  %res = call ptr @llvm.frameaddress.p0(i32 0)
  ret ptr %res
}

; Check we go along with the existing landingpad type, even if it's a bit
; outside the normal.
define void @weird_landingpad() sspreq personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @weird_landingpad
; CHECK: landingpad { ptr, i64 }
; CHECK: landingpad { ptr, i64 }
  invoke void @bar() to label %done unwind label %catch

catch:
  %lp = landingpad { ptr, i64 }
           catch ptr null
  resume { ptr, i64 } %lp
;  br label %done

done:
  call void @bar()
  ret void
}

declare void @bar()
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare ptr @llvm.frameaddress.p0(i32)
