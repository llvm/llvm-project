; RUN: opt -global-merge -debug-only=global-merge -S -o - %s 2>&1 | FileCheck %s

;; Checks from the debug info.
; CHECK:      Number of GV that must be kept:  5
; CHECK-NEXT: Kept: @_ZTIi = external constant ptr
; CHECK-NEXT: Kept: @_ZTIf = external constant ptr
; CHECK-NEXT: Kept: @_ZTId = external constant ptr
; CHECK-NEXT: Kept: @_ZTIc = external constant ptr
; CHECK-NEXT: Kept: @_ZTIPi = external constant ptr

;; Check that the landingpad, catch and filter have not changed.
; CHECK:      %0 = landingpad { ptr, i32 }
; CHECK-NEXT:         catch ptr @_ZTIi
; CHECK-NEXT:         filter [5 x ptr] [ptr @_ZTIi, ptr @_ZTIf, ptr @_ZTId, ptr @_ZTIc, ptr @_ZTIPi]

@_ZTIi = external constant ptr
@_ZTIf = external constant ptr
@_ZTId = external constant ptr
@_ZTIc = external constant ptr
@_ZTIPi = external constant ptr

define noundef signext i32 @_Z6calleri(i32 noundef signext %a) local_unnamed_addr personality ptr @__xlcxx_personality_v1 {
entry:
  invoke void @_Z16callee_can_throwi(i32 noundef signext %a)
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          filter [5 x ptr] [ptr @_ZTIi, ptr @_ZTIf, ptr @_ZTId, ptr @_ZTIc, ptr @_ZTIPi]
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = tail call i32 @llvm.eh.typeid.for(ptr nonnull @_ZTIi)
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %filter.dispatch

filter.dispatch:                                  ; preds = %lpad
  %ehspec.fails = icmp slt i32 %2, 0
  br i1 %ehspec.fails, label %ehspec.unexpected, label %eh.resume

ehspec.unexpected:                                ; preds = %filter.dispatch
  tail call void @__cxa_call_unexpected(ptr %1)
  unreachable

catch:                                            ; preds = %lpad
  %4 = tail call ptr @__cxa_begin_catch(ptr %1)
  %5 = load i32, ptr %4, align 4
  tail call void @__cxa_end_catch()
  br label %return

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ %5, %catch ], [ 0, %entry ]
  ret i32 %retval.0

eh.resume:                                        ; preds = %filter.dispatch
  resume { ptr, i32 } %0
}

declare void @_Z16callee_can_throwi(i32 noundef signext) local_unnamed_addr

declare i32 @__xlcxx_personality_v1(...)

declare i32 @llvm.eh.typeid.for(ptr)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

declare void @__cxa_call_unexpected(ptr) local_unnamed_addr
