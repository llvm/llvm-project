; RUN: opt < %s -passes=gvn -S | FileCheck %s
%struct.__fundamental_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { ptr, ptr }

@_ZTIi = external constant %struct.__fundamental_type_info_pseudo
@_ZTIb = external constant %struct.__fundamental_type_info_pseudo

declare void @_Z4barv()

declare void @_Z7cleanupv()

declare i32 @llvm.eh.typeid.for(ptr) nounwind readonly

declare ptr @__cxa_begin_catch(ptr) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(i32, i64, ptr, ptr)

define void @_Z3foov() uwtable personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z4barv()
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
          catch ptr @_ZTIb
          catch ptr @_ZTIi
          catch ptr @_ZTIb
  %exc_ptr2.i = extractvalue { ptr, i32 } %0, 0
  %filter3.i = extractvalue { ptr, i32 } %0, 1
  %typeid.i = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
; CHECK: call i32 @llvm.eh.typeid.for
  %1 = icmp eq i32 %filter3.i, %typeid.i
  br i1 %1, label %ppad, label %next

next:                                             ; preds = %lpad
  %typeid1.i = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIb)
; CHECK: call i32 @llvm.eh.typeid.for
  %2 = icmp eq i32 %filter3.i, %typeid1.i
  br i1 %2, label %ppad2, label %next2

ppad:                                             ; preds = %lpad
  %3 = tail call ptr @__cxa_begin_catch(ptr %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

ppad2:                                            ; preds = %next
  %D.2073_5.i = tail call ptr @__cxa_begin_catch(ptr %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

next2:                                            ; preds = %next
  call void @_Z7cleanupv()
  %typeid = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
; CHECK-NOT: call i32 @llvm.eh.typeid.for
  %4 = icmp eq i32 %filter3.i, %typeid
  br i1 %4, label %ppad3, label %next3

next3:                                            ; preds = %next2
  %typeid1 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIb)
  %5 = icmp eq i32 %filter3.i, %typeid1
  br i1 %5, label %ppad4, label %unwind

unwind:                                           ; preds = %next3
  resume { ptr, i32 } %0

ppad3:                                            ; preds = %next2
  %6 = tail call ptr @__cxa_begin_catch(ptr %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

ppad4:                                            ; preds = %next3
  %D.2080_5 = tail call ptr @__cxa_begin_catch(ptr %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

return:                                           ; preds = %ppad4, %ppad3, %ppad2, %ppad, %entry
  ret void
}
