; RUN: opt < %s -passes=gvn -S | FileCheck %s
%struct.__fundamental_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { i8*, i8* }

@_ZTIi = external constant %struct.__fundamental_type_info_pseudo
@_ZTIb = external constant %struct.__fundamental_type_info_pseudo

declare void @_Z4barv()

declare void @_Z7cleanupv()

declare i32 @llvm.eh.typeid.for(i8*) nounwind readonly

declare i8* @__cxa_begin_catch(i8*) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)

define void @_Z3foov() uwtable personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0 {
entry:
  invoke void @_Z4barv()
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch %struct.__fundamental_type_info_pseudo* @_ZTIi
          catch %struct.__fundamental_type_info_pseudo* @_ZTIb
          catch %struct.__fundamental_type_info_pseudo* @_ZTIi
          catch %struct.__fundamental_type_info_pseudo* @_ZTIb
  %exc_ptr2.i = extractvalue { i8*, i32 } %0, 0
  %filter3.i = extractvalue { i8*, i32 } %0, 1
  %typeid.i = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*))
; CHECK: call i32 @llvm.eh.typeid.for
  %1 = icmp eq i32 %filter3.i, %typeid.i
  br i1 %1, label %ppad, label %next

next:                                             ; preds = %lpad
  %typeid1.i = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIb to i8*))
; CHECK: call i32 @llvm.eh.typeid.for
  %2 = icmp eq i32 %filter3.i, %typeid1.i
  br i1 %2, label %ppad2, label %next2

ppad:                                             ; preds = %lpad
  %3 = tail call i8* @__cxa_begin_catch(i8* %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

ppad2:                                            ; preds = %next
  %D.2073_5.i = tail call i8* @__cxa_begin_catch(i8* %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

next2:                                            ; preds = %next
  call void @_Z7cleanupv()
  %typeid = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*))
; CHECK-NOT: call i32 @llvm.eh.typeid.for
  %4 = icmp eq i32 %filter3.i, %typeid
  br i1 %4, label %ppad3, label %next3

next3:                                            ; preds = %next2
  %typeid1 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIb to i8*))
  %5 = icmp eq i32 %filter3.i, %typeid1
  br i1 %5, label %ppad4, label %unwind

unwind:                                           ; preds = %next3
  resume { i8*, i32 } %0

ppad3:                                            ; preds = %next2
  %6 = tail call i8* @__cxa_begin_catch(i8* %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

ppad4:                                            ; preds = %next3
  %D.2080_5 = tail call i8* @__cxa_begin_catch(i8* %exc_ptr2.i) nounwind
  tail call void @__cxa_end_catch() nounwind
  br label %return

return:                                           ; preds = %ppad4, %ppad3, %ppad2, %ppad, %entry
  ret void
}
