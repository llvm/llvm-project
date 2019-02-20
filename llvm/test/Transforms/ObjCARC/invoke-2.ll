; RUN: opt -mtriple x86_64-unknown-windows-msvc -objc-arc -o - %s | llvm-dis -o - - | FileCheck %s

target triple = "x86_64-unknown-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

declare dllimport i8* @llvm.objc.msgSend(i8*, i8*, ...) local_unnamed_addr

declare dllimport i8* @llvm.objc.retain(i8* returned) local_unnamed_addr
declare dllimport void @llvm.objc.release(i8*) local_unnamed_addr
declare dllimport i8* @llvm.objc.retainAutoreleasedReturnValue(i8* returned) local_unnamed_addr

declare dllimport i8* @llvm.objc.begin_catch(i8*) local_unnamed_addr
declare dllimport void @llvm.objc.end_catch() local_unnamed_addr

@llvm.objc.METH_VAR_NAME_ = private unnamed_addr constant [2 x i8] c"m\00", align 1
@llvm.objc.SELECTOR_REFERENCES_ = private externally_initialized global i8* getelementptr inbounds ([2 x i8], [2 x i8]* @llvm.objc.METH_VAR_NAME_, i64 0, i64 0), section ".objc_selrefs$B", align 8

define void @f(i8* %i) local_unnamed_addr personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %0 = tail call i8* @llvm.objc.retain(i8* %i)
  %1 = load i8*, i8** @llvm.objc.SELECTOR_REFERENCES_, align 8, !invariant.load !0
  %call = invoke i8* bitcast (i8* (i8*, i8*, ...)* @llvm.objc.msgSend to i8* (i8*, i8*)*)(i8* %0, i8* %1)
          to label %invoke.cont unwind label %catch.dispatch, !clang.arc.no_objc_arc_exceptions !0

catch.dispatch:                                   ; preds = %entry
  %2 = catchswitch within none [label %catch] unwind to caller

invoke.cont:                                      ; preds = %entry
  %3 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call)
  tail call void @llvm.objc.release(i8* %3) #0, !clang.imprecise_release !0
  br label %eh.cont

eh.cont:                                          ; preds = %invoke.cont, %catch
  tail call void @llvm.objc.release(i8* %0) #0, !clang.imprecise_release !0
  ret void

catch:                                            ; preds = %catch.dispatch
  %4 = catchpad within %2 [i8* null, i32 0, i8* null]
  %exn.adjusted = tail call i8* @llvm.objc.begin_catch(i8* undef)
  tail call void @llvm.objc.end_catch(), !clang.arc.no_objc_arc_exceptions !0
  br label %eh.cont
}

; CHECK-LABEL: @f

; CHECK-NOT: tail call i8* @llvm.objc.retain(i8* %i)
; CHECK: load i8*, i8** @llvm.objc.SELECTOR_REFERENCES_, align 8

; CHECK: eh.cont:
; CHECK-NOT: call void @llvm.objc.release(i8*
; CHECK: ret void

attributes #0 = { nounwind }

!0 = !{}

