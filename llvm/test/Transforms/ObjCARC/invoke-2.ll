; RUN: opt -mtriple x86_64-unknown-windows-msvc -passes=objc-arc -o - %s | llvm-dis -o - - | FileCheck %s

target triple = "x86_64-unknown-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

declare dllimport ptr @objc_msgSend(ptr, ptr, ...) local_unnamed_addr

declare dllimport ptr @llvm.objc.retain(ptr returned) local_unnamed_addr
declare dllimport void @llvm.objc.release(ptr) local_unnamed_addr
declare dllimport ptr @llvm.objc.retainAutoreleasedReturnValue(ptr returned) local_unnamed_addr

declare dllimport ptr @llvm.objc.begin_catch(ptr) local_unnamed_addr
declare dllimport void @llvm.objc.end_catch() local_unnamed_addr

@llvm.objc.METH_VAR_NAME_ = private unnamed_addr constant [2 x i8] c"m\00", align 1
@llvm.objc.SELECTOR_REFERENCES_ = private externally_initialized global ptr @llvm.objc.METH_VAR_NAME_, section ".objc_selrefs$B", align 8

define void @f(ptr %i) local_unnamed_addr personality ptr @__CxxFrameHandler3 {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %i)
  %1 = load ptr, ptr @llvm.objc.SELECTOR_REFERENCES_, align 8, !invariant.load !0
  %call = invoke ptr @objc_msgSend(ptr %0, ptr %1)
          to label %invoke.cont unwind label %catch.dispatch, !clang.arc.no_objc_arc_exceptions !0

catch.dispatch:                                   ; preds = %entry
  %2 = catchswitch within none [label %catch] unwind to caller

invoke.cont:                                      ; preds = %entry
  %3 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call)
  tail call void @llvm.objc.release(ptr %3) #0, !clang.imprecise_release !0
  br label %eh.cont

eh.cont:                                          ; preds = %invoke.cont, %catch
  tail call void @llvm.objc.release(ptr %0) #0, !clang.imprecise_release !0
  ret void

catch:                                            ; preds = %catch.dispatch
  %4 = catchpad within %2 [ptr null, i32 0, ptr null]
  %exn.adjusted = tail call ptr @llvm.objc.begin_catch(ptr undef)
  tail call void @llvm.objc.end_catch(), !clang.arc.no_objc_arc_exceptions !0
  catchret from %4 to label %eh.cont
}

; CHECK-LABEL: @f

; CHECK-NOT: tail call ptr @llvm.objc.retain(ptr %i)
; CHECK: load ptr, ptr @llvm.objc.SELECTOR_REFERENCES_, align 8

; CHECK: eh.cont:
; CHECK-NOT: call void @llvm.objc.release(ptr
; CHECK: ret void

attributes #0 = { nounwind }

!0 = !{}

