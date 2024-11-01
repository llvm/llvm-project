; RUN: opt -passes=lint -disable-output < %s

; This test is meant to prove that the verifier does not report errors for correct
; use of the llvm.eh.begincatch and llvm.eh.endcatch intrinsics.

target triple = "x86_64-pc-windows-msvc"

declare void @llvm.eh.begincatch(ptr, ptr)

declare void @llvm.eh.endcatch()

@_ZTIi = external constant ptr

; Function Attrs: uwtable
define void @test_ref_clean() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %exn = extractvalue { ptr, i32 } %0, 0
  %sel = extractvalue { ptr, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  call void @llvm.eh.begincatch(ptr %exn, ptr null)
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { ptr, i32 } %0
}

; Function Attrs: uwtable
define void @test_ref_clean_multibranch() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad1

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %exn = extractvalue { ptr, i32 } %0, 0
  %sel = extractvalue { ptr, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad1:                                            ; preds = %entry
  %l1.0 = landingpad { ptr, i32 }
		  cleanup
          catch ptr @_ZTIi
  %exn1 = extractvalue { ptr, i32 } %l1.0, 0
  %sel1 = extractvalue { ptr, i32 } %l1.0, 1
  %l1.1 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matchesl1 = icmp eq i32 %sel1, %l1.1
  br i1 %matchesl1, label %catch, label %eh.resume

catch:                                            ; preds = %lpad, %lpad1
  %exn2 = phi ptr [%exn, %lpad], [%exn1, %lpad1]
  %sel2 = phi i32 [%sel, %lpad], [%sel1, %lpad1]
  call void @llvm.eh.begincatch(ptr %exn2, ptr null)
  call void @_Z10handle_intv()
  %matches1 = icmp eq i32 %sel2, 0
  br i1 %matches1, label %invoke.cont2, label %invoke.cont3

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

invoke.cont3:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %eh.resume

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  %lpad.val = insertvalue { ptr, i32 } undef, i32 0, 1
  resume { ptr, i32 } %lpad.val
}

declare void @_Z9may_throwv()

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr)

declare void @_Z10handle_intv()

