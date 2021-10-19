; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s
; EHCont Guard is currently only available on Windows

; CHECK: .set @feat.00, 16384

; CHECK: .section .gehcont$y

define dso_local void @"?func1@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @"?func2@@YAXXZ"()
          to label %invoke.cont unwind label %catch.dispatch
catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller
catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  catchret from %1 to label %catchret.dest
catchret.dest:                                    ; preds = %catch
  br label %try.cont
try.cont:                                         ; preds = %catchret.dest, %invoke.cont
  ret void
invoke.cont:                                      ; preds = %entry
  br label %try.cont
}

declare dso_local void @"?func2@@YAXXZ"() #1
declare dso_local i32 @__CxxFrameHandler3(...)

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ehcontguard", i32 1}
