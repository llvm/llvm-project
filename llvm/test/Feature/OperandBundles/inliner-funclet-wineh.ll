; RUN: opt -S -passes=always-inline -mtriple=x86_64-windows-msvc < %s | FileCheck %s

; WinEH requires funclet tokens on nounwind intrinsics if they can lower to
; regular function calls in the course of IR transformations.
;
; Test that the inliner propagates funclet tokens to such intrinsics when
; inlining into EH funclets, i.e.: llvm.objc.storeStrong inherits the "funclet"
; token from inlined_fn.

define void @inlined_fn(ptr %ex) #1 {
entry:
  call void @llvm.objc.storeStrong(ptr %ex, ptr null)
  ret void
}

define void @test_catch_with_inline() personality ptr @__CxxFrameHandler3 {
entry:
  %exn.slot = alloca ptr, align 8
  %ex = alloca ptr, align 8
  invoke void @opaque() to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchswitch within none [label %catch] unwind to caller

invoke.cont:
  unreachable

catch:
  %1 = catchpad within %0 [ptr null, i32 64, ptr %exn.slot]
  call void @inlined_fn(ptr %ex) [ "funclet"(token %1) ]
  catchret from %1 to label %catchret.dest

catchret.dest:
  ret void
}

declare void @opaque()
declare void @llvm.objc.storeStrong(ptr, ptr) #0
declare i32 @__CxxFrameHandler3(...)

attributes #0 = { nounwind }
attributes #1 = { alwaysinline }

; After inlining, llvm.objc.storeStrong inherited the "funclet" token:
;
;   CHECK-LABEL:  define void @test_catch_with_inline()
;                   ...
;   CHECK:        catch:
;   CHECK-NEXT:     %1 = catchpad within %0 [ptr null, i32 64, ptr %exn.slot]
;   CHECK-NEXT:     call void @llvm.objc.storeStrong(ptr %ex, ptr null) [ "funclet"(token %1) ]
;   CHECK-NEXT:     catchret from %1 to label %catchret.dest
