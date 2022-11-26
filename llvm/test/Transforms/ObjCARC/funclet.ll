; RUN: opt -mtriple x86_64-unknown-windows-msvc -passes=objc-arc -S -o - %s | FileCheck %s

; bool g();
; id h();
;
; void f() {
;   id a = nullptr;
;   if (g())
;     a = h();
;   id b = nullptr;
;   g();
; }

declare zeroext i1 @"\01?g@@YA_NXZ"() local_unnamed_addr
declare ptr @"\01?h@@YAPEAUobjc_object@@XZ"() local_unnamed_addr

declare dllimport void @llvm.objc.release(ptr) local_unnamed_addr
declare dllimport ptr @llvm.objc.retainAutoreleasedReturnValue(ptr returned) local_unnamed_addr

declare i32 @__CxxFrameHandler3(...)

define void @"\01?f@@YAXXZ"() local_unnamed_addr personality ptr @__CxxFrameHandler3 {
entry:
  %call = invoke zeroext i1 @"\01?g@@YA_NXZ"()
          to label %invoke.cont unwind label %ehcleanup6

invoke.cont:                                      ; preds = %entry
  br i1 %call, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont
  %call2 = invoke ptr @"\01?h@@YAPEAUobjc_object@@XZ"()
          to label %invoke.cont1 unwind label %ehcleanup6

invoke.cont1:                                     ; preds = %if.then
  %0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call2)
  tail call void @llvm.objc.release(ptr null), !clang.imprecise_release !1
  br label %if.end

if.end:                                           ; preds = %invoke.cont1, %invoke.cont
  %a.0 = phi ptr [ %call2, %invoke.cont1 ], [ null, %invoke.cont ]
  %call4 = invoke zeroext i1 @"\01?g@@YA_NXZ"()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %if.end
  tail call void @llvm.objc.release(ptr null), !clang.imprecise_release !1
  tail call void @llvm.objc.release(ptr %a.0), !clang.imprecise_release !1
  ret void

ehcleanup:                                        ; preds = %if.end
  %1 = cleanuppad within none []
  call void @llvm.objc.release(ptr null) [ "funclet"(token %1) ], !clang.imprecise_release !1
  cleanupret from %1 unwind label %ehcleanup6

ehcleanup6:                                       ; preds = %ehcleanup, %if.then, %entry
  %a.1 = phi ptr [ %a.0, %ehcleanup ], [ null, %if.then ], [ null, %entry ]
  %2 = cleanuppad within none []
  call void @llvm.objc.release(ptr %a.1) [ "funclet"(token %2) ], !clang.imprecise_release !1
  cleanupret from %2 unwind to caller
}

; CHECK-LABEL: ?f@@YAXXZ
; CHECK: call void @llvm.objc.release(ptr {{.*}}) {{.*}}[ "funclet"(token %1) ]
; CHECK-NOT: call void @llvm.objc.release(ptr {{.*}}) {{.*}}[ "funclet"(token %2) ]

define void @"\01?i@@YAXXZ"() local_unnamed_addr personality ptr @__CxxFrameHandler3 {
entry:
  %call = invoke zeroext i1 @"\01?g@@YA_NXZ"()
          to label %invoke.cont unwind label %ehcleanup6

invoke.cont:                                      ; preds = %entry
  br i1 %call, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont
  %call2 = invoke ptr @"\01?h@@YAPEAUobjc_object@@XZ"()
          to label %invoke.cont1 unwind label %ehcleanup6

invoke.cont1:                                     ; preds = %if.then
  %0 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %call2)
  tail call void @llvm.objc.release(ptr null), !clang.imprecise_release !1
  br label %if.end

if.end:                                           ; preds = %invoke.cont1, %invoke.cont
  %a.0 = phi ptr [ %call2, %invoke.cont1 ], [ null, %invoke.cont ]
  %call4 = invoke zeroext i1 @"\01?g@@YA_NXZ"()
          to label %invoke.cont3 unwind label %ehcleanup

invoke.cont3:                                     ; preds = %if.end
  tail call void @llvm.objc.release(ptr null), !clang.imprecise_release !1
  tail call void @llvm.objc.release(ptr %a.0), !clang.imprecise_release !1
  ret void

ehcleanup:                                        ; preds = %if.end
  %1 = cleanuppad within none []
  call void @llvm.objc.release(ptr null) [ "funclet"(token %1) ], !clang.imprecise_release !1
  br label %ehcleanup.1

ehcleanup.1:
  cleanupret from %1 unwind label %ehcleanup6

ehcleanup6:                                       ; preds = %ehcleanup, %if.then, %entry
  %a.1 = phi ptr [ %a.0, %ehcleanup.1 ], [ null, %if.then ], [ null, %entry ]
  %2 = cleanuppad within none []
  call void @llvm.objc.release(ptr %a.1) [ "funclet"(token %2) ], !clang.imprecise_release !1
  cleanupret from %2 unwind to caller
}

; CHECK-LABEL: ?i@@YAXXZ
; CHECK: call void @llvm.objc.release(ptr {{.*}}) {{.*}}[ "funclet"(token %1) ]
; CHECK-NOT: call void @llvm.objc.release(ptr {{.*}}) {{.*}}[ "funclet"(token %2) ]

!1 = !{}

