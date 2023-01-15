; RUN: llc -o - %s | FileCheck --check-prefix=SELDAG --check-prefix=CHECK %s
; RUN: llc -global-isel -o - %s | FileCheck --check-prefix=GISEL --check-prefix=CHECK %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

declare ptr @foo0(i32)
declare ptr @foo1()

declare void @llvm.objc.release(ptr)
declare void @objc_object(ptr)

declare void @foo2(ptr)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare ptr @_ZN1SD1Ev(ptr nonnull dereferenceable(1))

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)


%struct.S = type { i8 }

@g = dso_local global ptr null, align 8
@fptr = dso_local global ptr null, align 8

define dso_local ptr @rv_marker_1_retain() {
; CHECK-LABEL: _rv_marker_1_retain:
; CHECK:         bl _foo1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
;
entry:
  %call = call ptr @foo1() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  ret ptr %call
}

define dso_local ptr @rv_marker_1_unsafeClaim() {
; CHECK-LABEL: _rv_marker_1_unsafeClaim:
; CHECK:         bl _foo1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_unsafeClaimAutoreleasedReturnValue
;
entry:
  %call = call ptr @foo1() [ "clang.arc.attachedcall"(ptr @objc_unsafeClaimAutoreleasedReturnValue) ]
  ret ptr %call
}

define dso_local void @rv_marker_2_select(i32 %c) {
; CHECK-LABEL: _rv_marker_2_select:
; SELDAG:        cinc  w0, w8, eq
; GISEL:         csinc w0, w8, wzr, eq
; CHECK-NEXT:    bl _foo0
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    ldp x29, x30, [sp], #16
; CHECK-NEXT:    b _foo2
;
entry:
  %tobool.not = icmp eq i32 %c, 0
  %.sink = select i1 %tobool.not, i32 2, i32 1
  %call1 = call ptr @foo0(i32 %.sink) [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(ptr %call1)
  ret void
}

define dso_local void @rv_marker_3() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: _rv_marker_3:
; CHECK:         bl _foo1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
;
entry:
  %call = call ptr @foo1() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  invoke void @objc_object(ptr %call) #5
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  tail call void @llvm.objc.release(ptr %call)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
  tail call void @llvm.objc.release(ptr %call)
  resume { ptr, i32 } %0
}

define dso_local void @rv_marker_4() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: _rv_marker_4:
; CHECK:       Ltmp3:
; CHECK-NEXT:    bl _foo1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
; CHECK-NEXT:  Ltmp4:
;
entry:
  %s = alloca %struct.S, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %s) #2
  %call = invoke ptr @foo1() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void @objc_object(ptr %call) #5
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  tail call void @llvm.objc.release(ptr %call)
  %call3 = call ptr @_ZN1SD1Ev(ptr nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %s)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
  br label %ehcleanup

lpad1:                                            ; preds = %invoke.cont
  %1 = landingpad { ptr, i32 }
          cleanup
  tail call void @llvm.objc.release(ptr %call)
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad1, %lpad
  %.pn = phi { ptr, i32 } [ %1, %lpad1 ], [ %0, %lpad ]
  %call4 = call ptr @_ZN1SD1Ev(ptr nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %s)
  resume { ptr, i32 } %.pn
}

define dso_local ptr @rv_marker_5_indirect_call() {
; CHECK-LABEL: _rv_marker_5_indirect_call:
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blr [[ADDR]]
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
entry:
  %0 = load ptr, ptr @fptr, align 8
  %call = call ptr %0() [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(ptr %call)
  ret ptr %call
}

declare ptr @foo(i64, i64, i64)

define dso_local void @rv_marker_multiarg(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: _rv_marker_multiarg:
; CHECK:         mov [[TMP:x[0-9]+]], x0
; CHECK-NEXT:    mov x0, x2
; CHECK-NEXT:    mov x2, [[TMP]]
; CHECK-NEXT:    bl  _foo
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl _objc_retainAutoreleasedReturnValue
  call ptr @foo(i64 %c, i64 %b, i64 %a) [ "clang.arc.attachedcall"(ptr @objc_retainAutoreleasedReturnValue) ]
  ret void
}

declare ptr @objc_retainAutoreleasedReturnValue(ptr)
declare ptr @objc_unsafeClaimAutoreleasedReturnValue(ptr)
declare i32 @__gxx_personality_v0(...)
