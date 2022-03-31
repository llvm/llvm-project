; RUN: llc -o - %s | FileCheck --check-prefix=SELDAG --check-prefix=CHECK %s
; RUN: llc -global-isel -o - %s | FileCheck --check-prefix=GISEL --check-prefix=CHECK %s

; TODO: support marker generation with GlobalISel
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-iphoneos"

declare i8* @foo0(i32)
declare i8* @foo1()

declare void @llvm.objc.release(i8*)
declare void @objc_object(i8*)

declare void @foo2(i8*)

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1))

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)


%struct.S = type { i8 }

@g = dso_local global i8* null, align 8
@fptr = dso_local global i8* ()* null, align 8

define dso_local i8* @rv_marker_1_retain() {
; CHECK-LABEL:    rv_marker_1_retain:
; CHECK:           .cfi_offset w30, -16
; CHECK-NEXT:      bl foo1
; CHECK-NEXT:     mov x29, x29
; CHECK-NEXT:     bl objc_retainAutoreleasedReturnValue
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  ret i8* %call
}

define dso_local i8* @rv_marker_1_unsafeClaim() {
; CHECK-LABEL:    rv_marker_1_unsafeClaim:
; CHECK:           .cfi_offset w30, -16
; CHECK-NEXT:      bl foo1
; CHECK-NEXT:     mov x29, x29
; CHECK-NEXT:     bl objc_unsafeClaimAutoreleasedReturnValue
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_unsafeClaimAutoreleasedReturnValue) ]
  ret i8* %call
}

define dso_local void @rv_marker_2_select(i32 %c) {
; CHECK-LABEL: rv_marker_2_select:
; SELDAG:        cinc  w0, w8, eq
; GISEL:         csinc w0, w8, wzr, eq
; CHECK-NEXT:    bl  foo0
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    b  foo2
;
entry:
  %tobool.not = icmp eq i32 %c, 0
  %.sink = select i1 %tobool.not, i32 2, i32 1
  %call1 = call i8* @foo0(i32 %.sink) [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call1)
  ret void
}

define dso_local void @rv_marker_3() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: rv_marker_3
; CHECK:         .cfi_offset w30, -32
; CHECK-NEXT:    bl  foo1
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
;
entry:
  %call = call i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  invoke void @objc_object(i8* %call) #5
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  tail call void @llvm.objc.release(i8* %call)
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  tail call void @llvm.objc.release(i8* %call)
  resume { i8*, i32 } %0
}

define dso_local void @rv_marker_4() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: rv_marker_4
; CHECK:       .Ltmp3:
; CHECK-NEXT:     bl  foo1
; CHECK-NEXT:    mov x29, x29
; CHECK-NEXT:    bl objc_retainAutoreleasedReturnValue
; CHECK-NEXT: .Ltmp4:
;
entry:
  %s = alloca %struct.S, align 1
  %0 = getelementptr inbounds %struct.S, %struct.S* %s, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #2
  %call = invoke i8* @foo1() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void @objc_object(i8* %call) #5
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  tail call void @llvm.objc.release(i8* %call)
  %call3 = call %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0)
  ret void

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

lpad1:                                            ; preds = %invoke.cont
  %2 = landingpad { i8*, i32 }
          cleanup
  tail call void @llvm.objc.release(i8* %call)
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad1, %lpad
  %.pn = phi { i8*, i32 } [ %2, %lpad1 ], [ %1, %lpad ]
  %call4 = call %struct.S* @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %s)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0)
  resume { i8*, i32 } %.pn
}

define dso_local i8* @rv_marker_5_indirect_call() {
; CHECK-LABEL: rv_marker_5_indirect_call
; CHECK:         ldr [[ADDR:x[0-9]+]], [
; CHECK-NEXT:    blr [[ADDR]]
; CHECK-NEXT:   mov x29, x29
; CHECK-NEXT:   bl objc_retainAutoreleasedReturnValue
entry:
  %0 = load i8* ()*, i8* ()** @fptr, align 8
  %call = call i8* %0() [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  tail call void @foo2(i8* %call)
  ret i8* %call
}

declare i8* @foo(i64, i64, i64)

define dso_local void @rv_marker_multiarg(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: rv_marker_multiarg
; CHECK:        mov [[TMP:x[0-9]+]], x0
; CHECK-NEXT:   mov x0, x2
; CHECK-NEXT:   mov x2, [[TMP]]
; CHECK-NEXT:   bl  foo
; CHECK-NEXT:  mov x29, x29
; CHECK-NEXT:  bl objc_retainAutoreleasedReturnValue
  call i8* @foo(i64 %c, i64 %b, i64 %a) [ "clang.arc.attachedcall"(i8* (i8*)* @objc_retainAutoreleasedReturnValue) ]
  ret void
}

declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_unsafeClaimAutoreleasedReturnValue(i8*)
declare i32 @__gxx_personality_v0(...)
