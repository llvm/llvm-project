; RUN: opt        -lower-global-dtors -S < %s | FileCheck %s --implicit-check-not=llvm.global_dtors
; RUN: opt -passes=lower-global-dtors -S < %s | FileCheck %s --implicit-check-not=llvm.global_dtors

; Test that @llvm.global_dtors is properly lowered into @llvm.global_ctors,
; grouping dtor calls by priority and associated symbol.

declare void @orig_ctor()
declare void @orig_dtor0()
declare void @orig_dtor1a()
declare void @orig_dtor1b()
declare void @orig_dtor1c0()
declare void @orig_dtor1c1a()
declare void @orig_dtor1c1b()
declare void @orig_dtor1c2a()
declare void @orig_dtor1c2b()
declare void @orig_dtor1c3()
declare void @orig_dtor1d()
declare void @orig_dtor65535()
declare void @orig_dtor65535c0()
declare void @after_the_null()

@associatedc0 = external global i8
@associatedc1 = external global i8
@associatedc2 = global i8 42
@associatedc3 = global i8 84

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 200, void ()* @orig_ctor, i8* null }
]

@llvm.global_dtors = appending global [14 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 0, void ()* @orig_dtor0, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1a, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1b, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c0, i8* @associatedc0 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c1a, i8* @associatedc1 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c1b, i8* @associatedc1 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c2a, i8* @associatedc2 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c2b, i8* @associatedc2 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c3, i8* @associatedc3 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1d, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* @orig_dtor65535c0, i8* @associatedc0 },
  { i32, void ()*, i8* } { i32 65535, void ()* @orig_dtor65535, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* null, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* @after_the_null, i8* null }
]

; CHECK: @associatedc0 = external global i8
; CHECK: @associatedc1 = external global i8
; CHECK: @associatedc2 = global i8 42
; CHECK: @associatedc3 = global i8 84
; CHECK: @__dso_handle = extern_weak hidden constant i8

; CHECK-LABEL: @llvm.global_ctors = appending global [10 x { i32, void ()*, i8* }] [
; CHECK-SAME:  { i32, void ()*, i8* } { i32 200, void ()* @orig_ctor, i8* null },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 0, void ()* @register_call_dtors.0, i8* null },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$0", i8* null },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$1.associatedc0", i8* @associatedc0 },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$2.associatedc1", i8* @associatedc1 },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$3.associatedc2", i8* @associatedc2 },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$4.associatedc3", i8* @associatedc3 },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 1, void ()* @"register_call_dtors.1$5", i8* null },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 65535, void ()* @"register_call_dtors$0.associatedc0", i8* @associatedc0 },
; CHECK-SAME:  { i32, void ()*, i8* } { i32 65535, void ()* @"register_call_dtors$1", i8* null }]

; CHECK: declare void @orig_ctor()
; CHECK: declare void @orig_dtor0()
; --- other dtors here ---
; CHECK: declare void @after_the_null()

; CHECK: declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)

; CHECK-LABEL: define private void @call_dtors.0(i8* %0)
; CHECK:       call void @orig_dtor0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @register_call_dtors.0()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @call_dtors.0, i8* null, i8* @__dso_handle)
; CHECK-NEXT:  %0 = icmp ne i32 %call, 0
; CHECK-NEXT:  br i1 %0, label %fail, label %return
; CHECK-EMPTY:
; CHECK-NEXT:  fail:
; CHECK-NEXT:    call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  return:
; CHECK-NEXT:    ret void

; CHECK-LABEL: define private void @"call_dtors.1$0"(i8* %0)
; CHECK:       call void @orig_dtor1b()
; CHECK-NEXT:  call void @orig_dtor1a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$0"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$0", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$1.associatedc0"(i8* %0)
; CHECK:       call void @orig_dtor1c0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$1.associatedc0"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$1.associatedc0", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$2.associatedc1"(i8* %0)
; CHECK:       call void @orig_dtor1c1b()
; CHECK-NEXT:  call void @orig_dtor1c1a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$2.associatedc1"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$2.associatedc1", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$3.associatedc2"(i8* %0)
; CHECK:       call void @orig_dtor1c2b()
; CHECK-NEXT:  call void @orig_dtor1c2a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$3.associatedc2"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$3.associatedc2", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$4.associatedc3"(i8* %0)
; CHECK:       call void @orig_dtor1c3()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$4.associatedc3"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$4.associatedc3", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$5"(i8* %0)
; CHECK:       call void @orig_dtor1d()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$5"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors.1$5", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors$0.associatedc0"(i8* %0)
; CHECK:       call void @orig_dtor65535c0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors$0.associatedc0"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors$0.associatedc0", i8* null, i8* @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors$1"(i8* %0)
; CHECK:       call void @orig_dtor65535()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors$1"()
; CHECK:       %call = call i32 @__cxa_atexit(void (i8*)* @"call_dtors$1", i8* null, i8* @__dso_handle)


; This function is listed after the null terminator, so it should
; be excluded.

; CHECK-NOT: after_the_null
