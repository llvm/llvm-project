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

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 200, ptr @orig_ctor, ptr null }
]

@llvm.global_dtors = appending global [14 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 0, ptr @orig_dtor0, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1a, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1b, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c0, ptr @associatedc0 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c1a, ptr @associatedc1 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c1b, ptr @associatedc1 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c2a, ptr @associatedc2 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c2b, ptr @associatedc2 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c3, ptr @associatedc3 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1d, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr @orig_dtor65535c0, ptr @associatedc0 },
  { i32, ptr, ptr } { i32 65535, ptr @orig_dtor65535, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr null, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr @after_the_null, ptr null }
]

; CHECK: @associatedc0 = external global i8
; CHECK: @associatedc1 = external global i8
; CHECK: @associatedc2 = global i8 42
; CHECK: @associatedc3 = global i8 84
; CHECK: @__dso_handle = extern_weak hidden constant i8

; CHECK-LABEL: @llvm.global_ctors = appending global [10 x { i32, ptr, ptr }] [
; CHECK-SAME:  { i32, ptr, ptr } { i32 200, ptr @orig_ctor, ptr null },
; CHECK-SAME:  { i32, ptr, ptr } { i32 0, ptr @register_call_dtors.0, ptr null },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$0", ptr null },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$1.associatedc0", ptr @associatedc0 },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$2.associatedc1", ptr @associatedc1 },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$3.associatedc2", ptr @associatedc2 },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$4.associatedc3", ptr @associatedc3 },
; CHECK-SAME:  { i32, ptr, ptr } { i32 1, ptr @"register_call_dtors.1$5", ptr null },
; CHECK-SAME:  { i32, ptr, ptr } { i32 65535, ptr @"register_call_dtors$0.associatedc0", ptr @associatedc0 },
; CHECK-SAME:  { i32, ptr, ptr } { i32 65535, ptr @"register_call_dtors$1", ptr null }]

; CHECK: declare void @orig_ctor()
; CHECK: declare void @orig_dtor0()
; --- other dtors here ---
; CHECK: declare void @after_the_null()

; CHECK: declare i32 @__cxa_atexit(ptr, ptr, ptr)

; CHECK-LABEL: define private void @call_dtors.0(ptr %0)
; CHECK:       call void @orig_dtor0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @register_call_dtors.0()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @call_dtors.0, ptr null, ptr @__dso_handle)
; CHECK-NEXT:  %0 = icmp ne i32 %call, 0
; CHECK-NEXT:  br i1 %0, label %fail, label %return
; CHECK-EMPTY:
; CHECK-NEXT:  fail:
; CHECK-NEXT:    call void @llvm.trap()
; CHECK-NEXT:    unreachable
; CHECK-EMPTY:
; CHECK-NEXT:  return:
; CHECK-NEXT:    ret void

; CHECK-LABEL: define private void @"call_dtors.1$0"(ptr %0)
; CHECK:       call void @orig_dtor1b()
; CHECK-NEXT:  call void @orig_dtor1a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$0"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$0", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$1.associatedc0"(ptr %0)
; CHECK:       call void @orig_dtor1c0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$1.associatedc0"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$1.associatedc0", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$2.associatedc1"(ptr %0)
; CHECK:       call void @orig_dtor1c1b()
; CHECK-NEXT:  call void @orig_dtor1c1a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$2.associatedc1"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$2.associatedc1", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$3.associatedc2"(ptr %0)
; CHECK:       call void @orig_dtor1c2b()
; CHECK-NEXT:  call void @orig_dtor1c2a()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$3.associatedc2"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$3.associatedc2", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$4.associatedc3"(ptr %0)
; CHECK:       call void @orig_dtor1c3()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$4.associatedc3"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$4.associatedc3", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors.1$5"(ptr %0)
; CHECK:       call void @orig_dtor1d()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors.1$5"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors.1$5", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors$0.associatedc0"(ptr %0)
; CHECK:       call void @orig_dtor65535c0()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors$0.associatedc0"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors$0.associatedc0", ptr null, ptr @__dso_handle)

; CHECK-LABEL: define private void @"call_dtors$1"(ptr %0)
; CHECK:       call void @orig_dtor65535()
; CHECK-NEXT:  ret void

; CHECK-LABEL: define private void @"register_call_dtors$1"()
; CHECK:       %call = call i32 @__cxa_atexit(ptr @"call_dtors$1", ptr null, ptr @__dso_handle)


; This function is listed after the null terminator, so it should
; be excluded.

; CHECK-NOT: after_the_null
