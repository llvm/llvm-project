; RUN: opt -passes=lower-global-dtors -S < %s | FileCheck %s

; Test that @llvm.global_dtors is left alone when the module is built for an
; environment without __cxa_atexit().

; CHECK: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @dtor, ptr null }]
; CHECK-NOT: __cxa_atexit

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @dtor, ptr null }]

declare void @dtor()

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"disable-atexit-based-global-dtor-lowering", i32 1}
