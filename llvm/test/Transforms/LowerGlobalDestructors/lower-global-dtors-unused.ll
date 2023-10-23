; RUN: opt -passes=lower-global-dtors -S < %s | FileCheck %s --implicit-check-not=llvm.global_dtors

; Test that @llvm.global_dtors is completely removed if __cxa_atexit
; is a no-op (i.e. doesn't use its first argument).

declare void @orig_dtor()

define i32 @__cxa_atexit(ptr, ptr, ptr) {
  ret i32 0
}

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 0, ptr @orig_dtor, ptr null }
]

; CHECK-NOT: @llvm.global_dtors
; CHECK-NOT: call void @orig_dtor()
