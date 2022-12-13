; RUN: opt -S -passes=objc-arc-apelim < %s | FileCheck %s

; See PR26774

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_x, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_y, ptr null }]

@x = global i32 0

declare i32 @bar() nounwind

define linkonce_odr i32 @foo() nounwind {
entry:
  ret i32 5
}

define internal void @__cxx_global_var_init() {
entry:
  %call = call i32 @foo()
  store i32 %call, ptr @x, align 4
  ret void
}

define internal void @__dxx_global_var_init() {
entry:
  %call = call i32 @bar()
  store i32 %call, ptr @x, align 4
  ret void
}

; CHECK-LABEL: define internal void @_GLOBAL__I_x() {
define internal void @_GLOBAL__I_x() {
entry:
; CHECK:  call ptr @llvm.objc.autoreleasePoolPush()
; CHECK-NEXT:  call void @__cxx_global_var_init()
; CHECK-NEXT:  call void @llvm.objc.autoreleasePoolPop(ptr %0)
; CHECK-NEXT:  ret void

  %0 = call ptr @llvm.objc.autoreleasePoolPush() nounwind
  call void @__cxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(ptr %0) nounwind
  ret void
}

define internal void @_GLOBAL__I_y() {
entry:
  %0 = call ptr @llvm.objc.autoreleasePoolPush() nounwind
  call void @__dxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(ptr %0) nounwind
  ret void
}

declare ptr @llvm.objc.autoreleasePoolPush()
declare void @llvm.objc.autoreleasePoolPop(ptr)
