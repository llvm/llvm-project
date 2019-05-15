; RUN: opt -S -objc-arc-apelim < %s | FileCheck %s

; See PR26774

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_x, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_y, i8* null }]

@x = global i32 0

declare i32 @bar() nounwind

define linkonce_odr i32 @foo() nounwind {
entry:
  ret i32 5
}

define internal void @__cxx_global_var_init() {
entry:
  %call = call i32 @foo()
  store i32 %call, i32* @x, align 4
  ret void
}

define internal void @__dxx_global_var_init() {
entry:
  %call = call i32 @bar()
  store i32 %call, i32* @x, align 4
  ret void
}

; CHECK-LABEL: define internal void @_GLOBAL__I_x() {
define internal void @_GLOBAL__I_x() {
entry:
; CHECK:  call i8* @llvm.objc.autoreleasePoolPush()
; CHECK-NEXT:  call void @__cxx_global_var_init()
; CHECK-NEXT:  call void @llvm.objc.autoreleasePoolPop(i8* %0)
; CHECK-NEXT:  ret void

  %0 = call i8* @llvm.objc.autoreleasePoolPush() nounwind
  call void @__cxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(i8* %0) nounwind
  ret void
}

define internal void @_GLOBAL__I_y() {
entry:
  %0 = call i8* @llvm.objc.autoreleasePoolPush() nounwind
  call void @__dxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(i8* %0) nounwind
  ret void
}

declare i8* @llvm.objc.autoreleasePoolPush()
declare void @llvm.objc.autoreleasePoolPop(i8*)
