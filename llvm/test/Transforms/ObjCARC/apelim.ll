; RUN: opt -S -objc-arc-apelim < %s | FileCheck %s
; RUN: opt -S -passes=objc-arc-apelim < %s | FileCheck %s
; rdar://10227311

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_x, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_y, i8* null }]

@x = global i32 0

declare i32 @bar() nounwind

define i32 @foo() nounwind {
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

; CHECK: define internal void @_GLOBAL__I_x() {
; CHECK-NOT: @objc
; CHECK: }
define internal void @_GLOBAL__I_x() {
entry:
  %0 = call i8* @llvm.objc.autoreleasePoolPush() nounwind
  call void @__cxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(i8* %0) nounwind
  ret void
}

; CHECK: define internal void @_GLOBAL__I_y() {
; CHECK: %0 = call i8* @llvm.objc.autoreleasePoolPush() [[NUW:#[0-9]+]]
; CHECK: call void @llvm.objc.autoreleasePoolPop(i8* %0) [[NUW]]
; CHECK: }
define internal void @_GLOBAL__I_y() {
entry:
  %0 = call i8* @llvm.objc.autoreleasePoolPush() nounwind
  call void @__dxx_global_var_init()
  call void @llvm.objc.autoreleasePoolPop(i8* %0) nounwind
  ret void
}

declare i8* @llvm.objc.autoreleasePoolPush()
declare void @llvm.objc.autoreleasePoolPop(i8*)

; CHECK: attributes #0 = { nounwind }
