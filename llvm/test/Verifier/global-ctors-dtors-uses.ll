; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: invalid uses of intrinsic global variable
; CHECK-NEXT: ptr @llvm.global_ctors
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr } ] [
  { i32, ptr, ptr } { i32 65535, ptr null, ptr null }
]

; CHECK: invalid uses of intrinsic global variable
; CHECK-NEXT: ptr @llvm.global_dtors
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr } ] [
  { i32, ptr, ptr } { i32 65535, ptr null, ptr null }
]

@ctor_user = global ptr @llvm.global_ctors
@dtor_user = global ptr @llvm.global_dtors
