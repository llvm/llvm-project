; RUN: opt -S -rewrite-statepoints-for-gc -rs4gc-use-deopt-bundles < %s | FileCheck %s

declare void @g()
declare i32 @h()

define i32 addrspace(1)* @f0(i32 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: @f0(
 entry:
; CHECK: [[TOKEN_0:%[^ ]+]] = call i32 {{[^@]*}} @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @g, i32 0, i32 0, i32 0, i32 1, i32 100, i32 addrspace(1)* %arg)
  call void @g() [ "deopt"(i32 100) ]

; CHECK: %arg.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 [[TOKEN_0]], i32 8, i32 8)
  ret i32 addrspace(1)* %arg
}

define i32 addrspace(1)* @f1(i32 addrspace(1)* %arg) gc "statepoint-example"  personality i32 8  {
; CHECK-LABEL: @f1(
 entry:
; CHECK: [[TOKEN_1:%[^ ]+]] = invoke i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 2882400000, i32 0, void ()* @g, i32 0, i32 0, i32 0, i32 1, i32 100, i32 addrspace(1)* %arg)
  invoke void @g() [ "deopt"(i32 100) ] to label %normal_dest unwind label %unwind_dest

 normal_dest:
; CHECK: %arg.relocated1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 [[TOKEN_1]], i32 8, i32 8)
  ret i32 addrspace(1)* %arg

 unwind_dest: 
  %lpad = landingpad { i8*, i32 } cleanup
  resume { i8*, i32 } undef
}

define i32 addrspace(1)* @f2(i32 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: @f2(
 entry:
; CHECK: [[TOKEN_2:%[^ ]+]] = call i32 (i64, i32, i32 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32f(i64 2882400000, i32 0, i32 ()* @h, i32 0, i32 0, i32 0, i32 1, i32 100, i32 addrspace(1)* %arg)
  %val = call i32 @h() [ "deopt"(i32 100) ]

; CHECK: [[RESULT_F2:%[^ ]+]] = call i32 @llvm.experimental.gc.result.i32(i32 [[TOKEN_2]])
; CHECK: %arg.relocated = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 [[TOKEN_2]], i32 8, i32 8)
; CHECK: %arg.relocated.casted = bitcast i8 addrspace(1)* %arg.relocated to i32 addrspace(1)*

  store i32 %val, i32 addrspace(1)* %arg
; CHECK: store i32 [[RESULT_F2]], i32 addrspace(1)* %arg.relocated.casted
  ret i32 addrspace(1)* %arg
}

define i32 addrspace(1)* @f3(i32 addrspace(1)* %arg) gc "statepoint-example"  personality i32 8  {
; CHECK-LABEL: @f3(
 entry:
; CHECK: [[TOKEN_3:%[^ ]+]] = invoke i32 (i64, i32, i32 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32f(i64 2882400000, i32 0, i32 ()* @h, i32 0, i32 0, i32 0, i32 1, i32 100, i32 addrspace(1)* %arg)
  %val = invoke i32 @h() [ "deopt"(i32 100) ] to label %normal_dest unwind label %unwind_dest

 normal_dest:
; CHECK: [[RESULT_F3:%[^ ]+]] = call i32 @llvm.experimental.gc.result.i32(i32 [[TOKEN_3]])
; CHECK: [[ARG_RELOCATED:%[^ ]+]] = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(i32 [[TOKEN_3]], i32 8, i32 8)
; CHECK: [[ARG_RELOCATED_CASTED:%[^ ]+]] = bitcast i8 addrspace(1)* [[ARG_RELOCATED]] to i32 addrspace(1)*

  store i32 %val, i32 addrspace(1)* %arg

; CHECK: store i32 [[RESULT_F3]], i32 addrspace(1)* [[ARG_RELOCATED_CASTED]]
  ret i32 addrspace(1)* %arg

 unwind_dest: 
  %lpad = landingpad { i8*, i32 } cleanup
  resume { i8*, i32 } undef
}
