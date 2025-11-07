; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[Int8Ty:[0-9]+]] = OpTypeInt 8 0
; CHECK: %[[Int8PtrTy:[0-9]+]] = OpTypePointer Generic %[[Int8Ty]]
; CHECK-DAG: %[[GlobInt8PtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[Int8Ty]]
; CHECK: %[[GlobInt8PtrPtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[GlobInt8PtrTy]]
; CHECK: %[[Int8PtrGlobPtrPtrTy:[0-9]+]] = OpTypePointer Generic %[[GlobInt8PtrPtrTy]]
; CHECK: %[[Int32Ty:[0-9]+]] = OpTypeInt 32 0
; CHECK: %[[Const5:[0-9]+]] = OpConstant %[[Int32Ty]] 5
; CHECK: %[[ArrTy:[0-9]+]] = OpTypeArray %[[GlobInt8PtrTy]] %[[Const5]]
; CHECK: %[[VtblTy:[0-9]+]] = OpTypeStruct %[[ArrTy]] %[[ArrTy]] %[[ArrTy]] %[[ArrTy]] %[[ArrTy]]
; CHECK: %[[Int64Ty:[0-9]+]] = OpTypeInt 64 0
; CHECK: %[[GlobVtblPtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[VtblTy]]
; CHECK: %[[ConstMinus184:[0-9]+]] = OpConstant %[[Int64Ty]] 18446744073709551432
; CHECK: %[[ConstMinus16:[0-9]+]] = OpConstant %[[Int64Ty]] 18446744073709551600
; CHECK: %[[Const168:[0-9]+]] = OpConstant %[[Int64Ty]] 168
; CHECK: %[[Nullptr:[0-9]+]] = OpConstantNull %[[GlobInt8PtrTy]]
; CHECK: %[[Const184:[0-9]+]] = OpConstant %[[Int64Ty]] 184
; CHECK: %[[Const184toPtr:[0-9]+]] = OpSpecConstantOp %[[GlobInt8PtrTy]] ConvertUToPtr %[[Const184]]
; CHECK: %[[Const168toPtr:[0-9]+]] = OpSpecConstantOp %[[GlobInt8PtrTy]] ConvertUToPtr %[[Const168]]
; CHECK: %[[ConstMinus16toPtr:[0-9]+]] = OpSpecConstantOp %[[GlobInt8PtrTy]] ConvertUToPtr %[[ConstMinus16]]
; CHECK: %[[ConstMinus184toPtr:[0-9]+]] = OpSpecConstantOp %[[GlobInt8PtrTy]] ConvertUToPtr %[[ConstMinus184]]
; CHECK: %[[Vtbl012:[0-9]+]] = OpConstantComposite %[[ArrTy]] %[[Const184toPtr]] %[[Nullptr]] %[[Nullptr]] %[[Nullptr]] %[[Nullptr]]
; CHECK: %[[Vtbl3:[0-9]+]] = OpConstantComposite %[[ArrTy]] %[[Const168toPtr]] %[[ConstMinus16toPtr]] %[[Nullptr]] %[[Nullptr]] %[[Nullptr]]
; CHECK: %[[Vtbl4:[0-9]+]] = OpConstantComposite %[[ArrTy]] %[[ConstMinus184toPtr]] %[[ConstMinus184toPtr]] %[[Nullptr]] %[[Nullptr]] %[[Nullptr]]
; CHECK: %[[Vtbl:[0-9]+]] = OpConstantComposite %[[VtblTy]] %[[Vtbl012]] %[[Vtbl012]] %[[Vtbl012]] %[[Vtbl3]] %[[Vtbl4]]
; CHECK: %[[#]] = OpVariable %[[GlobVtblPtrTy]] CrossWorkgroup %[[Vtbl]]

@vtable = linkonce_odr unnamed_addr addrspace(1) constant { [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }
                                                          { [5 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 184 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null],
                                                            [5 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 184 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null],
                                                            [5 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 184 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null],
                                                            [5 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 168 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 -16 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null],
                                                            [5 x ptr addrspace(1)] [ptr addrspace(1) inttoptr (i64 -184 to ptr addrspace(1)), ptr addrspace(1) inttoptr (i64 -184 to ptr addrspace(1)), ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) null] }

define linkonce_odr spir_func void @foo(ptr addrspace(4) %this) {
entry:
  %0 = getelementptr inbounds i8, ptr addrspace(4) %this, i64 184
  store ptr addrspace(1) getelementptr inbounds inrange(-24, 16) ({ [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }, ptr addrspace(1) @vtable, i32 0, i32 0, i32 3), ptr addrspace(4) %this
  store ptr addrspace(1) getelementptr inbounds inrange(-24, 16) ({ [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }, ptr addrspace(1) @vtable, i32 0, i32 1, i32 3), ptr addrspace(4) %this
  store ptr addrspace(1) getelementptr inbounds inrange(-24, 16) ({ [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }, ptr addrspace(1) @vtable, i32 0, i32 2, i32 3), ptr addrspace(4) %this
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %this, i64 184
  store ptr addrspace(1) getelementptr inbounds inrange(-24, 16) ({ [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }, ptr addrspace(1) @vtable, i32 0, i32 4, i32 3), ptr addrspace(4) %add.ptr
  %add.ptr2 = getelementptr inbounds i8, ptr addrspace(4) %this, i64 16
  store ptr addrspace(1) getelementptr inbounds inrange(-24, 16) ({ [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)], [5 x ptr addrspace(1)] }, ptr addrspace(1) @vtable, i32 0, i32 3, i32 3), ptr addrspace(4) %add.ptr2

  ret void
}
