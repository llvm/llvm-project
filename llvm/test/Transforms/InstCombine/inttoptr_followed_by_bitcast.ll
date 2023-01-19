; RUN: opt -opaque-pointers=0 < %s -passes=instcombine -S | FileCheck %s

define i32 @inttoptr_followed_by_bitcast(i32 %i0, i32 %i1, float %i2) {
; CHECK-LABEL: @inttoptr_followed_by_bitcast(
; CHECK:  [[PTR1:%.*]] = inttoptr i64 [[I:%.*]] to i32 addrspace(1)*
; CHECK:  [[PTR2:%.*]] = bitcast i32 addrspace(1)* [[PTR1]] to float addrspace(1)*
; CHECK:  [[F:%.*]] = load float, float addrspace(1)* [[PTR2]], align 4

  %i3 = zext i32 %i0 to i64
  %i4 = shl i32 %i1, 3
  %i5 = and i32 %i4, -64
  %i6 = zext i32 %i5 to i64
  %i7 = add nuw nsw i64 %i3, %i6

  %ip = inttoptr i64 %i7 to i32 addrspace(1)*
  %fp = bitcast i32 addrspace(1)* %ip to float addrspace(1)*
  %if0 = load float, float addrspace(1)* %fp, align 4

  %ip2 = getelementptr i32, i32 addrspace(1)* %ip, i64 1
  %i8 = load i32, i32 addrspace(1)* %ip2, align 4

  %mul0 = fmul reassoc nnan nsz arcp contract afn float %if0, %i2

  %ip3 = bitcast i32 addrspace(1)* %ip to float addrspace(1)*
  store float %mul0, float addrspace(1)* %ip3, align 4

  ret i32 %i8
}
