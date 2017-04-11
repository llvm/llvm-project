; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64-A2"

declare void @llvm.memcpy.p2i8.p2i8.i32(i8 addrspace(2)* nocapture, i8 addrspace(2)* nocapture readonly, i32, i32, i1)
declare void @llvm.memcpy.p1i8.p2i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(2)* nocapture readonly, i32, i32, i1)
declare void @llvm.memcpy.p2i8.p1i8.i32(i8 addrspace(2)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i32, i1)
declare void @llvm.memcpy.p1i8.p1i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i32, i1)



; CHECK-LABEL: @test_address_space_1_1(
; CHECK: load <2 x i64>, <2 x i64> addrspace(1)* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64> addrspace(1)* {{.*}}, align 2
; CHECK: ret void
define void @test_address_space_1_1(<2 x i64> addrspace(1)* %a, i16 addrspace(1)* %b) {
  %aa = alloca <2 x i64>, align 16, addrspace(2)
  %aptr = bitcast <2 x i64> addrspace(1)* %a to i8 addrspace(1)*
  %aaptr = bitcast <2 x i64> addrspace(2)* %aa to i8 addrspace(2)*
  call void @llvm.memcpy.p2i8.p1i8.i32(i8 addrspace(2)* %aaptr, i8 addrspace(1)* %aptr, i32 16, i32 2, i1 false)
  %bptr = bitcast i16 addrspace(1)* %b to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p2i8.i32(i8 addrspace(1)* %bptr, i8 addrspace(2)* %aaptr, i32 16, i32 2, i1 false)
  ret void
}

; CHECK-LABEL: @test_address_space_1_0(
; CHECK: load <2 x i64>, <2 x i64> addrspace(1)* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64> addrspace(2)* {{.*}}, align 2
; CHECK: ret void
define void @test_address_space_1_0(<2 x i64> addrspace(1)* %a, i16 addrspace(2)* %b) {
  %aa = alloca <2 x i64>, align 16, addrspace(2)
  %aptr = bitcast <2 x i64> addrspace(1)* %a to i8 addrspace(1)*
  %aaptr = bitcast <2 x i64> addrspace(2)* %aa to i8 addrspace(2)*
  call void @llvm.memcpy.p2i8.p1i8.i32(i8 addrspace(2)* %aaptr, i8 addrspace(1)* %aptr, i32 16, i32 2, i1 false)
  %bptr = bitcast i16 addrspace(2)* %b to i8 addrspace(2)*
  call void @llvm.memcpy.p2i8.p2i8.i32(i8 addrspace(2)* %bptr, i8 addrspace(2)* %aaptr, i32 16, i32 2, i1 false)
  ret void
}

; CHECK-LABEL: @test_address_space_0_1(
; CHECK: load <2 x i64>, <2 x i64> addrspace(2)* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64> addrspace(1)* {{.*}}, align 2
; CHECK: ret void
define void @test_address_space_0_1(<2 x i64> addrspace(2)* %a, i16 addrspace(1)* %b) {
  %aa = alloca <2 x i64>, align 16, addrspace(2)
  %aptr = bitcast <2 x i64> addrspace(2)* %a to i8 addrspace(2)*
  %aaptr = bitcast <2 x i64> addrspace(2)* %aa to i8 addrspace(2)*
  call void @llvm.memcpy.p2i8.p2i8.i32(i8 addrspace(2)* %aaptr, i8 addrspace(2)* %aptr, i32 16, i32 2, i1 false)
  %bptr = bitcast i16 addrspace(1)* %b to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p2i8.i32(i8 addrspace(1)* %bptr, i8 addrspace(2)* %aaptr, i32 16, i32 2, i1 false)
  ret void
}

%struct.struct_test_27.0.13 = type { i32, float, i64, i8, [4 x i32] }

; CHECK-LABEL: @copy_struct(
; CHECK-NOT: memcpy
define void @copy_struct([5 x i64] %in.coerce) {
for.end:
  %in = alloca %struct.struct_test_27.0.13, align 8, addrspace(2)
  %0 = bitcast %struct.struct_test_27.0.13 addrspace(2)* %in to [5 x i64] addrspace(2)*
  store [5 x i64] %in.coerce, [5 x i64] addrspace(2)* %0, align 8
  %scevgep9 = getelementptr %struct.struct_test_27.0.13, %struct.struct_test_27.0.13 addrspace(2)* %in, i32 0, i32 4, i32 0
  %scevgep910 = bitcast i32 addrspace(2)* %scevgep9 to i8 addrspace(2)*
  call void @llvm.memcpy.p1i8.p2i8.i32(i8 addrspace(1)* undef, i8 addrspace(2)* %scevgep910, i32 16, i32 4, i1 false)
  ret void
}

%union.anon = type { i32* }

@g = common global i32 0, align 4
@l = common addrspace(3) global i32 0, align 4

; Make sure an illegal bitcast isn't introduced
; CHECK-LABEL: @pr27557(
; CHECK: %[[CAST:.*]] = bitcast i32* addrspace(2)* {{.*}} to i32 addrspace(3)* addrspace(2)*
; CHECK: store i32 addrspace(3)* @l, i32 addrspace(3)* addrspace(2)* %[[CAST]]
define void @pr27557() {
  %1 = alloca %union.anon, align 8, addrspace(2)
  %2 = bitcast %union.anon addrspace(2)* %1 to i32* addrspace(2)*
  store i32* @g, i32* addrspace(2)* %2, align 8
  %3 = bitcast %union.anon addrspace(2)* %1 to i32 addrspace(3)* addrspace(2)*
  store i32 addrspace(3)* @l, i32 addrspace(3)* addrspace(2)* %3, align 8
  ret void
}
