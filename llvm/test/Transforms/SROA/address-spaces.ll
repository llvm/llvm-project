; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
declare void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* nocapture, i8* nocapture readonly, i32, i1)
declare void @llvm.memcpy.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture readonly, i32, i1)
declare void @llvm.memcpy.p1i8.p1i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i1)


; Make sure an illegal bitcast isn't introduced
define void @test_address_space_1_1(<2 x i64> addrspace(1)* %a, i16 addrspace(1)* %b) {
; CHECK-LABEL: @test_address_space_1_1(
; CHECK: load <2 x i64>, <2 x i64> addrspace(1)* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64> addrspace(1)* {{.*}}, align 2
; CHECK: ret void
  %aa = alloca <2 x i64>, align 16
  %aptr = bitcast <2 x i64> addrspace(1)* %a to i8 addrspace(1)*
  %aaptr = bitcast <2 x i64>* %aa to i8*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* align 2 %aaptr, i8 addrspace(1)* align 2 %aptr, i32 16, i1 false)
  %bptr = bitcast i16 addrspace(1)* %b to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 2 %bptr, i8* align 2 %aaptr, i32 16, i1 false)
  ret void
}

define void @test_address_space_1_0(<2 x i64> addrspace(1)* %a, i16* %b) {
; CHECK-LABEL: @test_address_space_1_0(
; CHECK: load <2 x i64>, <2 x i64> addrspace(1)* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64>* {{.*}}, align 2
; CHECK: ret void
  %aa = alloca <2 x i64>, align 16
  %aptr = bitcast <2 x i64> addrspace(1)* %a to i8 addrspace(1)*
  %aaptr = bitcast <2 x i64>* %aa to i8*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* align 2 %aaptr, i8 addrspace(1)* align 2 %aptr, i32 16, i1 false)
  %bptr = bitcast i16* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %bptr, i8* align 2 %aaptr, i32 16, i1 false)
  ret void
}

define void @test_address_space_0_1(<2 x i64>* %a, i16 addrspace(1)* %b) {
; CHECK-LABEL: @test_address_space_0_1(
; CHECK: load <2 x i64>, <2 x i64>* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64> addrspace(1)* {{.*}}, align 2
; CHECK: ret void
  %aa = alloca <2 x i64>, align 16
  %aptr = bitcast <2 x i64>* %a to i8*
  %aaptr = bitcast <2 x i64>* %aa to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %aaptr, i8* align 2 %aptr, i32 16, i1 false)
  %bptr = bitcast i16 addrspace(1)* %b to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 2 %bptr, i8* align 2 %aaptr, i32 16, i1 false)
  ret void
}

%struct.struct_test_27.0.13 = type { i32, float, i64, i8, [4 x i32] }

; Function Attrs: nounwind
define void @copy_struct([5 x i64] %in.coerce) {
; CHECK-LABEL: @copy_struct(
; CHECK-NOT: memcpy
for.end:
  %in = alloca %struct.struct_test_27.0.13, align 8
  %0 = bitcast %struct.struct_test_27.0.13* %in to [5 x i64]*
  store [5 x i64] %in.coerce, [5 x i64]* %0, align 8
  %scevgep9 = getelementptr %struct.struct_test_27.0.13, %struct.struct_test_27.0.13* %in, i32 0, i32 4, i32 0
  %scevgep910 = bitcast i32* %scevgep9 to i8*
  call void @llvm.memcpy.p1i8.p0i8.i32(i8 addrspace(1)* align 4 undef, i8* align 4 %scevgep910, i32 16, i1 false)
  ret void
}
 
%union.anon = type { i32* }

@g = common global i32 0, align 4
@l = common addrspace(3) global i32 0, align 4

; If pointers from different address spaces have different sizes, make sure an
; illegal bitcast isn't introduced
define void @pr27557() {
; CHECK-LABEL: @pr27557(
; CHECK: %[[CAST:.*]] = bitcast i32** {{.*}} to i32 addrspace(3)**
; CHECK: store i32 addrspace(3)* @l, i32 addrspace(3)** %[[CAST]]
  %1 = alloca %union.anon, align 8
  %2 = bitcast %union.anon* %1 to i32**
  store i32* @g, i32** %2, align 8
  %3 = bitcast %union.anon* %1 to i32 addrspace(3)**
  store i32 addrspace(3)* @l, i32 addrspace(3)** %3, align 8
  ret void
}

@l2 = common addrspace(2) global i32 0, align 4

; If pointers from different address spaces have the same size, that pointer
; should be promoted through the pair of `ptrtoint`/`inttoptr`.
define i32* @pr27557.alt() {
; CHECK-LABEL: @pr27557.alt(
; CHECK: ret i32* inttoptr (i64 ptrtoint (i32 addrspace(2)* @l2 to i64) to i32*)
  %1 = alloca %union.anon, align 8
  %2 = bitcast %union.anon* %1 to i32 addrspace(2)**
  store i32 addrspace(2)* @l2, i32 addrspace(2)** %2, align 8
  %3 = bitcast %union.anon* %1 to i32**
  %4 = load i32*, i32** %3, align 8
  ret i32* %4
}

; Make sure pre-splitting doesn't try to introduce an illegal bitcast
define float @presplit(i64 addrspace(1)* %p) {
entry:
; CHECK-LABEL: @presplit(
; CHECK: %[[CAST:.*]] = bitcast i64 addrspace(1)* {{.*}} to i32 addrspace(1)*
; CHECK: load i32, i32 addrspace(1)* %[[CAST]]
   %b = alloca i64
   %b.cast = bitcast i64* %b to [2 x float]*
   %b.gep1 = getelementptr [2 x float], [2 x float]* %b.cast, i32 0, i32 0
   %b.gep2 = getelementptr [2 x float], [2 x float]* %b.cast, i32 0, i32 1
   %l = load i64, i64 addrspace(1)* %p
   store i64 %l, i64* %b
   %f1 = load float, float* %b.gep1
   %f2 = load float, float* %b.gep2
   %ret = fadd float %f1, %f2
   ret float %ret
}

; Test load from and store to non-zero address space.
define void @test_load_store_diff_addr_space([2 x float] addrspace(1)* %complex1, [2 x float] addrspace(1)* %complex2) {
; CHECK-LABEL: @test_load_store_diff_addr_space
; CHECK-NOT: alloca
; CHECK: load i32, i32 addrspace(1)*
; CHECK: load i32, i32 addrspace(1)*
; CHECK: store i32 %{{.*}}, i32 addrspace(1)*
; CHECK: store i32 %{{.*}}, i32 addrspace(1)*
  %a = alloca i64
  %a.cast = bitcast i64* %a to [2 x float]*
  %a.gep1 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 0
  %a.gep2 = getelementptr [2 x float], [2 x float]* %a.cast, i32 0, i32 1
  %complex1.gep = getelementptr [2 x float], [2 x float] addrspace(1)* %complex1, i32 0, i32 0
  %p1 = bitcast float addrspace(1)* %complex1.gep to i64 addrspace(1)*
  %v1 = load i64, i64 addrspace(1)* %p1
  store i64 %v1, i64* %a
  %f1 = load float, float* %a.gep1
  %f2 = load float, float* %a.gep2
  %sum = fadd float %f1, %f2
  store float %sum, float* %a.gep1
  store float %sum, float* %a.gep2
  %v2 = load i64, i64* %a
  %complex2.gep = getelementptr [2 x float], [2 x float] addrspace(1)* %complex2, i32 0, i32 0
  %p2 = bitcast float addrspace(1)* %complex2.gep to i64 addrspace(1)*
  store i64 %v2, i64 addrspace(1)* %p2
  ret void
}
