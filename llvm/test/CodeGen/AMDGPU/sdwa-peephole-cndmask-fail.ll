; RUN: llc %s -march=amdgcn -mcpu=gfx1030 -o - 2>&1 | FileCheck %s

; In this test, V_CNDMASK_B32_e64 gets converted to V_CNDMASK_B32_e32,
; but the expected conversion to SDWA does not occur.  This led to a
; compilation error, because the use of $vcc in the resulting
; instruction must be fixed to $vcc_lo for wave32 which only happened
; after the full conversion to SDWA.


; CHECK-NOT: {{.*}}V_CNDMASK_B32_e32{{.*}}$vcc
; CHECK-NOT: {{.*}}Bad machine code: Virtual register defs don't dominate all uses
; CHECK: {{.*}}v_cndmask_b32_e32{{.*}}vcc_lo

define amdgpu_kernel void @quux(i32 %arg, i1 %arg1, i1 %arg2) {
bb:
  br i1 %arg1, label %bb9, label %bb3

bb3:                                              ; preds = %bb
  %call = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul = mul i32 %call, 5
  %zext = zext i32 %mul to i64
  %getelementptr = getelementptr i8, ptr addrspace(1) null, i64 %zext
  %getelementptr4 = getelementptr i8, ptr addrspace(1) %getelementptr, i64 4
  %load = load i8, ptr addrspace(1) %getelementptr4, align 1
  %getelementptr5 = getelementptr i8, ptr addrspace(1) %getelementptr, i64 3
  %load6 = load i8, ptr addrspace(1) %getelementptr5, align 1
  %insertelement = insertelement <5 x i8> poison, i8 %load, i64 4
  %select = select i1 %arg2, <5 x i8> %insertelement, <5 x i8> <i8 poison, i8 poison, i8 poison, i8 poison, i8 0>
  %insertelement7 = insertelement <5 x i8> %select, i8 %load6, i64 0
  %icmp = icmp ult i32 0, %arg
  %select8 = select i1 %icmp, <5 x i8> zeroinitializer, <5 x i8> %insertelement7
  %shufflevector = shufflevector <5 x i8> zeroinitializer, <5 x i8> %select8, <5 x i32> <i32 0, i32 1, i32 7, i32 8, i32 9>
  br label %bb9

bb9:                                              ; preds = %bb3, %bb
  %phi = phi <5 x i8> [ %shufflevector, %bb3 ], [ zeroinitializer, %bb ]
  %extractelement = extractelement <5 x i8> %phi, i64 0
  store i8 %extractelement, ptr addrspace(1) null, align 1
  ret void
}
