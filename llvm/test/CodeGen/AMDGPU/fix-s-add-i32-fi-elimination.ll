; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs 2>&1 %s | FileCheck %s --check-prefix=ASSERTION

; This test case hit the assertion below, when register scavenger is unable to find a valid register.
; ASSERTION: Assertion `getReg().isPhysical() && "setIsRenamable should only be called on physical registers

define amdgpu_gfx [13 x i32] @_sect_5() {
bb:
  %i = alloca [8 x { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }], i32 0, align 16, addrspace(5)
  %i1 = getelementptr [8 x { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }], ptr addrspace(5) %i, i32 0, i32 0, i32 20
  %i2 = getelementptr [8 x { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }], ptr addrspace(5) %i, i32 0, i32 6, i32 20
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %i4 = phi i32 [ 1, %bb ], [ 0, %bb3 ]
  %i5 = icmp eq i32 %i4, 0
  %i6 = select i1 %i5, ptr addrspace(5) %i2, ptr addrspace(5) %i1
  store i32 0, ptr addrspace(5) %i6, align 16
  %i7 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 1
  %i8 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i7
  store float 0.000000e+00, ptr addrspace(5) %i8, align 4
  %i9 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 2
  %i10 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i9
  store i32 0, ptr addrspace(5) %i10, align 8
  %i11 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 3
  %i12 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i11
  store i32 0, ptr addrspace(5) %i12, align 4
  %i13 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 4
  %i14 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i13
  store i32 0, ptr addrspace(5) %i14, align 16
  %i15 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 5
  %i16 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i15
  store i32 0, ptr addrspace(5) %i16, align 4
  %i17 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 6
  %i18 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i17
  store <2 x float> zeroinitializer, ptr addrspace(5) %i18, align 8
  %i19 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 7
  %i20 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i19
  store i32 0, ptr addrspace(5) %i20, align 16
  %i21 = getelementptr { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, ptr addrspace(5) %i1, i32 0, i32 8
  %i22 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i21
  store <3 x float> zeroinitializer, ptr addrspace(5) %i22, align 16
  %i23 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i1
  store <3 x float> zeroinitializer, ptr addrspace(5) %i23, align 16
  %i24 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 1
  %i25 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i24
  store i32 0, ptr addrspace(5) %i25, align 4
  %i26 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 2
  %i27 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i26
  store i32 0, ptr addrspace(5) %i27, align 8
  %i28 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 3
  %i29 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i28
  store i32 0, ptr addrspace(5) %i29, align 4
  %i30 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 4
  %i31 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i30
  store i32 0, ptr addrspace(5) %i31, align 16
  %i32 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 5
  %i33 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i32
  store i32 0, ptr addrspace(5) %i33, align 4
  %i34 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, { <3 x float>, float, <3 x float>, float }, float, i32, i32, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, { i32, float, i32, i32, i32, i32, <2 x float>, i32, <3 x float>, <3 x float> }, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, ptr addrspace(5) %i, i32 0, i32 6
  %i35 = select i1 %i5, ptr addrspace(5) null, ptr addrspace(5) %i34
  store i32 0, ptr addrspace(5) %i35, align 8
  br label %bb3
}
