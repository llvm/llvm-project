; RUN: opt -S -mtriple=amdgcn--amdhsa -load-store-vectorizer < %s | FileCheck %s

; Check that vectorizer can find a GEP through bitcast
; CHECK-LABEL: @vect_zext_bitcast_f32_to_i32_idx
; CHECK: load <4 x i32>
define void @vect_zext_bitcast_f32_to_i32_idx(float addrspace(1)* %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 %zext1
  %f2i1 = bitcast float addrspace(1)* %gep1 to i32 addrspace(1)*
  %load1 = load i32, i32 addrspace(1)* %f2i1, align 4
  %add2 = add nuw i32 %base, 1
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 %zext2
  %f2i2 = bitcast float addrspace(1)* %gep2 to i32 addrspace(1)*
  %load2 = load i32, i32 addrspace(1)* %f2i2, align 4
  %add3 = add nuw i32 %base, 2
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 %zext3
  %f2i3 = bitcast float addrspace(1)* %gep3 to i32 addrspace(1)*
  %load3 = load i32, i32 addrspace(1)* %f2i3, align 4
  %add4 = add nuw i32 %base, 3
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 %zext4
  %f2i4 = bitcast float addrspace(1)* %gep4 to i32 addrspace(1)*
  %load4 = load i32, i32 addrspace(1)* %f2i4, align 4
  ret void
}

; CHECK-LABEL: @vect_zext_bitcast_i8_st1_to_i32_idx
; CHECK: load i32
; CHECK: load i32
; CHECK: load i32
; CHECK: load i32
define void @vect_zext_bitcast_i8_st1_to_i32_idx(i8 addrspace(1)* %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext1
  %f2i1 = bitcast i8 addrspace(1)* %gep1 to i32 addrspace(1)*
  %load1 = load i32, i32 addrspace(1)* %f2i1, align 4
  %add2 = add nuw i32 %base, 1
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds i8,i8 addrspace(1)* %arg1, i64 %zext2
  %f2i2 = bitcast i8 addrspace(1)* %gep2 to i32 addrspace(1)*
  %load2 = load i32, i32 addrspace(1)* %f2i2, align 4
  %add3 = add nuw i32 %base, 2
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext3
  %f2i3 = bitcast i8 addrspace(1)* %gep3 to i32 addrspace(1)*
  %load3 = load i32, i32 addrspace(1)* %f2i3, align 4
  %add4 = add nuw i32 %base, 3
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext4
  %f2i4 = bitcast i8 addrspace(1)* %gep4 to i32 addrspace(1)*
  %load4 = load i32, i32 addrspace(1)* %f2i4, align 4
  ret void
}

; TODO: This can be vectorized, but currently vectorizer unable to do it.
; CHECK-LABEL: @vect_zext_bitcast_i8_st4_to_i32_idx
define void @vect_zext_bitcast_i8_st4_to_i32_idx(i8 addrspace(1)* %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext1
  %f2i1 = bitcast i8 addrspace(1)* %gep1 to i32 addrspace(1)*
  %load1 = load i32, i32 addrspace(1)* %f2i1, align 4
  %add2 = add nuw i32 %base, 4
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds i8,i8 addrspace(1)* %arg1, i64 %zext2
  %f2i2 = bitcast i8 addrspace(1)* %gep2 to i32 addrspace(1)*
  %load2 = load i32, i32 addrspace(1)* %f2i2, align 4
  %add3 = add nuw i32 %base, 8
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext3
  %f2i3 = bitcast i8 addrspace(1)* %gep3 to i32 addrspace(1)*
  %load3 = load i32, i32 addrspace(1)* %f2i3, align 4
  %add4 = add nuw i32 %base, 16
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext4
  %f2i4 = bitcast i8 addrspace(1)* %gep4 to i32 addrspace(1)*
  %load4 = load i32, i32 addrspace(1)* %f2i4, align 4
  ret void
}
