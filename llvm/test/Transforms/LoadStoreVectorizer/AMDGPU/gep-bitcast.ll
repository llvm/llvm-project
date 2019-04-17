; RUN: opt -S -mtriple=amdgcn--amdhsa -load-store-vectorizer < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn--amdhsa -passes='function(load-store-vectorizer)' < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

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

; CHECK-LABEL: @vect_zext_bitcast_i8_st4_to_i32_idx
; CHECK: load <4 x i32>
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
  %add4 = add nuw i32 %base, 12
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %zext4
  %f2i4 = bitcast i8 addrspace(1)* %gep4 to i32 addrspace(1)*
  %load4 = load i32, i32 addrspace(1)* %f2i4, align 4
  ret void
}

; CHECK-LABEL: @vect_zext_bitcast_negative_ptr_delta
; CHECK: load <2 x i32>
define void @vect_zext_bitcast_negative_ptr_delta(i32 addrspace(1)* %p, i32 %base) {
  %p.bitcasted = bitcast i32 addrspace(1)* %p to i16 addrspace(1)*
  %a.offset = add nuw i32 %base, 4
  %t.offset.zexted = zext i32 %base to i64
  %a.offset.zexted = zext i32 %a.offset to i64
  %t.ptr = getelementptr inbounds i16, i16 addrspace(1)* %p.bitcasted, i64 %t.offset.zexted
  %a.ptr = getelementptr inbounds i16, i16 addrspace(1)* %p.bitcasted, i64 %a.offset.zexted
  %b.ptr = getelementptr inbounds i16, i16 addrspace(1)* %t.ptr, i64 6
  %a.ptr.bitcasted = bitcast i16 addrspace(1)* %a.ptr to i32 addrspace(1)*
  %b.ptr.bitcasted = bitcast i16 addrspace(1)* %b.ptr to i32 addrspace(1)*
  %a.val = load i32, i32 addrspace(1)* %a.ptr.bitcasted
  %b.val = load i32, i32 addrspace(1)* %b.ptr.bitcasted
  ret void
}

; Check i1 corner case
; CHECK-LABEL: @zexted_i1_gep_index
; CHECK: load i32
; CHECK: load i32
define void @zexted_i1_gep_index(i32 addrspace(1)* %p, i32 %val) {
  %selector = icmp eq i32 %val, 0
  %flipped = xor i1 %selector, 1
  %index.0 = zext i1 %selector to i64
  %index.1 = zext i1 %flipped to i64
  %gep.0 = getelementptr inbounds i32, i32 addrspace(1)* %p, i64 %index.0
  %gep.1 = getelementptr inbounds i32, i32 addrspace(1)* %p, i64 %index.1
  %val0 = load i32, i32 addrspace(1)* %gep.0
  %val1 = load i32, i32 addrspace(1)* %gep.1
  ret void
}

; Check i1 corner case
; CHECK-LABEL: @sexted_i1_gep_index
; CHECK: load i32
; CHECK: load i32
define void @sexted_i1_gep_index(i32 addrspace(1)* %p, i32 %val) {
  %selector = icmp eq i32 %val, 0
  %flipped = xor i1 %selector, 1
  %index.0 = sext i1 %selector to i64
  %index.1 = sext i1 %flipped to i64
  %gep.0 = getelementptr inbounds i32, i32 addrspace(1)* %p, i64 %index.0
  %gep.1 = getelementptr inbounds i32, i32 addrspace(1)* %p, i64 %index.1
  %val0 = load i32, i32 addrspace(1)* %gep.0
  %val1 = load i32, i32 addrspace(1)* %gep.1
  ret void
}
