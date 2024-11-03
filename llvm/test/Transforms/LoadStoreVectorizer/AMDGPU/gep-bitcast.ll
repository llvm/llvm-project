; RUN: opt -S -mtriple=amdgcn--amdhsa -passes=load-store-vectorizer < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn--amdhsa -passes='function(load-store-vectorizer)' < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Check that vectorizer can find a GEP through bitcast
; CHECK-LABEL: @vect_zext_bitcast_f32_to_i32_idx
; CHECK: load <4 x i32>
define void @vect_zext_bitcast_f32_to_i32_idx(ptr addrspace(1) %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 %zext1
  %load1 = load i32, ptr addrspace(1) %gep1, align 4
  %add2 = add nuw i32 %base, 1
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 %zext2
  %load2 = load i32, ptr addrspace(1) %gep2, align 4
  %add3 = add nuw i32 %base, 2
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 %zext3
  %load3 = load i32, ptr addrspace(1) %gep3, align 4
  %add4 = add nuw i32 %base, 3
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 %zext4
  %load4 = load i32, ptr addrspace(1) %gep4, align 4
  ret void
}

; CHECK-LABEL: @vect_zext_bitcast_i8_st1_to_i32_idx
; CHECK: load i32
; CHECK: load i32
; CHECK: load i32
; CHECK: load i32
define void @vect_zext_bitcast_i8_st1_to_i32_idx(ptr addrspace(1) %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext1
  %load1 = load i32, ptr addrspace(1) %gep1, align 4
  %add2 = add nuw i32 %base, 1
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds i8,ptr addrspace(1) %arg1, i64 %zext2
  %load2 = load i32, ptr addrspace(1) %gep2, align 4
  %add3 = add nuw i32 %base, 2
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext3
  %load3 = load i32, ptr addrspace(1) %gep3, align 4
  %add4 = add nuw i32 %base, 3
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext4
  %load4 = load i32, ptr addrspace(1) %gep4, align 4
  ret void
}

; CHECK-LABEL: @vect_zext_bitcast_i8_st4_to_i32_idx
; CHECK: load <4 x i32>
define void @vect_zext_bitcast_i8_st4_to_i32_idx(ptr addrspace(1) %arg1, i32 %base) {
  %add1 = add nuw i32 %base, 0
  %zext1 = zext i32 %add1 to i64
  %gep1 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext1
  %load1 = load i32, ptr addrspace(1) %gep1, align 4
  %add2 = add nuw i32 %base, 4
  %zext2 = zext i32 %add2 to i64
  %gep2 = getelementptr inbounds i8,ptr addrspace(1) %arg1, i64 %zext2
  %load2 = load i32, ptr addrspace(1) %gep2, align 4
  %add3 = add nuw i32 %base, 8
  %zext3 = zext i32 %add3 to i64
  %gep3 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext3
  %load3 = load i32, ptr addrspace(1) %gep3, align 4
  %add4 = add nuw i32 %base, 12
  %zext4 = zext i32 %add4 to i64
  %gep4 = getelementptr inbounds i8, ptr addrspace(1) %arg1, i64 %zext4
  %load4 = load i32, ptr addrspace(1) %gep4, align 4
  ret void
}

; CHECK-LABEL: @vect_zext_bitcast_negative_ptr_delta
; CHECK: load <2 x i32>
define void @vect_zext_bitcast_negative_ptr_delta(ptr addrspace(1) %p, i32 %base) {
  %a.offset = add nuw i32 %base, 4
  %t.offset.zexted = zext i32 %base to i64
  %a.offset.zexted = zext i32 %a.offset to i64
  %t.ptr = getelementptr inbounds i16, ptr addrspace(1) %p, i64 %t.offset.zexted
  %a.ptr = getelementptr inbounds i16, ptr addrspace(1) %p, i64 %a.offset.zexted
  %b.ptr = getelementptr inbounds i16, ptr addrspace(1) %t.ptr, i64 6
  %a.val = load i32, ptr addrspace(1) %a.ptr
  %b.val = load i32, ptr addrspace(1) %b.ptr
  ret void
}

; Check i1 corner case
; CHECK-LABEL: @zexted_i1_gep_index
; CHECK: load i32
; CHECK: load i32
define void @zexted_i1_gep_index(ptr addrspace(1) %p, i32 %val) {
  %selector = icmp eq i32 %val, 0
  %flipped = xor i1 %selector, 1
  %index.0 = zext i1 %selector to i64
  %index.1 = zext i1 %flipped to i64
  %gep.0 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.0
  %gep.1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.1
  %val0 = load i32, ptr addrspace(1) %gep.0
  %val1 = load i32, ptr addrspace(1) %gep.1
  ret void
}

; Check i1 corner case
; CHECK-LABEL: @sexted_i1_gep_index
; CHECK: load i32
; CHECK: load i32
define void @sexted_i1_gep_index(ptr addrspace(1) %p, i32 %val) {
  %selector = icmp eq i32 %val, 0
  %flipped = xor i1 %selector, 1
  %index.0 = sext i1 %selector to i64
  %index.1 = sext i1 %flipped to i64
  %gep.0 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.0
  %gep.1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.1
  %val0 = load i32, ptr addrspace(1) %gep.0
  %val1 = load i32, ptr addrspace(1) %gep.1
  ret void
}

; CHECK-LABEL: @zexted_i1_gep_index_different_bbs
; CHECK: load i32
; CHECK: load i32
define void @zexted_i1_gep_index_different_bbs(ptr addrspace(1) %p, i32 %val) {
entry:
  %selector = icmp eq i32 %val, 0
  %flipped = xor i1 %selector, 1
  %index.0 = zext i1 %selector to i64
  %index.1 = zext i1 %flipped to i64
  %gep.0 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.0
  br label %next

next:
  %gep.1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %index.1
  %val0 = load i32, ptr addrspace(1) %gep.0
  %val1 = load i32, ptr addrspace(1) %gep.1
  ret void
}
