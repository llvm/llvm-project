; RUN: opt -S -codegenprepare -mtriple=thumbv7m -disable-complex-addr-modes=false -addr-sink-new-select=true -addr-sink-new-phis=true < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@gv1 = common global i32 0, align 4
@gv2 = common global i32 0, align 4

; Phi selects between ptr and gep with ptr as base and constant offset
define void @test_phi_onegep_offset(i32* %ptr, i32 %value) {
; CHECK-LABEL: @test_phi_onegep_offset
; CHECK-NOT: phi i32* [ %ptr, %entry ], [ %gep, %if.then ]
; CHECK: phi i32 [ 4, %if.then ], [ 0, %entry ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %gep = getelementptr inbounds i32, i32* %ptr, i32 1
  br label %if.end

if.end:
  %phi = phi i32* [ %ptr, %entry ], [ %gep, %if.then ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between two geps with same base, different constant offsets
define void @test_phi_twogep_offset(i32* %ptr, i32 %value) {
; CHECK-LABEL: @test_phi_twogep_offset
; CHECK-NOT: phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
; CHECK: phi i32 [ 8, %if.else ], [ 4, %if.then ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 1
  br label %if.end

if.else:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 2
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between ptr and gep with ptr as base and nonconstant offset
define void @test_phi_onegep_nonconst_offset(i32* %ptr, i32 %value, i32 %off) {
; CHECK-LABEL: @test_phi_onegep_nonconst_offset
; CHECK-NOT: phi i32* [ %ptr, %entry ], [ %gep, %if.then ]
; CHECK: phi i32 [ %off, %if.then ], [ 0, %entry ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %gep = getelementptr inbounds i32, i32* %ptr, i32 %off
  br label %if.end

if.end:
  %phi = phi i32* [ %ptr, %entry ], [ %gep, %if.then ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between two geps with same base, different nonconstant offsets
define void @test_phi_twogep_nonconst_offset(i32* %ptr, i32 %value, i32 %off1, i32 %off2) {
; CHECK-LABEL: @test_phi_twogep_nonconst_offset
; CHECK-NOT: phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
; CHECK: phi i32 [ %off2, %if.else ], [ %off1, %if.then ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 %off1
  br label %if.end

if.else:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 %off2
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between two geps with different base, same constant offset
define void @test_phi_twogep_base(i32* %ptr1, i32* %ptr2, i32 %value) {
; CHECK-LABEL: @test_phi_twogep_base
; CHECK-NOT: phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
; CHECK: phi i32* [ %ptr2, %if.else ], [ %ptr1, %if.then ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* %ptr1, i32 1
  br label %if.end

if.else:
  %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 1
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between two geps with different base global variables, same constant offset
define void @test_phi_twogep_base_gv(i32 %value) {
; CHECK-LABEL: @test_phi_twogep_base_gv
; CHECK-NOT: phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
; CHECK: phi i32* [ @gv2, %if.else ], [ @gv1, %if.then ]
entry:
  %cmp = icmp sgt i32 %value, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* @gv1, i32 1
  br label %if.end

if.else:
  %gep2 = getelementptr inbounds i32, i32* @gv2, i32 1
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep2, %if.else ]
  store i32 %value, i32* %phi, align 4
  ret void
}

; Phi selects between ptr and gep with ptr as base and constant offset
define void @test_select_onegep_offset(i32* %ptr, i32 %value) {
; CHECK-LABEL: @test_select_onegep_offset
; CHECK-NOT: select i1 %cmp, i32* %ptr, i32* %gep
; CHECK: select i1 %cmp, i32 0, i32 4
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep = getelementptr inbounds i32, i32* %ptr, i32 1
  %select = select i1 %cmp, i32* %ptr, i32* %gep
  store i32 %value, i32* %select, align 4
  ret void
}

; Select between two geps with same base, different constant offsets
define void @test_select_twogep_offset(i32* %ptr, i32 %value) {
; CHECK-LABEL: @test_select_twogep_offset
; CHECK-NOT: select i1 %cmp, i32* %gep1, i32* %gep2
; CHECK: select i1 %cmp, i32 4, i32 8
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 1
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 2
  %select = select i1 %cmp, i32* %gep1, i32* %gep2
  store i32 %value, i32* %select, align 4
  ret void
}

; Select between ptr and gep with ptr as base and nonconstant offset
define void @test_select_onegep_nonconst_offset(i32* %ptr, i32 %value, i32 %off) {
; CHECK-LABEL: @test_select_onegep_nonconst_offset
; CHECK-NOT: select i1 %cmp, i32* %ptr, i32* %gep
; CHECK: select i1 %cmp, i32 0, i32 %off
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep = getelementptr inbounds i32, i32* %ptr, i32 %off
  %select = select i1 %cmp, i32* %ptr, i32* %gep
  store i32 %value, i32* %select, align 4
  ret void
}

; Select between two geps with same base, different nonconstant offsets
define void @test_select_twogep_nonconst_offset(i32* %ptr, i32 %value, i32 %off1, i32 %off2) {
; CHECK-LABEL: @test_select_twogep_nonconst_offset
; CHECK-NOT: select i1 %cmp, i32* %gep1, i32* %gep2
; CHECK: select i1 %cmp, i32 %off1, i32 %off2
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 %off1
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 %off2
  %select = select i1 %cmp, i32* %gep1, i32* %gep2
  store i32 %value, i32* %select, align 4
  ret void
}

; Select between two geps with different base, same constant offset
define void @test_select_twogep_base(i32* %ptr1, i32* %ptr2, i32 %value) {
; CHECK-LABEL: @test_select_twogep_base
; CHECK-NOT: select i1 %cmp, i32* %gep1, i32* %gep2
; CHECK: select i1 %cmp, i32* %ptr1, i32* %ptr2
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep1 = getelementptr inbounds i32, i32* %ptr1, i32 1
  %gep2 = getelementptr inbounds i32, i32* %ptr2, i32 1
  %select = select i1 %cmp, i32* %gep1, i32* %gep2
  store i32 %value, i32* %select, align 4
  ret void
}

; Select between two geps with different base global variables, same constant offset
define void @test_select_twogep_base_gv(i32 %value) {
; CHECK-LABEL: @test_select_twogep_base_gv
; CHECK-NOT: select i1 %cmp, i32* %gep1, i32* %gep2
; CHECK: select i1 %cmp, i32* @gv1, i32* @gv2
entry:
  %cmp = icmp sgt i32 %value, 0
  %gep1 = getelementptr inbounds i32, i32* @gv1, i32 1
  %gep2 = getelementptr inbounds i32, i32* @gv2, i32 1
  %select = select i1 %cmp, i32* %gep1, i32* %gep2
  store i32 %value, i32* %select, align 4
  ret void
}

; If the phi is in a different block to where the gep will be, the phi goes where
; the original phi was not where the gep is.
; CHECK-LABEL: @test_phi_different_block
; CHECK-LABEL: if1.end
; CHECK-NOT: phi i32* [ %ptr, %entry ], [ %gep, %if1.then ]
; CHECK: phi i32 [ 4, %if1.then ], [ 0, %entry ]
define void @test_phi_different_block(i32* %ptr, i32 %value1, i32 %value2) {
entry:
  %cmp1 = icmp sgt i32 %value1, 0
  br i1 %cmp1, label %if1.then, label %if1.end

if1.then:
  %gep = getelementptr inbounds i32, i32* %ptr, i32 1
  br label %if1.end

if1.end:
  %phi = phi i32* [ %ptr, %entry ], [ %gep, %if1.then ]
  %cmp2 = icmp sgt i32 %value2, 0
  br i1 %cmp2, label %if2.then, label %if2.end

if2.then:
  store i32 %value1, i32* %ptr, align 4
  br label %if2.end

if2.end:
  store i32 %value2, i32* %phi, align 4
  ret void
}

; A phi with three incoming values should be optimised
; CHECK-LABEL: @test_phi_threegep
; CHECK-NOT: phi i32* [ %gep1, %if.then ], [ %gep2, %if.else.then ], [ %gep3, %if.else.else ]
; CHECK: phi i32 [ 12, %if.else.else ], [ 8, %if.else.then ], [ 4, %if.then ]
define void @test_phi_threegep(i32* %ptr, i32 %value1, i32 %value2) {
entry:
  %cmp1 = icmp sgt i32 %value1, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 1
  br label %if.end

if.else:
  %cmp2 = icmp sgt i32 %value2, 0
  br i1 %cmp2, label %if.else.then, label %if.else.else

if.else.then:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 2
  br label %if.end

if.else.else:
  %gep3 = getelementptr inbounds i32, i32* %ptr, i32 3
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep2, %if.else.then ], [ %gep3, %if.else.else ]
  store i32 %value1, i32* %phi, align 4
  ret void
}

; A phi with two incoming values but three geps due to nesting should be
; optimised
; CHECK-LABEL: @test_phi_threegep_nested
; CHECK: %[[PHI:[a-z0-9_]+]] = phi i32 [ 12, %if.else.else ], [ 8, %if.else.then ]
; CHECK: phi i32 [ %[[PHI]], %if.else.end ], [ 4, %if.then ]
define void @test_phi_threegep_nested(i32* %ptr, i32 %value1, i32 %value2) {
entry:
  %cmp1 = icmp sgt i32 %value1, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 1
  br label %if.end

if.else:
  %cmp2 = icmp sgt i32 %value2, 0
  br i1 %cmp2, label %if.else.then, label %if.else.else

if.else.then:
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 2
  br label %if.else.end

if.else.else:
  %gep3 = getelementptr inbounds i32, i32* %ptr, i32 3
  br label %if.else.end

if.else.end:
  %gep4 = phi i32* [ %gep2, %if.else.then ], [ %gep3, %if.else.else ]
  store i32 %value2, i32* %ptr, align 4
  br label %if.end

if.end:
  %phi = phi i32* [ %gep1, %if.then ], [ %gep4, %if.else.end ]
  store i32 %value1, i32* %phi, align 4
  ret void
}

; A nested select is expected to be optimised
; CHECK-LABEL: @test_nested_select
; CHECK: %[[SELECT:[a-z0-9_]+]] = select i1 %cmp2, i32 4, i32 8
; CHECK: select i1 %cmp1, i32 4, i32 %[[SELECT]]
define void @test_nested_select(i32* %ptr, i32 %value1, i32 %value2) {
entry:
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 1
  %gep2 = getelementptr inbounds i32, i32* %ptr, i32 2
  %cmp1 = icmp sgt i32 %value1, 0
  %cmp2 = icmp sgt i32 %value2, 0
  %select1 = select i1 %cmp2, i32* %gep1, i32* %gep2
  %select2 = select i1 %cmp1, i32* %gep1, i32* %select1
  store i32 %value1, i32* %select2, align 4
  ret void
}

; Scaling the offset by a different amount is expected not to be optimised
; CHECK-LABEL: @test_select_different_scale
; CHECK: select i1 %cmp, i32* %gep1, i32* %castgep
define void @test_select_different_scale(i32* %ptr, i32 %value, i32 %off) {
entry:
  %cmp = icmp sgt i32 %value, 0
  %castptr = bitcast i32* %ptr to i16*
  %gep1 = getelementptr inbounds i32, i32* %ptr, i32 %off
  %gep2 = getelementptr inbounds i16, i16* %castptr, i32 %off
  %castgep = bitcast i16* %gep2 to i32*
  %select = select i1 %cmp, i32* %gep1, i32* %castgep
  store i32 %value, i32* %select, align 4
  ret void
}

; A select between two values is already the best we can do
; CHECK-LABEL: @test_select_trivial
; CHECK: select i1 %cmp, i32* %ptr1, i32* %ptr2
define void @test_select_trivial(i32* %ptr1, i32* %ptr2, i32 %value) {
entey:
  %cmp = icmp sgt i32 %value, 0
  %select = select i1 %cmp, i32* %ptr1, i32* %ptr2
  store i32 %value, i32* %select, align 4
  ret void
}

; A select between two global variables is already the best we can do
; CHECK-LABEL: @test_select_trivial_gv
; CHECK: select i1 %cmp, i32* @gv1, i32* @gv2
define void @test_select_trivial_gv(i32 %value) {
entey:
  %cmp = icmp sgt i32 %value, 0
  %select = select i1 %cmp, i32* @gv1, i32* @gv2
  store i32 %value, i32* %select, align 4
  ret void
}

; Same for a select between a value and global variable
; CHECK-LABEL: @test_select_trivial_ptr_gv
; CHECK: select i1 %cmp, i32* %ptr, i32* @gv2
define void @test_select_trivial_ptr_gv(i32* %ptr, i32 %value) {
entry:
  %cmp = icmp sgt i32 %value, 0
  %select = select i1 %cmp, i32* %ptr, i32* @gv2
  store i32 %value, i32* %select, align 4
  ret void
}

; Same for a select between a global variable and null, though the test needs to
; be a little more complicated to avoid dereferencing a potential null pointer
; CHECK-LABEL: @test_select_trivial_gv_null
; CHECK: select i1 %cmp.i, i32* @gv1, i32* null
define void @test_select_trivial_gv_null(){
entry:
  %gv1_val = load i32, i32* @gv1, align 4
  %cmp.i = icmp eq i32 %gv1_val, 0
  %spec.select.i = select i1 %cmp.i, i32* @gv1, i32* null
  br i1 %cmp.i, label %if.then, label %if.end

if.then:
  %val = load i32, i32* %spec.select.i, align 4
  %inc = add nsw i32 %val, 1
  store i32 %inc, i32* %spec.select.i, align 4
  br label %if.end

if.end:
  ret void
}

; Same for a select between a value and null
; CHECK-LABEL: @test_select_trivial_ptr_null
; CHECK: select i1 %cmp.i, i32* %ptr, i32* null
define void @test_select_trivial_ptr_null(i32* %ptr){
entry:
  %gv1_val = load i32, i32* %ptr, align 4
  %cmp.i = icmp eq i32 %gv1_val, 0
  %spec.select.i = select i1 %cmp.i, i32* %ptr, i32* null
  br i1 %cmp.i, label %if.then, label %if.end

if.then:
  %val = load i32, i32* %spec.select.i, align 4
  %inc = add nsw i32 %val, 1
  store i32 %inc, i32* %spec.select.i, align 4
  br label %if.end

if.end:
  ret void
}
