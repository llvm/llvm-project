; RUN: opt -passes=loop-unroll %s -o - | llc -O3 - -mtriple=aarch64-unknown-unknown -mcpu=neoverse-v2 -o - | FileCheck %s

define i64 @sabalb_i32_to_i64_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalb_i32_to_i64_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlb
; CHECK: sabalb z0.d
; CHECK: sabalb z1.d
; CHECK: sabalb z2.d
; CHECK: add	z0.d, z2.d, z0.d
; CHECK: add	z0.d, z0.d, z1.d
; CHECK: uaddv	d0, p0, z0.d
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 2 x i64> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <vscale x 4 x i32>, ptr %ptr1_i, align 1
  %b = load <vscale x 4 x i32>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 2 x i64> @llvm.aarch64.sve.sabalb.nxv2i64(<vscale x 2 x i64> %acc_phi,
                                                                       <vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i64 @llvm.vector.reduce.add.nxv2i64(<vscale x 2 x i64> %acc_next)
  ret i64 %reduce
}

declare <vscale x  2 x i64> @llvm.aarch64.sve.sabalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare i64 @llvm.vector.reduce.add.nxv2i64(<vscale x 2 x i64>)

define i32 @sabalb_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalb_i16_to_i32_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlb
; CHECK: sabalb z0.s
; CHECK: sabalb z1.s
; CHECK: sabalb z2.s
; CHECK: add	z0.s, z2.s, z0.s
; CHECK: add	z0.s, z0.s, z1.s
; CHECK: uaddv	d0, p0, z0.s
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <vscale x 8 x i16>, ptr %ptr1_i, align 1
  %b = load <vscale x 8 x i16>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 4 x i32> @llvm.aarch64.sve.sabalb.nxv4i32(<vscale x 4 x i32> %acc_phi,
                                                                       <vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b)
  
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %acc_next)
  ret i32 %reduce
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.sabalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32>)

define i16 @sabalb_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalb_i8_to_i16_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlb
; CHECK: sabalb z0.h
; CHECK: sabalb z1.h
; CHECK: sabalb z2.h
; CHECK: add	z0.h, z2.h, z0.h
; CHECK: add	z0.h, z0.h, z1.h
; CHECK: uaddv	d0, p0, z0.h
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <vscale x 16 x i8>, ptr %ptr1_i, align 1
  %b = load <vscale x 16 x i8>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 8 x i16> @llvm.aarch64.sve.sabalb.nxv8i16(<vscale x 8 x i16> %acc_phi,
                                                                       <vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 256
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16> %acc_next)
  ret i16 %reduce
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16>)

define i64 @sabalt_i32_to_i64_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalt_i32_to_i64_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlt
; CHECK: sabalt z0.d
; CHECK: sabalt z1.d
; CHECK: sabalt z2.d
; CHECK: add	z0.d, z2.d, z0.d
; CHECK: add	z0.d, z0.d, z1.d
; CHECK: uaddv	d0, p0, z0.d
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 2 x i64> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <vscale x 4 x i32>, ptr %ptr1_i, align 1
  %b = load <vscale x 4 x i32>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 2 x i64> @llvm.aarch64.sve.sabalt.nxv2i64(<vscale x 2 x i64> %acc_phi,
                                                                       <vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i64 @llvm.vector.reduce.add.nxv2i64(<vscale x 2 x i64> %acc_next)
  ret i64 %reduce
}

declare <vscale x  2 x i64> @llvm.aarch64.sve.sabalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

define i32 @sabalt_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalt_i16_to_i32_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlt
; CHECK: sabalt z0.s
; CHECK: sabalt z1.s
; CHECK: sabalt z2.s
; CHECK: add	z0.s, z2.s, z0.s
; CHECK: add	z0.s, z0.s, z1.s
; CHECK: uaddv	d0, p0, z0.s
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <vscale x 8 x i16>, ptr %ptr1_i, align 1
  %b = load <vscale x 8 x i16>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 4 x i32> @llvm.aarch64.sve.sabalt.nxv4i32(<vscale x 4 x i32> %acc_phi,
                                                                       <vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b)
  
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %acc_next)
  ret i32 %reduce
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.sabalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)

define i16 @sabalt_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabalt_i8_to_i16_accumulation
entry:
  br label %loop
loop:
; CHECK: sabdlt
; CHECK: sabalt z0.h
; CHECK: sabalt z1.h
; CHECK: sabalt z2.h
; CHECK: add	z0.h, z2.h, z0.h
; CHECK: add	z0.h, z0.h, z1.h
; CHECK: uaddv	d0, p0, z0.h
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <vscale x 16 x i8>, ptr %ptr1_i, align 1
  %b = load <vscale x 16 x i8>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 8 x i16> @llvm.aarch64.sve.sabalt.nxv8i16(<vscale x 8 x i16> %acc_phi,
                                                                       <vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 256
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16> %acc_next)
  ret i16 %reduce
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.sabalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)

define i64 @uabalb_i32_to_i64_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalb_i32_to_i64_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlb
; CHECK: uabalb z0.d
; CHECK: uabalb z1.d
; CHECK: uabalb z2.d
; CHECK: add	z0.d, z2.d, z0.d
; CHECK: add	z0.d, z0.d, z1.d
; CHECK: uaddv	d0, p0, z0.d
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 2 x i64> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <vscale x 4 x i32>, ptr %ptr1_i, align 1
  %b = load <vscale x 4 x i32>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 2 x i64> @llvm.aarch64.sve.uabalb.nxv2i64(<vscale x 2 x i64> %acc_phi,
                                                                       <vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i64 @llvm.vector.reduce.add.nxv2i64(<vscale x 2 x i64> %acc_next)
  ret i64 %reduce
}

declare <vscale x  2 x i64> @llvm.aarch64.sve.uabalb.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

define i32 @uabalb_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalb_i16_to_i32_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlb
; CHECK: uabalb z0.s
; CHECK: uabalb z1.s
; CHECK: uabalb z2.s
; CHECK: add	z0.s, z2.s, z0.s
; CHECK: add	z0.s, z0.s, z1.s
; CHECK: uaddv	d0, p0, z0.s
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <vscale x 8 x i16>, ptr %ptr1_i, align 1
  %b = load <vscale x 8 x i16>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 4 x i32> @llvm.aarch64.sve.uabalb.nxv4i32(<vscale x 4 x i32> %acc_phi,
                                                                       <vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b)
  
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %acc_next)
  ret i32 %reduce
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.uabalb.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)

define i16 @uabalb_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalb_i8_to_i16_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlb
; CHECK: uabalb z0.h
; CHECK: uabalb z1.h
; CHECK: uabalb z2.h
; CHECK: add	z0.h, z2.h, z0.h
; CHECK: add	z0.h, z0.h, z1.h
; CHECK: uaddv	d0, p0, z0.h
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <vscale x 16 x i8>, ptr %ptr1_i, align 1
  %b = load <vscale x 16 x i8>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalb.nxv8i16(<vscale x 8 x i16> %acc_phi,
                                                                       <vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 256
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16> %acc_next)
  ret i16 %reduce
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabalb.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)

define i64 @uabalt_i32_to_i64_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalt_i32_to_i64_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlt
; CHECK: uabalt z0.d
; CHECK: uabalt z1.d
; CHECK: uabalt z2.d
; CHECK: add	z0.d, z2.d, z0.d
; CHECK: add	z0.d, z0.d, z1.d
; CHECK: uaddv	d0, p0, z0.d
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 2 x i64> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <vscale x 4 x i32>, ptr %ptr1_i, align 1
  %b = load <vscale x 4 x i32>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 2 x i64> @llvm.aarch64.sve.uabalt.nxv2i64(<vscale x 2 x i64> %acc_phi,
                                                                       <vscale x 4 x i32> %a,
                                                                       <vscale x 4 x i32> %b)
  
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i64 @llvm.vector.reduce.add.nxv2i64(<vscale x 2 x i64> %acc_next)
  ret i64 %reduce
}

declare <vscale x  2 x i64> @llvm.aarch64.sve.uabalt.nxv2i64(<vscale x 2 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>)

define i32 @uabalt_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalt_i16_to_i32_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlt
; CHECK: uabalt z0.s
; CHECK: uabalt z1.s
; CHECK: uabalt z2.s
; CHECK: add	z0.s, z2.s, z0.s
; CHECK: add	z0.s, z0.s, z1.s
; CHECK: uaddv	d0, p0, z0.s
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <vscale x 8 x i16>, ptr %ptr1_i, align 1
  %b = load <vscale x 8 x i16>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 4 x i32> @llvm.aarch64.sve.uabalt.nxv4i32(<vscale x 4 x i32> %acc_phi,
                                                                       <vscale x 8 x i16> %a,
                                                                       <vscale x 8 x i16> %b)
  
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %acc_next)
  ret i32 %reduce
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.uabalt.nxv4i32(<vscale x 4 x i32>, <vscale x 8 x i16>, <vscale x 8 x i16>)

define i16 @uabalt_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalt_i8_to_i16_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlt
; CHECK: uabalt z0.h
; CHECK: uabalt z1.h
; CHECK: uabalt z2.h
; CHECK: add	z0.h, z2.h, z0.h
; CHECK: add	z0.h, z0.h, z1.h
; CHECK: uaddv	d0, p0, z0.h
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <vscale x 16 x i8>, ptr %ptr1_i, align 1
  %b = load <vscale x 16 x i8>, ptr %ptr2_i, align 1
  %acc_next = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalt.nxv8i16(<vscale x 8 x i16> %acc_phi,
                                                                       <vscale x 16 x i8> %a,
                                                                       <vscale x 16 x i8> %b)
  
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 256
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = tail call i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16> %acc_next)
  ret i16 %reduce
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.uabalt.nxv8i16(<vscale x 8 x i16>, <vscale x 16 x i8>, <vscale x 16 x i8>)

define i16 @uabalt_and_uabalb_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: uabalt_and_uabalb_accumulation
entry:
  br label %loop
loop:
; CHECK: uabdlt
; CHECK: uabdlb
; CHECK: uabalt z0.h
; CHECK: uabalt z2.h
; CHECK: uabalt z4.h
; CHECK: uabalb z1.h
; CHECK: uabalb z6.h
; CHECK: uabalb z5.h
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_hi_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next_hi, %loop ]
  %acc_lo_phi = phi <vscale x 8 x i16> [ zeroinitializer, %entry ], [ %acc_next_lo, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <vscale x 16 x i8>, ptr %ptr1_i, align 1
  %b = load <vscale x 16 x i8>, ptr %ptr2_i, align 1
  %acc_next_lo = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalb.nxv8i16(<vscale x 8 x i16> %acc_lo_phi,
                                                                         <vscale x 16 x i8> %a,
                                                                         <vscale x 16 x i8> %b)
  %acc_next_hi = call <vscale x 8 x i16> @llvm.aarch64.sve.uabalt.nxv8i16(<vscale x 8 x i16> %acc_hi_phi,
                                                                         <vscale x 16 x i8> %a,
                                                                         <vscale x 16 x i8> %b)
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 256
  br i1 %cmp, label %loop, label %exit
exit:
  %mask = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %acc_next = tail call <vscale x 8 x i16> @llvm.aarch64.sve.add.nxv8i16(<vscale x 8 x i1> %mask, <vscale x 8 x i16> %acc_next_lo, <vscale x 8 x i16> %acc_next_hi)
  %reduce = tail call i16 @llvm.vector.reduce.add.nxv8i16(<vscale x 8 x i16> %acc_next)
  ret i16 %reduce
}

declare <vscale x 8 x i16> @llvm.aarch64.sve.add.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)