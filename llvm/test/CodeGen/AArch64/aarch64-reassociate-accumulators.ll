; RUN: opt -passes=loop-unroll %s -o - | llc -O3 - -mtriple=arm64e-apple-darwin -o - | FileCheck %s


define i16 @sabal_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
  ; CHECK-LABEL: sabal_i8_to_i16_accumulation
entry:
  br label %loop

loop:
; CHECK: sabdl.8h v1
; CHECK: sabdl.8h v0
; CHECK: sabdl.8h v2
; CHECK: sabal.8h v1
; CHECK: sabal.8h v0
; CHECK: sabal.8h v2
; CHECK: sabal.8h v1
; CHECK: sabal.8h v0
; CHECK: add.8h v1, v2, v1
; CHECK: add.8h v0, v1, v0
; CHECK: addv.8h

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <8 x i8>, <8 x i8>* %ptr1_i, align 1
  %b = load <8 x i8>, <8 x i8>* %ptr2_i, align 1
  %vabd = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %vabd_ext = zext <8 x i8> %vabd to <8 x i16>
  %acc_next = add <8 x i16> %vabd_ext, %acc_phi
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %acc_next)
  ret i16 %reduce
}

; Declare the signed absolute difference intrinsic
declare <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8>, <8 x i8>)
declare i16 @llvm.vector.reduce.add.v8i16(<8 x i16>)


define i32 @sabal_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: sabal_i16_to_i32_accumulation
entry:
  br label %loop

loop:
; CHECK: sabdl.4s v1
; CHECK: sabdl.4s v0
; CHECK: sabdl.4s v2
; CHECK: sabal.4s v1
; CHECK: sabal.4s v0
; CHECK: sabal.4s v2
; CHECK: sabal.4s v1
; CHECK: sabal.4s v0
; CHECK: add.4s v1, v2, v1
; CHECK: add.4s v0, v1, v0
; CHECK: addv.4s

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <4 x i16>, <4 x i16>* %ptr1_i, align 1
  %b = load <4 x i16>, <4 x i16>* %ptr2_i, align 1
  %vabd = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %a, <4 x i16> %b)
  %vmov = zext <4 x i16> %vabd to <4 x i32>
  %acc_next = add <4 x i32> %vmov, %acc_phi
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 32
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %acc_next)
  ret i32 %reduce
}

declare <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16>, <4 x i16>)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)

define i16 @uabal2_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uabal2_accumulation
entry:
  br label %loop

loop:
; CHECK:  uabdl2.8h       v4
; CHECK:  uabdl.8h        v1 
; CHECK:  uabdl2.8h       v24
; CHECK:  uabdl2.8h       v25
; CHECK:  uabal2.8h       v4 
; CHECK:  uabal2.8h       v24
; CHECK:  uabal2.8h       v25
; CHECK:  uabal2.8h       v4
; CHECK:  uabal2.8h       v24
; CHECK:  add.8h          v4, v25, v4
; CHECK:  add.8h          v4, v4, v24
; CHECK:  uabdl.8h        v0
; CHECK:  uabdl.8h        v2
; CHECK:  uabal.8h        v1
; CHECK:  uabal.8h        v0
; CHECK:  uabal.8h        v2
; CHECK:  uabal.8h        v1
; CHECK:  uabal.8h        v0
; CHECK:  add.8h          v1, v2, v1
; CHECK:  add.8h          v0, v1, v0
; CHECK:  add.8h          v0, v4, v0
; CHECK:  addv.8h         h0, v0

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi_hi = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next_hi, %loop ]
  %acc_phi_lo = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next_lo, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <16 x i8>, <16 x i8>* %ptr1_i, align 1
  %b = load <16 x i8>, <16 x i8>* %ptr2_i, align 1
  %a_hi = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b_hi = shufflevector <16 x i8> %b, <16 x i8> zeroinitializer, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %a_lo = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %b_lo = shufflevector <16 x i8> %b, <16 x i8> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vabd_hi = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a_hi, <8 x i8> %b_hi)
  %vabd_lo = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a_lo, <8 x i8> %b_lo)
  %vmov_hi = zext <8 x i8> %vabd_hi to <8 x i16>
  %vmov_lo = zext <8 x i8> %vabd_lo to <8 x i16>
  %acc_next_hi = add <8 x i16> %vmov_hi, %acc_phi_hi
  %acc_next_lo = add <8 x i16> %vmov_lo, %acc_phi_lo
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit

exit:
  %hi_plus_lo = add <8 x i16> %acc_next_hi, %acc_next_lo
  %reduce = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %hi_plus_lo)
  ret i16 %reduce
}

define i32 @uaba_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uaba_accumulation
entry:
  br label %loop

loop:
; CHECK: uabd.4s v0
; CHECK: uabd.4s v1
; CHECK: uabd.4s v2
; CHECK: uaba.4s v0
; CHECK: uaba.4s v1
; CHECK: uaba.4s v2
; CHECK: uaba.4s v0
; CHECK: uaba.4s v1
; CHECK: add.4s v0, v2, v0
; CHECK: add.4s v0, v0, v1
; CHECK: addv.4s 

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <4 x i32>, <4 x i32>* %ptr1_i, align 1
  %b = load <4 x i32>, <4 x i32>* %ptr2_i, align 1
  %vabd = tail call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> %a, <4 x i32> %b)
  %acc_next = add <4 x i32> %acc_phi, %vabd
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 32
  br i1 %cmp, label %loop, label %exit
exit:

  %reduce = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %acc_next)
  ret i32 %reduce
}

declare <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define i32 @saba_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: saba_accumulation
entry:
  br label %loop

loop:
; CHECK: sabd.4s v0
; CHECK: sabd.4s v1
; CHECK: sabd.4s v2
; CHECK: saba.4s v0
; CHECK: saba.4s v1
; CHECK: saba.4s v2
; CHECK: saba.4s v0
; CHECK: saba.4s v1
; CHECK: add.4s v0, v2, v0
; CHECK: add.4s v0, v0, v1
; CHECK: addv.4s

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  ; Load values from ptr1 and ptr2
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <4 x i32>, <4 x i32>* %ptr1_i, align 1
  %b = load <4 x i32>, <4 x i32>* %ptr2_i, align 1
  ; Perform the intrinsic operation
  %vabd = tail call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> %a, <4 x i32> %b)
  %acc_next = add <4 x i32> %acc_phi, %vabd
  ; Increment loop counter and check the bound
  %next_i = add i32 %i, 4 
  %cmp = icmp slt i32 %next_i, 32  
  br i1 %cmp, label %loop, label %exit

exit:                            
  %reduce = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %acc_next)
  ret i32 %reduce
}

declare <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define i32 @uaba_v2i32_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uaba_v2i32_accumulation
entry:
  br label %loop

loop:
; CHECK: uabd.2s v0
; CHECK: uabd.2s v1
; CHECK: uabd.2s v2
; CHECK: uaba.2s v0
; CHECK: uaba.2s v1
; CHECK: uaba.2s v2
; CHECK: uaba.2s v0
; CHECK: uaba.2s v1
; CHECK: add.2s v0, v2, v0
; CHECK: add.2s v0, v0, v1
; CHECK: addp.2s

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <2 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i32, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i32, ptr %ptr2, i32 %i
  %a = load <2 x i32>, <2 x i32>* %ptr1_i, align 1
  %b = load <2 x i32>, <2 x i32>* %ptr2_i, align 1
  %vabd = tail call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %a, <2 x i32> %b)
  %acc_next = add <2 x i32> %acc_phi, %vabd
  %next_i = add i32 %i, 2
  %cmp = icmp slt i32 %next_i, 16
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %acc_next)
  ret i32 %reduce
}

define i8 @uaba_v8i8_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uaba_v8i8_accumulation
entry:
  br label %loop

loop:
; CHECK: uabd.8b v0
; CHECK: uabd.8b v1
; CHECK: uabd.8b v2
; CHECK: uaba.8b v0
; CHECK: uaba.8b v1
; CHECK: uaba.8b v2
; CHECK: uaba.8b v0
; CHECK: uaba.8b v1
; CHECK: add.8b v0, v2, v0
; CHECK: add.8b v0, v0, v1
; CHECK: addv.8b

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i8> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <8 x i8>, <8 x i8>* %ptr1_i, align 1
  %b = load <8 x i8>, <8 x i8>* %ptr2_i, align 1
  %vabd = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %acc_next = add <8 x i8> %acc_phi, %vabd
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i8 @llvm.vector.reduce.add.v8i8(<8 x i8> %acc_next)
  ret i8 %reduce
}

define i8 @uaba_v16i8_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uaba_v16i8_accumulation
entry:
  br label %loop

loop:
; CHECK: uabd.16b v0
; CHECK: uabd.16b v1
; CHECK: uabd.16b v2
; CHECK: uaba.16b v0
; CHECK: uaba.16b v1
; CHECK: uaba.16b v2
; CHECK: uaba.16b v0
; CHECK: uaba.16b v1
; CHECK: add.16b v0, v2, v0
; CHECK: add.16b v0, v0, v1
; CHECK: addv.16b

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <16 x i8> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <16 x i8>, <16 x i8>* %ptr1_i, align 1
  %b = load <16 x i8>, <16 x i8>* %ptr2_i, align 1
  %vabd = tail call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> %a, <16 x i8> %b)
  %acc_next = add <16 x i8> %acc_phi, %vabd
  %next_i = add i32 %i, 16
  %cmp = icmp slt i32 %next_i, 128
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %acc_next)
  ret i8 %reduce
}

define i16 @uaba_v8i16_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uaba_v8i16_accumulation
entry:
  br label %loop

loop:
; CHECK: uabd.8h v0
; CHECK: uabd.8h v1
; CHECK: uabd.8h v2
; CHECK: uaba.8h v0
; CHECK: uaba.8h v1
; CHECK: uaba.8h v2
; CHECK: uaba.8h v0
; CHECK: uaba.8h v1
; CHECK: add.8h v0, v2, v0
; CHECK: add.8h v0, v0, v1
; CHECK: addv.8h

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <8 x i16>, <8 x i16>* %ptr1_i, align 1
  %b = load <8 x i16>, <8 x i16>* %ptr2_i, align 1
  %vabd = tail call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %a, <8 x i16> %b)
  %acc_next = add <8 x i16> %acc_phi, %vabd
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %acc_next)
  ret i16 %reduce
}

define i8 @saba_v8i8_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: saba_v8i8_accumulation
entry:
  br label %loop

loop:
; CHECK: sabd.8b v0
; CHECK: sabd.8b v1
; CHECK: sabd.8b v2
; CHECK: saba.8b v0
; CHECK: saba.8b v1
; CHECK: saba.8b v2
; CHECK: saba.8b v0
; CHECK: saba.8b v1
; CHECK: add.8b v0, v2, v0
; CHECK: add.8b v0, v0, v1
; CHECK: addv.8b

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i8> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <8 x i8>, <8 x i8>* %ptr1_i, align 1
  %b = load <8 x i8>, <8 x i8>* %ptr2_i, align 1
  %vabd = tail call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %acc_next = add <8 x i8> %acc_phi, %vabd
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i8 @llvm.vector.reduce.add.v8i8(<8 x i8> %acc_next)
  ret i8 %reduce
}

define i16 @saba_v4i16_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: saba_v4i16_accumulation
entry:
  br label %loop
loop:
; CHECK: sabd.4h v0
; CHECK: sabd.4h v1
; CHECK: sabd.4h v2
; CHECK: saba.4h v0
; CHECK: saba.4h v1
; CHECK: saba.4h v2
; CHECK: saba.4h v0
; CHECK: saba.4h v1
; CHECK: add.4h v0, v2, v0
; CHECK: add.4h v0, v0, v1
; CHECK: addv.4h

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <4 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <4 x i16>, <4 x i16>* %ptr1_i, align 1
  %b = load <4 x i16>, <4 x i16>* %ptr2_i, align 1
  %vabd = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %a, <4 x i16> %b)
  %acc_next = add <4 x i16> %acc_phi, %vabd
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 32
  br i1 %cmp, label %loop, label %exit
exit:
  %reduce = call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %acc_next)
  ret i16 %reduce
}

define i16 @saba_v8i16_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: saba_v8i16_accumulation
entry:
  br label %loop

loop:
; CHECK: sabd.8h v0
; CHECK: sabd.8h v1
; CHECK: sabd.8h v2
; CHECK: saba.8h v0
; CHECK: saba.8h v1
; CHECK: saba.8h v2
; CHECK: saba.8h v0
; CHECK: saba.8h v1
; CHECK: add.8h v0, v2, v0
; CHECK: add.8h v0, v0, v1
; CHECK: addv.8h

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <8 x i16>, <8 x i16>* %ptr1_i, align 1
  %b = load <8 x i16>, <8 x i16>* %ptr2_i, align 1
  %vabd = tail call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> %a, <8 x i16> %b)
  %acc_next = add <8 x i16> %acc_phi, %vabd
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %acc_next)
  ret i16 %reduce
}

define i16 @uabal_i8_to_i16_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uabal_i8_to_i16_accumulation
entry:
  br label %loop

loop:
; CHECK: uabdl.8h v1
; CHECK: uabdl.8h v0
; CHECK: uabdl.8h v2
; CHECK: uabal.8h v1
; CHECK: uabal.8h v0
; CHECK: uabal.8h v2
; CHECK: uabal.8h v1
; CHECK: uabal.8h v0
; CHECK: add.8h v1, v2, v1
; CHECK: add.8h v0, v1, v0
; CHECK: addv.8h

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i8, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i8, ptr %ptr2, i32 %i
  %a = load <8 x i8>, <8 x i8>* %ptr1_i, align 1
  %b = load <8 x i8>, <8 x i8>* %ptr2_i, align 1
  %vabd = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %vmov = zext <8 x i8> %vabd to <8 x i16>
  %acc_next = add <8 x i16> %vmov, %acc_phi
  %next_i = add i32 %i, 8
  %cmp = icmp slt i32 %next_i, 64
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %acc_next)
  ret i16 %reduce
}

define i32 @uabal_i16_to_i32_accumulation(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: uabal_i16_to_i32_accumulation
entry:
  br label %loop

loop:
; CHECK: uabdl.4s v1
; CHECK: uabdl.4s v0
; CHECK: uabdl.4s v2
; CHECK: uabal.4s v1
; CHECK: uabal.4s v0
; CHECK: uabal.4s v2
; CHECK: uabal.4s v1
; CHECK: uabal.4s v0
; CHECK: add.4s v1, v2, v1
; CHECK: add.4s v0, v1, v0
; CHECK: addv.4s

  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %acc_phi = phi <4 x i32> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %ptr1_i = getelementptr i16, ptr %ptr1, i32 %i
  %ptr2_i = getelementptr i16, ptr %ptr2, i32 %i
  %a = load <4 x i16>, <4 x i16>* %ptr1_i, align 1
  %b = load <4 x i16>, <4 x i16>* %ptr2_i, align 1
  %vabd = tail call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %a, <4 x i16> %b)
  %vmov = zext <4 x i16> %vabd to <4 x i32>
  %acc_next = add <4 x i32> %vmov, %acc_phi
  %next_i = add i32 %i, 4
  %cmp = icmp slt i32 %next_i, 32
  br i1 %cmp, label %loop, label %exit

exit:
  %reduce = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %acc_next)
  ret i32 %reduce
}
