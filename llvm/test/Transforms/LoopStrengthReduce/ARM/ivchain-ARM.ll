; RUN: llc < %s -O3 -march=thumb -mcpu=cortex-a9 | FileCheck %s -check-prefix=A9

; @simple is the most basic chain of address induction variables. Chaining
; saves at least one register and avoids complex addressing and setup
; code.
;
; A9: @simple
; no expensive address computation in the preheader
; A9: lsl
; A9-NOT: lsl
; A9: %loop
; no complex address modes
; A9-NOT: lsl
define i32 @simple(i32* %a, i32* %b, i32 %x) nounwind {
entry:
  br label %loop
loop:
  %iv = phi i32* [ %a, %entry ], [ %iv4, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s4, %loop ]
  %v = load i32* %iv
  %iv1 = getelementptr inbounds i32* %iv, i32 %x
  %v1 = load i32* %iv1
  %iv2 = getelementptr inbounds i32* %iv1, i32 %x
  %v2 = load i32* %iv2
  %iv3 = getelementptr inbounds i32* %iv2, i32 %x
  %v3 = load i32* %iv3
  %s1 = add i32 %s, %v
  %s2 = add i32 %s1, %v1
  %s3 = add i32 %s2, %v2
  %s4 = add i32 %s3, %v3
  %iv4 = getelementptr inbounds i32* %iv3, i32 %x
  %cmp = icmp eq i32* %iv4, %b
  br i1 %cmp, label %exit, label %loop
exit:
  ret i32 %s4
}

; @user is not currently chained because the IV is live across memory ops.
;
; A9: @user
; stride multiples computed in the preheader
; A9: lsl
; A9: lsl
; A9: %loop
; complex address modes
; A9: lsl
; A9: lsl
define i32 @user(i32* %a, i32* %b, i32 %x) nounwind {
entry:
  br label %loop
loop:
  %iv = phi i32* [ %a, %entry ], [ %iv4, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s4, %loop ]
  %v = load i32* %iv
  %iv1 = getelementptr inbounds i32* %iv, i32 %x
  %v1 = load i32* %iv1
  %iv2 = getelementptr inbounds i32* %iv1, i32 %x
  %v2 = load i32* %iv2
  %iv3 = getelementptr inbounds i32* %iv2, i32 %x
  %v3 = load i32* %iv3
  %s1 = add i32 %s, %v
  %s2 = add i32 %s1, %v1
  %s3 = add i32 %s2, %v2
  %s4 = add i32 %s3, %v3
  %iv4 = getelementptr inbounds i32* %iv3, i32 %x
  store i32 %s4, i32* %iv
  %cmp = icmp eq i32* %iv4, %b
  br i1 %cmp, label %exit, label %loop
exit:
  ret i32 %s4
}

; @extrastride is a slightly more interesting case of a single
; complete chain with multiple strides. The test case IR is what LSR
; used to do, and exactly what we don't want to do. LSR's new IV
; chaining feature should now undo the damage.
;
; A9: extrastride:
; no spills
; A9-NOT: str
; only one stride multiple in the preheader
; A9: lsl
; A9-NOT: {{str r|lsl}}
; A9: %for.body{{$}}
; no complex address modes or reloads
; A9-NOT: {{ldr .*[sp]|lsl}}
define void @extrastride(i8* nocapture %main, i32 %main_stride, i32* nocapture %res, i32 %x, i32 %y, i32 %z) nounwind {
entry:
  %cmp8 = icmp eq i32 %z, 0
  br i1 %cmp8, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %add.ptr.sum = shl i32 %main_stride, 1 ; s*2
  %add.ptr1.sum = add i32 %add.ptr.sum, %main_stride ; s*3
  %add.ptr2.sum = add i32 %x, %main_stride ; s + x
  %add.ptr4.sum = shl i32 %main_stride, 2 ; s*4
  %add.ptr3.sum = add i32 %add.ptr2.sum, %add.ptr4.sum ; total IV stride = s*5+x
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %main.addr.011 = phi i8* [ %main, %for.body.lr.ph ], [ %add.ptr6, %for.body ]
  %i.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %res.addr.09 = phi i32* [ %res, %for.body.lr.ph ], [ %add.ptr7, %for.body ]
  %0 = bitcast i8* %main.addr.011 to i32*
  %1 = load i32* %0, align 4
  %add.ptr = getelementptr inbounds i8* %main.addr.011, i32 %main_stride
  %2 = bitcast i8* %add.ptr to i32*
  %3 = load i32* %2, align 4
  %add.ptr1 = getelementptr inbounds i8* %main.addr.011, i32 %add.ptr.sum
  %4 = bitcast i8* %add.ptr1 to i32*
  %5 = load i32* %4, align 4
  %add.ptr2 = getelementptr inbounds i8* %main.addr.011, i32 %add.ptr1.sum
  %6 = bitcast i8* %add.ptr2 to i32*
  %7 = load i32* %6, align 4
  %add.ptr3 = getelementptr inbounds i8* %main.addr.011, i32 %add.ptr4.sum
  %8 = bitcast i8* %add.ptr3 to i32*
  %9 = load i32* %8, align 4
  %add = add i32 %3, %1
  %add4 = add i32 %add, %5
  %add5 = add i32 %add4, %7
  %add6 = add i32 %add5, %9
  store i32 %add6, i32* %res.addr.09, align 4
  %add.ptr6 = getelementptr inbounds i8* %main.addr.011, i32 %add.ptr3.sum
  %add.ptr7 = getelementptr inbounds i32* %res.addr.09, i32 %y
  %inc = add i32 %i.010, 1
  %cmp = icmp eq i32 %inc, %z
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; @foldedidx is an unrolled variant of this loop:
;  for (unsigned long i = 0; i < len; i += s) {
;    c[i] = a[i] + b[i];
;  }
; where 's' can be folded into the addressing mode.
; Consequently, we should *not* form any chains.
;
; A9: foldedidx:
; A9: ldrb.w {{r[0-9]|lr}}, [{{r[0-9]|lr}}, #3]
define void @foldedidx(i8* nocapture %a, i8* nocapture %b, i8* nocapture %c) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.07 = phi i32 [ 0, %entry ], [ %inc.3, %for.body ]
  %arrayidx = getelementptr inbounds i8* %a, i32 %i.07
  %0 = load i8* %arrayidx, align 1
  %conv5 = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8* %b, i32 %i.07
  %1 = load i8* %arrayidx1, align 1
  %conv26 = zext i8 %1 to i32
  %add = add nsw i32 %conv26, %conv5
  %conv3 = trunc i32 %add to i8
  %arrayidx4 = getelementptr inbounds i8* %c, i32 %i.07
  store i8 %conv3, i8* %arrayidx4, align 1
  %inc1 = or i32 %i.07, 1
  %arrayidx.1 = getelementptr inbounds i8* %a, i32 %inc1
  %2 = load i8* %arrayidx.1, align 1
  %conv5.1 = zext i8 %2 to i32
  %arrayidx1.1 = getelementptr inbounds i8* %b, i32 %inc1
  %3 = load i8* %arrayidx1.1, align 1
  %conv26.1 = zext i8 %3 to i32
  %add.1 = add nsw i32 %conv26.1, %conv5.1
  %conv3.1 = trunc i32 %add.1 to i8
  %arrayidx4.1 = getelementptr inbounds i8* %c, i32 %inc1
  store i8 %conv3.1, i8* %arrayidx4.1, align 1
  %inc.12 = or i32 %i.07, 2
  %arrayidx.2 = getelementptr inbounds i8* %a, i32 %inc.12
  %4 = load i8* %arrayidx.2, align 1
  %conv5.2 = zext i8 %4 to i32
  %arrayidx1.2 = getelementptr inbounds i8* %b, i32 %inc.12
  %5 = load i8* %arrayidx1.2, align 1
  %conv26.2 = zext i8 %5 to i32
  %add.2 = add nsw i32 %conv26.2, %conv5.2
  %conv3.2 = trunc i32 %add.2 to i8
  %arrayidx4.2 = getelementptr inbounds i8* %c, i32 %inc.12
  store i8 %conv3.2, i8* %arrayidx4.2, align 1
  %inc.23 = or i32 %i.07, 3
  %arrayidx.3 = getelementptr inbounds i8* %a, i32 %inc.23
  %6 = load i8* %arrayidx.3, align 1
  %conv5.3 = zext i8 %6 to i32
  %arrayidx1.3 = getelementptr inbounds i8* %b, i32 %inc.23
  %7 = load i8* %arrayidx1.3, align 1
  %conv26.3 = zext i8 %7 to i32
  %add.3 = add nsw i32 %conv26.3, %conv5.3
  %conv3.3 = trunc i32 %add.3 to i8
  %arrayidx4.3 = getelementptr inbounds i8* %c, i32 %inc.23
  store i8 %conv3.3, i8* %arrayidx4.3, align 1
  %inc.3 = add nsw i32 %i.07, 4
  %exitcond.3 = icmp eq i32 %inc.3, 400
  br i1 %exitcond.3, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; @testNeon is an important example of the nead for ivchains.
;
; Currently we have three extra add.w's that keep the store address
; live past the next increment because ISEL is unfortunately undoing
; the store chain. ISEL also fails to convert the stores to
; post-increment addressing. However, the loads should use
; post-increment addressing, no add's or add.w's beyond the three
; mentioned. Most importantly, there should be no spills or reloads!
;
; CHECK: testNeon:
; CHECK: %.lr.ph
; CHECK-NOT: lsl.w
; CHECK-NOT: {{ldr|str|adds|add r}}
; CHECK: add.w r
; CHECK-NOT: {{ldr|str|adds|add r}}
; CHECK: add.w r
; CHECK-NOT: {{ldr|str|adds|add r}}
; CHECK: add.w r
; CHECK-NOT: {{ldr|str|adds|add r}}
; CHECK-NOT: add.w r
; CHECK: bne
define hidden void @testNeon(i8* %ref_data, i32 %ref_stride, i32 %limit, <16 x i8>* nocapture %data) nounwind optsize {
  %1 = icmp sgt i32 %limit, 0
  br i1 %1, label %.lr.ph, label %45

.lr.ph:                                           ; preds = %0
  %2 = shl nsw i32 %ref_stride, 1
  %3 = mul nsw i32 %ref_stride, 3
  %4 = shl nsw i32 %ref_stride, 2
  %5 = mul nsw i32 %ref_stride, 5
  %6 = mul nsw i32 %ref_stride, 6
  %7 = mul nsw i32 %ref_stride, 7
  %8 = shl nsw i32 %ref_stride, 3
  %9 = sub i32 0, %8
  %10 = mul i32 %limit, -64
  br label %11

; <label>:11                                      ; preds = %11, %.lr.ph
  %.05 = phi i8* [ %ref_data, %.lr.ph ], [ %42, %11 ]
  %counter.04 = phi i32 [ 0, %.lr.ph ], [ %44, %11 ]
  %result.03 = phi <16 x i8> [ zeroinitializer, %.lr.ph ], [ %41, %11 ]
  %.012 = phi <16 x i8>* [ %data, %.lr.ph ], [ %43, %11 ]
  %12 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %.05, i32 1) nounwind
  %13 = getelementptr inbounds i8* %.05, i32 %ref_stride
  %14 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %13, i32 1) nounwind
  %15 = shufflevector <1 x i64> %12, <1 x i64> %14, <2 x i32> <i32 0, i32 1>
  %16 = bitcast <2 x i64> %15 to <16 x i8>
  %17 = getelementptr inbounds <16 x i8>* %.012, i32 1
  store <16 x i8> %16, <16 x i8>* %.012, align 4
  %18 = getelementptr inbounds i8* %.05, i32 %2
  %19 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %18, i32 1) nounwind
  %20 = getelementptr inbounds i8* %.05, i32 %3
  %21 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %20, i32 1) nounwind
  %22 = shufflevector <1 x i64> %19, <1 x i64> %21, <2 x i32> <i32 0, i32 1>
  %23 = bitcast <2 x i64> %22 to <16 x i8>
  %24 = getelementptr inbounds <16 x i8>* %.012, i32 2
  store <16 x i8> %23, <16 x i8>* %17, align 4
  %25 = getelementptr inbounds i8* %.05, i32 %4
  %26 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %25, i32 1) nounwind
  %27 = getelementptr inbounds i8* %.05, i32 %5
  %28 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %27, i32 1) nounwind
  %29 = shufflevector <1 x i64> %26, <1 x i64> %28, <2 x i32> <i32 0, i32 1>
  %30 = bitcast <2 x i64> %29 to <16 x i8>
  %31 = getelementptr inbounds <16 x i8>* %.012, i32 3
  store <16 x i8> %30, <16 x i8>* %24, align 4
  %32 = getelementptr inbounds i8* %.05, i32 %6
  %33 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %32, i32 1) nounwind
  %34 = getelementptr inbounds i8* %.05, i32 %7
  %35 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %34, i32 1) nounwind
  %36 = shufflevector <1 x i64> %33, <1 x i64> %35, <2 x i32> <i32 0, i32 1>
  %37 = bitcast <2 x i64> %36 to <16 x i8>
  store <16 x i8> %37, <16 x i8>* %31, align 4
  %38 = add <16 x i8> %16, %23
  %39 = add <16 x i8> %38, %30
  %40 = add <16 x i8> %39, %37
  %41 = add <16 x i8> %result.03, %40
  %42 = getelementptr i8* %.05, i32 %9
  %43 = getelementptr inbounds <16 x i8>* %.012, i32 -64
  %44 = add nsw i32 %counter.04, 1
  %exitcond = icmp eq i32 %44, %limit
  br i1 %exitcond, label %._crit_edge, label %11

._crit_edge:                                      ; preds = %11
  %scevgep = getelementptr <16 x i8>* %data, i32 %10
  br label %45

; <label>:45                                      ; preds = %._crit_edge, %0
  %result.0.lcssa = phi <16 x i8> [ %41, %._crit_edge ], [ zeroinitializer, %0 ]
  %.01.lcssa = phi <16 x i8>* [ %scevgep, %._crit_edge ], [ %data, %0 ]
  store <16 x i8> %result.0.lcssa, <16 x i8>* %.01.lcssa, align 4
  ret void
}

declare <1 x i64> @llvm.arm.neon.vld1.v1i64(i8*, i32) nounwind readonly
