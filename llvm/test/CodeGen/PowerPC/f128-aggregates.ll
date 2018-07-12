; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s \
; RUN:   | FileCheck -check-prefix=CHECK-BE %s

; Testing homogeneous aggregates.

%struct.With9fp128params = type { fp128, fp128, fp128, fp128, fp128, fp128,
                                  fp128, fp128, fp128 }

@a1 = common local_unnamed_addr global [3 x fp128] zeroinitializer, align 16

; Function Attrs: norecurse nounwind readonly
define fp128 @testArray_01(fp128* nocapture readonly %sa) {
; CHECK-LABEL: testArray_01:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 32(r3)
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testArray_01:
; CHECK-BE:       lxv v2, 32(r3)
; CHECK-BE-NEXT:  blr
entry:
  %arrayidx = getelementptr inbounds fp128, fp128* %sa, i64 2
  %0 = load fp128, fp128* %arrayidx, align 16
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readonly
define fp128 @testArray_02() {
; CHECK-LABEL: testArray_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r3, r2, .LC0@toc@ha
; CHECK-NEXT:    ld r3, .LC0@toc@l(r3)
; CHECK-NEXT:    lxv v2, 32(r3)
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testArray_02:
; CHECK-BE:       lxv v2, 32(r3)
; CHECK-BE-NEXT:  blr
entry:
  %0 = load fp128, fp128* getelementptr inbounds ([3 x fp128], [3 x fp128]* @a1,
                                                  i64 0, i64 2), align 16
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testStruct_01(fp128 inreg returned %a.coerce) {
; CHECK-LABEL: testStruct_01:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testStruct_01:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:  blr
entry:
  ret fp128 %a.coerce
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testStruct_02([8 x fp128] %a.coerce) {
; CHECK-LABEL: testStruct_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmr v2, v9
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testStruct_02:
; CHECK-BE:       vmr v2, v9
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.7.extract = extractvalue [8 x fp128] %a.coerce, 7
  ret fp128 %a.coerce.fca.7.extract
}

; Since we can only pass a max of 8 float128 value in VSX registers, ensure we
; store to stack if passing more.
; Function Attrs: norecurse nounwind readonly
define fp128 @testStruct_03(%struct.With9fp128params* byval nocapture readonly
                            align 16 %a) {
; CHECK-LABEL: testStruct_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 128(r1)
; CHECK-NEXT:    std r10, 88(r1)
; CHECK-NEXT:    std r9, 80(r1)
; CHECK-NEXT:    std r8, 72(r1)
; CHECK-NEXT:    std r7, 64(r1)
; CHECK-NEXT:    std r6, 56(r1)
; CHECK-NEXT:    std r5, 48(r1)
; CHECK-NEXT:    std r4, 40(r1)
; CHECK-NEXT:    std r3, 32(r1)
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testStruct_03:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    lxv v2, 144(r1)
; CHECK-BE-NEXT:    std r10, 104(r1)
; CHECK-BE-NEXT:    std r9, 96(r1)
; CHECK-BE-NEXT:    std r8, 88(r1)
; CHECK-BE-NEXT:    std r7, 80(r1)
; CHECK-BE-NEXT:    std r6, 72(r1)
; CHECK-BE-NEXT:    std r5, 64(r1)
; CHECK-BE-NEXT:    std r4, 56(r1)
; CHECK-BE-NEXT:    std r3, 48(r1)
; CHECK-BE-NEXT:    blr
entry:
  %a7 = getelementptr inbounds %struct.With9fp128params,
                               %struct.With9fp128params* %a, i64 0, i32 6
  %0 = load fp128, fp128* %a7, align 16
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testStruct_04([8 x fp128] %a.coerce) {
; CHECK-LABEL: testStruct_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmr v2, v5
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testStruct_04:
; CHECK-BE:       vmr v2, v5
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.3.extract = extractvalue [8 x fp128] %a.coerce, 3
  ret fp128 %a.coerce.fca.3.extract
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testHUnion_01([1 x fp128] %a.coerce) {
; CHECK-LABEL: testHUnion_01:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testHUnion_01:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.0.extract = extractvalue [1 x fp128] %a.coerce, 0
  ret fp128 %a.coerce.fca.0.extract
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testHUnion_02([3 x fp128] %a.coerce) {
; CHECK-LABEL: testHUnion_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testHUnion_02:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.0.extract = extractvalue [3 x fp128] %a.coerce, 0
  ret fp128 %a.coerce.fca.0.extract
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testHUnion_03([3 x fp128] %a.coerce) {
; CHECK-LABEL: testHUnion_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmr v2, v3
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testHUnion_03:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:  vmr v2, v3
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.1.extract = extractvalue [3 x fp128] %a.coerce, 1
  ret fp128 %a.coerce.fca.1.extract
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testHUnion_04([3 x fp128] %a.coerce) {
; CHECK-LABEL: testHUnion_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    vmr v2, v4
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testHUnion_04:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:  vmr v2, v4
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.2.extract = extractvalue [3 x fp128] %a.coerce, 2
  ret fp128 %a.coerce.fca.2.extract
}

; Testing mixed member aggregates.

%struct.MixedC = type { i32, %struct.SA, float, [12 x i8] }
%struct.SA = type { double, fp128, <4 x float> }

; Function Attrs: norecurse nounwind readnone
define fp128 @testMixedAggregate([3 x i128] %a.coerce) {
; CHECK-LABEL: testMixedAggregate:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrdd v2, r8, r7
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testMixedAggregate:
; CHECK-BE:       mtvsrdd v2, r8, r7
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.2.extract = extractvalue [3 x i128] %a.coerce, 2
  %0 = bitcast i128 %a.coerce.fca.2.extract to fp128
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testMixedAggregate_02([4 x i128] %a.coerce) {
; CHECK-LABEL: testMixedAggregate_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrdd v2, r6, r5
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testMixedAggregate_02:
; CHECK-BE:       mtvsrdd v2, r6, r5
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.1.extract = extractvalue [4 x i128] %a.coerce, 1
  %0 = bitcast i128 %a.coerce.fca.1.extract to fp128
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testMixedAggregate_03([4 x i128] %sa.coerce) {
; CHECK-LABEL: testMixedAggregate_03:
; CHECK:       # %bb.0: # %entry
; CHECK-DAG:     mtvsrwa v2, r3
; CHECK-DAG:     mtvsrdd v3, r6, r5
; CHECK:         mtvsrd v4, r10
; CHECK:         xscvsdqp v2, v2
; CHECK-DAG:     xscvsdqp v[[REG:[0-9]+]], v4
; CHECK-DAG:     xsaddqp v2, v3, v2
; CHECK:         xsaddqp v2, v2, v[[REG]]
; CHECK-NEXT:    blr
entry:
  %sa.coerce.fca.0.extract = extractvalue [4 x i128] %sa.coerce, 0
  %sa.sroa.0.0.extract.trunc = trunc i128 %sa.coerce.fca.0.extract to i32
  %sa.coerce.fca.1.extract = extractvalue [4 x i128] %sa.coerce, 1
  %sa.coerce.fca.3.extract = extractvalue [4 x i128] %sa.coerce, 3
  %sa.sroa.6.48.extract.shift = lshr i128 %sa.coerce.fca.3.extract, 64
  %sa.sroa.6.48.extract.trunc = trunc i128 %sa.sroa.6.48.extract.shift to i64
  %conv = sitofp i32 %sa.sroa.0.0.extract.trunc to fp128
  %0 = bitcast i128 %sa.coerce.fca.1.extract to fp128
  %add = fadd fp128 %0, %conv
  %conv2 = sitofp i64 %sa.sroa.6.48.extract.trunc to fp128
  %add3 = fadd fp128 %add, %conv2
  ret fp128 %add3
}


; Function Attrs: norecurse nounwind readonly
define fp128 @testNestedAggregate(%struct.MixedC* byval nocapture readonly align 16 %a) {
; CHECK-LABEL: testNestedAggregate:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    std r8, 72(r1)
; CHECK-NEXT:    std r7, 64(r1)
; CHECK-NEXT:    std r10, 88(r1)
; CHECK-NEXT:    std r9, 80(r1)
; CHECK-NEXT:    lxv v2, 64(r1)
; CHECK-NEXT:    std r6, 56(r1)
; CHECK-NEXT:    std r5, 48(r1)
; CHECK-NEXT:    std r4, 40(r1)
; CHECK-NEXT:    std r3, 32(r1)
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testNestedAggregate:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    std r8, 88(r1)
; CHECK-BE-NEXT:    std r7, 80(r1)
; CHECK-BE-NEXT:    std r10, 104(r1)
; CHECK-BE-NEXT:    std r9, 96(r1)
; CHECK-BE-NEXT:    lxv v2, 80(r1)
; CHECK-BE-NEXT:    std r6, 72(r1)
; CHECK-BE-NEXT:    std r5, 64(r1)
; CHECK-BE-NEXT:    std r4, 56(r1)
; CHECK-BE-NEXT:    std r3, 48(r1)
; CHECK-BE-NEXT:    blr
entry:
  %c = getelementptr inbounds %struct.MixedC, %struct.MixedC* %a, i64 0, i32 1, i32 1
  %0 = load fp128, fp128* %c, align 16
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testUnion_01([1 x i128] %a.coerce) {
; CHECK-LABEL: testUnion_01:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrdd v2, r4, r3
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testUnion_01:
; CHECK-BE:       mtvsrdd v2, r4, r3
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.0.extract = extractvalue [1 x i128] %a.coerce, 0
  %0 = bitcast i128 %a.coerce.fca.0.extract to fp128
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testUnion_02([1 x i128] %a.coerce) {
; CHECK-LABEL: testUnion_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrdd v2, r4, r3
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testUnion_02:
; CHECK-BE:       mtvsrdd v2, r4, r3
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.0.extract = extractvalue [1 x i128] %a.coerce, 0
  %0 = bitcast i128 %a.coerce.fca.0.extract to fp128
  ret fp128 %0
}

; Function Attrs: norecurse nounwind readnone
define fp128 @testUnion_03([4 x i128] %a.coerce) {
; CHECK-LABEL: testUnion_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrdd v2, r8, r7
; CHECK-NEXT:    blr

; CHECK-BE-LABEL: testUnion_03:
; CHECK-BE:       mtvsrdd v2, r8, r7
; CHECK-BE-NEXT:  blr
entry:
  %a.coerce.fca.2.extract = extractvalue [4 x i128] %a.coerce, 2
  %0 = bitcast i128 %a.coerce.fca.2.extract to fp128
  ret fp128 %0
}

; Function Attrs: nounwind
define fp128 @sum_float128(i32 signext %count, ...) {
; CHECK-LABEL: sum_float128:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r11, r2, .LCPI17_0@toc@ha
; CHECK-NEXT:    cmpwi cr0, r3, 1
; CHECK-NEXT:    std r10, 88(r1)
; CHECK-NEXT:    std r9, 80(r1)
; CHECK-NEXT:    std r8, 72(r1)
; CHECK-NEXT:    std r7, 64(r1)
; CHECK-NEXT:    std r6, 56(r1)
; CHECK-NEXT:    std r5, 48(r1)
; CHECK-NEXT:    std r4, 40(r1)
; CHECK-NEXT:    addi r11, r11, .LCPI17_0@toc@l
; CHECK-NEXT:    lxvx v2, 0, r11
; CHECK-NEXT:    bltlr cr0
; CHECK-NEXT:  # %bb.1: # %if.end
; CHECK-NEXT:    addi r3, r1, 40
; CHECK-NEXT:    lxvx v3, 0, r3
; CHECK-NEXT:    xsaddqp v2, v3, v2
; CHECK-NEXT:    lxv v3, 16(r3)
; CHECK-NEXT:    addi r3, r1, 72
; CHECK-NEXT:    std r3, -8(r1)
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    blr
entry:
  %ap = alloca i8*, align 8
  %0 = bitcast i8** %ap to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #2
  %cmp = icmp slt i32 %count, 1
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  call void @llvm.va_start(i8* nonnull %0)
  %argp.cur = load i8*, i8** %ap, align 8
  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i64 16
  %1 = bitcast i8* %argp.cur to fp128*
  %2 = load fp128, fp128* %1, align 8
  %add = fadd fp128 %2, 0xL00000000000000000000000000000000
  %argp.next3 = getelementptr inbounds i8, i8* %argp.cur, i64 32
  store i8* %argp.next3, i8** %ap, align 8
  %3 = bitcast i8* %argp.next to fp128*
  %4 = load fp128, fp128* %3, align 8
  %add4 = fadd fp128 %add, %4
  call void @llvm.va_end(i8* nonnull %0)
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
  %retval.0 = phi fp128 [ %add4, %if.end ], [ 0xL00000000000000000000000000000000, %entry ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #2
  ret fp128 %retval.0
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.va_start(i8*) #2
declare void @llvm.va_end(i8*) #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1
