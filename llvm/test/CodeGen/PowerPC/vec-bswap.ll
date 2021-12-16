; RUN: llc < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names | FileCheck %s

; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr9 \
; RUN:   -verify-machineinstrs -vec-extabi | \
; RUN:   FileCheck %s --check-prefixes=AIX,AIX64
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 \
; RUN:   -verify-machineinstrs  -vec-extabi | \
; RUN:   FileCheck %s --check-prefixes=AIX,AIX32

define dso_local void @test(i32* %Arr, i32 signext %Len) {
; CHECK-LABEL: test:
; CHECK:         lxv [[REG:vs[0-9]+]], 0(r{{[0-9]+}})
; CHECK-NOT:     [[REG]]
; CHECK:         xxbrw vs{{[0-9]+}}, [[REG]]

; AIX-LABEL:     test:
; AIX64:         lxv [[REG64:[0-9]+]], {{[0-9]+}}({{[0-9]+}})
; AIX32:         lxv [[REG32:[0-9]+]], {{[0-9]+}}({{[0-9]+}})
; AIX64-NOT:     [[REG64]]
; AIX64:         xxbrw {{[0-9]+}}, [[REG64]]
; AIX32:         xxbrw {{[0-9]+}}, [[REG32]]
entry:
  %cmp1 = icmp slt i32 0, %Len
  br i1 %cmp1, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %min.iters.check = icmp ult i32 %Len, 4
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.mod.vf = urem i32 %Len, 4
  %n.vec = sub i32 %Len, %n.mod.vf
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = add i32 %index, 0
  %1 = sext i32 %0 to i64
  %2 = getelementptr inbounds i32, i32* %Arr, i64 %1
  %3 = getelementptr inbounds i32, i32* %2, i32 0
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %4, align 4
  %5 = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %wide.load)
  %6 = sext i32 %0 to i64
  %7 = getelementptr inbounds i32, i32* %Arr, i64 %6
  %8 = getelementptr inbounds i32, i32* %7, i32 0
  %9 = bitcast i32* %8 to <4 x i32>*
  store <4 x i32> %5, <4 x i32>* %9, align 4
  %index.next = add i32 %index, 4
  %10 = icmp eq i32 %index.next, %n.vec
  br i1 %10, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %Len, %n.vec
  br i1 %cmp.n, label %for.cond.for.cond.cleanup_crit_edge, label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %for.body.lr.ph
  %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %for.body.lr.ph ]
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:              ; preds = %middle.block, %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.for.cond.cleanup_crit_edge, %entry
  br label %for.end

for.body:                                         ; preds = %for.inc, %scalar.ph
  %i.02 = phi i32 [ %bc.resume.val, %scalar.ph ], [ %inc, %for.inc ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %Arr, i64 %idxprom
  %11 = load i32, i32* %arrayidx, align 4
  %12 = call i32 @llvm.bswap.i32(i32 %11)
  %idxprom1 = sext i32 %i.02 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %Arr, i64 %idxprom1
  store i32 %12, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %Len
  br i1 %cmp, label %for.body, label %for.cond.for.cond.cleanup_crit_edge

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

define dso_local <8 x i16> @test_halfword(<8 x i16> %a) local_unnamed_addr {
; CHECK-LABEL: test_halfword:
; CHECK:       xxbrh vs34, vs34
; CHECK-NEXT:  blr

; AIX-LABEL:   test_halfword:
; AIX:         xxbrh 34, 34
; AIX-NEXT:    blr
entry:
  %0 = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %a)
  ret <8 x i16> %0
}

define dso_local <2 x i64> @test_doubleword(<2 x i64> %a) local_unnamed_addr {
; CHECK-LABEL: test_doubleword:
; CHECK:       xxbrd vs34, vs34
; CHECK-NEXT:  blr

; AIX-LABEL:   test_doubleword:
; AIX:         xxbrd 34, 34
; AIX-NEXT:    blr
entry:
  %0 = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %a)
  ret <2 x i64> %0
}

define dso_local <1 x i128> @test_quadword(<1 x i128> %a) local_unnamed_addr {
; CHECK-LABEL: test_quadword:
; CHECK:       xxbrq vs34, vs34
; CHECK-NEXT:  blr

; AIX-LABEL:   test_quadword:
; AIX:         xxbrq 34, 34
; AIX-NEXT:    blr
entry:
  %0 = call <1 x i128> @llvm.bswap.v1i128(<1 x i128> %a)
  ret <1 x i128> %0
}

; Function Attrs: nounwind readnone speculatable willreturn
declare <1 x i128> @llvm.bswap.v1i128(<1 x i128>)

; Function Attrs: nounwind readnone speculatable willreturn
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

; Function Attrs: nounwind readnone speculatable willreturn
declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.bswap.i32(i32)

; Function Attrs: nounwind readnone speculatable willreturn
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
