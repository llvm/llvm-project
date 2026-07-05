; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a57 -verify-machineinstrs < %s | FileCheck %s

; This file check a bug in MachineCopyPropagation pass. The last COPY will be
; incorrectly removed if the machine instructions are as follows:
;   %q5_q6 = COPY %q2_q3
;   %d5 =
;   %d3 =
;   %d3 = COPY %d6
; This is caused by a bug in function SourceNoLongerAvailable(), which fails to
; remove the relationship of D6 and "%q5_q6 = COPY %q2_q3".

@failed = internal unnamed_addr global i1 false

; CHECK-LABEL: foo:
; CHECK: ld2
; CHECK-NOT: // kill: def D{{[0-9]+}} killed D{{[0-9]+}}
define void @foo(<2 x i32> %shuffle251, <8 x i8> %vtbl1.i, ptr %t2, <2 x i32> %vrsubhn_v2.i1364) {
entry:
  %val0 = alloca [2 x i64], align 8
  %val1 = alloca <2 x i64>, align 16
  %vmull = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> <i32 -1, i32 -1>, <2 x i32> %shuffle251)
  %vgetq_lane = extractelement <2 x i64> %vmull, i32 0
  %cmp = icmp eq i64 %vgetq_lane, 1
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i1 true, ptr @failed, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  tail call void @f2()
  %sqdmull = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> <i16 1, i16 0, i16 0, i16 0>, <4 x i16> <i16 2, i16 0, i16 0, i16 0>)
  %sqadd = tail call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> zeroinitializer, <4 x i32> %sqdmull)
  %shuffle = shufflevector <4 x i32> %sqadd, <4 x i32> undef, <2 x i32> zeroinitializer
  %0 = mul <2 x i32> %shuffle, <i32 -1, i32 0>
  %sub = add <2 x i32> %0, <i32 1, i32 0>
  %sext = sext <2 x i32> %sub to <2 x i64>
  %vset_lane603 = shufflevector <2 x i64> %sext, <2 x i64> undef, <1 x i32> zeroinitializer
  call void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64> zeroinitializer, <2 x i64> zeroinitializer, i64 1, ptr %val0)
  call void @llvm.aarch64.neon.st2lane.v1i64.p0(<1 x i64> <i64 4096>, <1 x i64> <i64 -1>, i64 0, ptr %t2)
  %vld2_lane = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0(<1 x i64> <i64 11>, <1 x i64> <i64 11>, i64 0, ptr %t2)
  %vld2_lane.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane, 0
  %vld2_lane.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane, 1
  %vld2_lane1 = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0(<1 x i64> %vld2_lane.0.extract, <1 x i64> %vld2_lane.1.extract, i64 0, ptr %val0)
  %vld2_lane1.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane1, 0
  %vld2_lane1.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane1, 1
  call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> %vld2_lane1.0.extract, <1 x i64> %vld2_lane1.1.extract, ptr %val1)
  %t4 = load <2 x i64>, ptr %val1, align 16
  %vsubhn = sub <2 x i64> <i64 11, i64 0>, %t4
  %vsubhn1 = lshr <2 x i64> %vsubhn, <i64 32, i64 32>
  %vsubhn2 = trunc <2 x i64> %vsubhn1 to <2 x i32>
  %neg = xor <2 x i32> %vsubhn2, <i32 -1, i32 -1>
  %sqadd1 = call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> <i64 -1>, <1 x i64> <i64 1>)
  %sqadd2 = call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> %vset_lane603, <1 x i64> %sqadd1)
  %sqadd3 = call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> <i64 1>, <1 x i64> %sqadd2)
  %shuffle.i = shufflevector <2 x i32> <i32 undef, i32 0>, <2 x i32> %vrsubhn_v2.i1364, <2 x i32> <i32 1, i32 3>
  %cmp.i = icmp uge <2 x i32> %shuffle.i, %neg
  %sext.i = sext <2 x i1> %cmp.i to <2 x i32>
  %vpadal = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> %sext.i)
  %t5 = sub <1 x i64> %vpadal, %sqadd3
  %vget_lane1 = extractelement <1 x i64> %t5, i32 0
  %cmp2 = icmp eq i64 %vget_lane1, 15
  br i1 %cmp2, label %if.end2, label %if.then2

if.then2:                                       ; preds = %if.end
  store i1 true, ptr @failed, align 1
  br label %if.end2

if.end2:                                        ; preds = %if.then682, %if.end
  call void @f2()
  %vext = shufflevector <8 x i8> <i8 undef, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <8 x i8> %vtbl1.i, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  %t6 = bitcast <8 x i8> %vext to <2 x i32>
  call void @f0(<2 x i32> %t6)
  ret void
}

declare void @f0(<2 x i32>)

declare <8 x i8> @f1()

declare void @f2()

declare <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16>, <4 x i16>)

declare void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64>, <2 x i64>, i64, ptr nocapture)

declare void @llvm.aarch64.neon.st2lane.v1i64.p0(<1 x i64>, <1 x i64>, i64, ptr nocapture)

declare { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2lane.v1i64.p0(<1 x i64>, <1 x i64>, i64, ptr)

declare void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64>, <1 x i64>, ptr nocapture)

declare <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64>, <1 x i64>)

declare <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32>)

declare <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32>, <2 x i32>)
