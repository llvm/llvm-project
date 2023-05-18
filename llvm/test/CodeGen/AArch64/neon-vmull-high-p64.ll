; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon,+aes | FileCheck %s

; This test checks that pmull2 instruction is used for vmull_high_p64 intrinsic.
; There are two extraction operations located in different basic blocks:
;
; %4 = extractelement <2 x i64> %0, i32 1
; %12 = extractelement <2 x i64> %9, i32 1
;
; They are used by:
;
; @llvm.aarch64.neon.pmull64(i64 %12, i64 %4) #2
;
; We test that pattern replacing llvm.aarch64.neon.pmull64 with pmull2
; would be applied.

; IR for that test was generated from the following .cpp file:
;
; #include <arm_neon.h>
;
; struct SS {
;     uint64x2_t x, h;
; };
;
; void func (SS *g, unsigned int count, const unsigned char *buf, poly128_t* res )
; {
;   const uint64x2_t x = g->x;
;   const uint64x2_t h = g->h;
;   uint64x2_t ci = g->x;
;
;   for (int i = 0; i < count; i+=2, buf += 16) {
;     ci = vreinterpretq_u64_u8(veorq_u8(vreinterpretq_u8_u64(ci),
;                                            vrbitq_u8(vld1q_u8(buf))));
;     res[i] = vmull_p64((poly64_t)vget_low_p64(vreinterpretq_p64_u64(ci)),
;                        (poly64_t)vget_low_p64(vreinterpretq_p64_u64(h)));
;     res[i+1] = vmull_high_p64(vreinterpretq_p64_u64(ci),
;                               vreinterpretq_p64_u64(h));
;   }
; }


;CHECK-LABEL: func:
;CHECK: pmull2

%struct.SS = type { <2 x i64>, <2 x i64> }

; Function Attrs: nofree noinline nounwind
define dso_local void @func(ptr nocapture readonly %g, i32 %count, ptr nocapture readonly %buf, ptr nocapture %res) local_unnamed_addr #0 {
entry:
  %h2 = getelementptr inbounds %struct.SS, ptr %g, i64 0, i32 1
  %0 = load <2 x i64>, ptr %h2, align 16
  %cmp34 = icmp eq i32 %count, 0
  br i1 %cmp34, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %1 = load <16 x i8>, ptr %g, align 16
  %2 = extractelement <2 x i64> %0, i32 0
  %3 = extractelement <2 x i64> %0, i32 1
  %4 = zext i32 %count to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %buf.addr.036 = phi ptr [ %buf, %for.body.lr.ph ], [ %add.ptr, %for.body ]
  %5 = phi <16 x i8> [ %1, %for.body.lr.ph ], [ %xor.i, %for.body ]
  %6 = load <16 x i8>, ptr %buf.addr.036, align 16
  %vrbit.i = call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %6) #0
  %xor.i = xor <16 x i8> %vrbit.i, %5
  %7 = bitcast <16 x i8> %xor.i to <2 x i64>
  %8 = extractelement <2 x i64> %7, i32 0
  %vmull_p64.i = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %8, i64 %2) #0
  %arrayidx = getelementptr inbounds i128, ptr %res, i64 %indvars.iv
  store <16 x i8> %vmull_p64.i, ptr %arrayidx, align 16
  %9 = extractelement <2 x i64> %7, i32 1
  %vmull_p64.i.i = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %9, i64 %3) #0
  %10 = or i64 %indvars.iv, 1
  %arrayidx16 = getelementptr inbounds i128, ptr %res, i64 %10
  store <16 x i8> %vmull_p64.i.i, ptr %arrayidx16, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %add.ptr = getelementptr inbounds i8, ptr %buf.addr.036, i64 16
  %cmp = icmp ult i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup 
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8>) #0

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.pmull64(i64, i64) #0

attributes #0 = { nofree noinline nounwind }
