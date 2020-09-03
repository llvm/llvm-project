; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

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


;CHECK_LABEL: func:
;CHECK: pmull2

%struct.SS = type { <2 x i64>, <2 x i64> }

; Function Attrs: nofree noinline nounwind
define dso_local void @_Z4funcP2SSjPKhPo(%struct.SS* nocapture readonly %g, i32 %count, i8* nocapture readonly %buf, i128* nocapture %res) local_unnamed_addr #0 {
entry:
  %h2 = getelementptr inbounds %struct.SS, %struct.SS* %g, i64 0, i32 1
  %0 = load <2 x i64>, <2 x i64>* %h2, align 16
  %cmp34 = icmp eq i32 %count, 0
  br i1 %cmp34, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %1 = bitcast %struct.SS* %g to <16 x i8>*
  %2 = load <16 x i8>, <16 x i8>* %1, align 16
  %3 = extractelement <2 x i64> %0, i32 0
  %4 = extractelement <2 x i64> %0, i32 1
  %5 = zext i32 %count to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %buf.addr.036 = phi i8* [ %buf, %for.body.lr.ph ], [ %add.ptr, %for.body ]
  %6 = phi <16 x i8> [ %2, %for.body.lr.ph ], [ %xor.i, %for.body ]
  %7 = bitcast i8* %buf.addr.036 to <16 x i8>*
  %8 = load <16 x i8>, <16 x i8>* %7, align 16
  %vrbit.i = call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %8) #0
  %xor.i = xor <16 x i8> %vrbit.i, %6
  %9 = bitcast <16 x i8> %xor.i to <2 x i64>
  %10 = extractelement <2 x i64> %9, i32 0
  %vmull_p64.i = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %10, i64 %3) #0
  %arrayidx = getelementptr inbounds i128, i128* %res, i64 %indvars.iv
  %11 = bitcast i128* %arrayidx to <16 x i8>*
  store <16 x i8> %vmull_p64.i, <16 x i8>* %11, align 16
  %12 = extractelement <2 x i64> %9, i32 1
  %vmull_p64.i.i = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %12, i64 %4) #0
  %13 = or i64 %indvars.iv, 1
  %arrayidx16 = getelementptr inbounds i128, i128* %res, i64 %13
  %14 = bitcast i128* %arrayidx16 to <16 x i8>*
  store <16 x i8> %vmull_p64.i.i, <16 x i8>* %14, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %add.ptr = getelementptr inbounds i8, i8* %buf.addr.036, i64 16
  %cmp = icmp ult i64 %indvars.iv.next, %5
  br i1 %cmp, label %for.body, label %for.cond.cleanup 
}

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8>) #0

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.pmull64(i64, i64) #0

attributes #0 = { nofree noinline nounwind }
