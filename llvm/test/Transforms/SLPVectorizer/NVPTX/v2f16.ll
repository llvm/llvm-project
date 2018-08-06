; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 | FileCheck %s
; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=nvptx64-nvidia-cuda -mcpu=sm_40 | FileCheck %s -check-prefix=NOVECTOR

; CHECK-LABEL: @fusion
; CHECK: load <2 x half>, <2 x half>*
; CHECK: fmul fast <2 x half>
; CHECK: fadd fast <2 x half>
; CHECK: store <2 x half> %4, <2 x half>

; NOVECTOR-LABEL: @fusion
; NOVECTOR: load half
; NOVECTOR: fmul fast half
; NOVECTOR: fadd fast half
; NOVECTOR: fmul fast half
; NOVECTOR: fadd fast half
; NOVECTOR: store half
define void @fusion(i8* noalias nocapture align 256 dereferenceable(19267584) %arg, i8* noalias nocapture readonly align 256 dereferenceable(19267584) %arg1, i32 %arg2, i32 %arg3) local_unnamed_addr #0 {
  %tmp = shl nuw nsw i32 %arg2, 6
  %tmp4 = or i32 %tmp, %arg3
  %tmp5 = shl nuw nsw i32 %tmp4, 2
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = or i64 %tmp6, 1
  %tmp10 = bitcast i8* %arg1 to half*
  %tmp11 = getelementptr inbounds half, half* %tmp10, i64 %tmp6
  %tmp12 = load half, half* %tmp11, align 8
  %tmp13 = fmul fast half %tmp12, 0xH5380
  %tmp14 = fadd fast half %tmp13, 0xH57F0
  %tmp15 = bitcast i8* %arg to half*
  %tmp16 = getelementptr inbounds half, half* %tmp15, i64 %tmp6
  store half %tmp14, half* %tmp16, align 8
  %tmp17 = getelementptr inbounds half, half* %tmp10, i64 %tmp7
  %tmp18 = load half, half* %tmp17, align 2
  %tmp19 = fmul fast half %tmp18, 0xH5380
  %tmp20 = fadd fast half %tmp19, 0xH57F0
  %tmp21 = getelementptr inbounds half, half* %tmp15, i64 %tmp7
  store half %tmp20, half* %tmp21, align 2
  ret void
}

attributes #0 = { nounwind }
