; RUN: llc -mtriple x86_64-apple-macosx -mcpu=corei7-avx -combiner-stress-load-slicing < %s -o - | FileCheck %s --check-prefix=STRESS
; RUN: llc -mtriple x86_64-apple-macosx -mcpu=corei7-avx < %s -o - | FileCheck %s --check-prefix=REGULAR
;
; <rdar://problem/14477220>

%class.Complex = type { float, float }


; Check that independent slices leads to independent loads then the slices leads to
; different register file.
;
; The layout is:
; LSB 0 1 2 3 | 4 5 6 7 MSB
;       Low      High
; The base address points to 0 and is 8-bytes aligned.
; Low slice starts at 0 (base) and is 8-bytes aligned.
; High slice starts at 4 (base + 4-bytes) and is 4-bytes aligned.
;
; STRESS-LABEL: t1:
; Load out[out_start + 8].real, this is base + 8 * 8 + 0.
; STRESS: vmovss 64([[BASE:[^(]+]]), [[OUT_Real:%xmm[0-9]+]]
; Add low slice: out[out_start].real, this is base + 0.
; STRESS-NEXT: vaddss ([[BASE]]), [[OUT_Real]], [[RES_Real:%xmm[0-9]+]]
; Load out[out_start + 8].imm, this is base + 8 * 8 + 4.
; STRESS-NEXT: vmovss 68([[BASE]]), [[OUT_Imm:%xmm[0-9]+]]
; Add high slice: out[out_start].imm, this is base + 4.
; STRESS-NEXT: vaddss 4([[BASE]]), [[OUT_Imm]], [[RES_Imm:%xmm[0-9]+]]
; Swap Imm and Real.
; STRESS-NEXT: vinsertps $16, [[RES_Imm]], [[RES_Real]], [[RES_Vec:%xmm[0-9]+]]
; Put the results back into out[out_start].
; STRESS-NEXT: vmovlps [[RES_Vec]], ([[BASE]])
;
; Same for REGULAR, we eliminate register bank copy with each slices.
; REGULAR-LABEL: t1:
; Load out[out_start + 8].real, this is base + 8 * 8 + 0.
; REGULAR: vmovss 64([[BASE:[^)]+]]), [[OUT_Real:%xmm[0-9]+]]
; Add low slice: out[out_start].real, this is base + 0.
; REGULAR-NEXT: vaddss ([[BASE]]), [[OUT_Real]], [[RES_Real:%xmm[0-9]+]]
; Load out[out_start + 8].imm, this is base + 8 * 8 + 4.
; REGULAR-NEXT: vmovss 68([[BASE]]), [[OUT_Imm:%xmm[0-9]+]]
; Add high slice: out[out_start].imm, this is base + 4.
; REGULAR-NEXT: vaddss 4([[BASE]]), [[OUT_Imm]], [[RES_Imm:%xmm[0-9]+]]
; Swap Imm and Real.
; REGULAR-NEXT: vinsertps $16, [[RES_Imm]], [[RES_Real]], [[RES_Vec:%xmm[0-9]+]]
; Put the results back into out[out_start].
; REGULAR-NEXT: vmovlps [[RES_Vec]], ([[BASE]])
define void @t1(%class.Complex* nocapture %out, i64 %out_start) {
entry:
  %arrayidx = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %out_start
  %tmp = bitcast %class.Complex* %arrayidx to i64*
  %tmp1 = load i64, i64* %tmp, align 8
  %t0.sroa.0.0.extract.trunc = trunc i64 %tmp1 to i32
  %tmp2 = bitcast i32 %t0.sroa.0.0.extract.trunc to float
  %t0.sroa.2.0.extract.shift = lshr i64 %tmp1, 32
  %t0.sroa.2.0.extract.trunc = trunc i64 %t0.sroa.2.0.extract.shift to i32
  %tmp3 = bitcast i32 %t0.sroa.2.0.extract.trunc to float
  %add = add i64 %out_start, 8
  %arrayidx2 = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %add
  %i.i = getelementptr inbounds %class.Complex, %class.Complex* %arrayidx2, i64 0, i32 0
  %tmp4 = load float, float* %i.i, align 4
  %add.i = fadd float %tmp4, %tmp2
  %retval.sroa.0.0.vec.insert.i = insertelement <2 x float> undef, float %add.i, i32 0
  %r.i = getelementptr inbounds %class.Complex, %class.Complex* %arrayidx2, i64 0, i32 1
  %tmp5 = load float, float* %r.i, align 4
  %add5.i = fadd float %tmp5, %tmp3
  %retval.sroa.0.4.vec.insert.i = insertelement <2 x float> %retval.sroa.0.0.vec.insert.i, float %add5.i, i32 1
  %ref.tmp.sroa.0.0.cast = bitcast %class.Complex* %arrayidx to <2 x float>*
  store <2 x float> %retval.sroa.0.4.vec.insert.i, <2 x float>* %ref.tmp.sroa.0.0.cast, align 4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture)

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture)

; Check that we do not read outside of the chunk of bits of the original loads.
;
; The 64-bits should have been split in one 32-bits and one 16-bits slices.
; The 16-bits should be zero extended to match the final type.
;
; The memory layout is:
; LSB 0 1 2 3 | 4 5 | 6 7 MSB
;      Low            High
; The base address points to 0 and is 8-bytes aligned.
; Low slice starts at 0 (base) and is 8-bytes aligned.
; High slice starts at 6 (base + 6-bytes) and is 2-bytes aligned.
;
; STRESS-LABEL: t2:
; STRESS: movzwl 6([[BASE:[^)]+]]), %eax
; STRESS-NEXT: addl ([[BASE]]), %eax
; STRESS-NEXT: ret
;
; For the REGULAR heuristic, this is not profitable to slice things that are not
; next to each other in memory. Here we have a hole with bytes #4-5.
; REGULAR-LABEL: t2:
; REGULAR: shrq $48
define i32 @t2(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i64*
  %chunk64 = load i64, i64* %bitcast, align 8
  %slice32_low = trunc i64 %chunk64 to i32
  %shift48 = lshr i64 %chunk64, 48
  %slice32_high = trunc i64 %shift48 to i32
  %res = add i32 %slice32_high, %slice32_low
  ret i32 %res
}

; Check that we do not optimize overlapping slices.
;
; The 64-bits should NOT have been split in as slices are overlapping.
; First slice uses bytes numbered 0 to 3.
; Second slice uses bytes numbered 6 and 7.
; Third slice uses bytes numbered 4 to 7.
;
; STRESS-LABEL: t3:
; STRESS: shrq $48
; STRESS: shrq $32
;
; REGULAR-LABEL: t3:
; REGULAR: shrq $48
; REGULAR: shrq $32
define i32 @t3(%class.Complex* nocapture %out, i64 %out_start) {
  %arrayidx = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %out_start
  %bitcast = bitcast %class.Complex* %arrayidx to i64*
  %chunk64 = load i64, i64* %bitcast, align 8
  %slice32_low = trunc i64 %chunk64 to i32
  %shift48 = lshr i64 %chunk64, 48
  %slice32_high = trunc i64 %shift48 to i32
  %shift32 = lshr i64 %chunk64, 32
  %slice32_lowhigh = trunc i64 %shift32 to i32
  %tmpres = add i32 %slice32_high, %slice32_low
  %res = add i32 %slice32_lowhigh, %tmpres
  ret i32 %res
}
