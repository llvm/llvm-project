; RUN: llc < %s -mtriple=arm64-eabi -aarch64-redzone | FileCheck %s

define void @store64(i64** nocapture %out, i64 %index, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: store64:
; CHECK: str x{{[0-9+]}}, [x{{[0-9+]}}], #8
; CHECK: ret
  %tmp = load i64*, i64** %out, align 8
  %incdec.ptr = getelementptr inbounds i64, i64* %tmp, i64 1
  store i64 %spacing, i64* %tmp, align 4
  store i64* %incdec.ptr, i64** %out, align 8
  ret void
}

define void @store32(i32** nocapture %out, i32 %index, i32 %spacing) nounwind noinline ssp {
; CHECK-LABEL: store32:
; CHECK: str w{{[0-9+]}}, [x{{[0-9+]}}], #4
; CHECK: ret
  %tmp = load i32*, i32** %out, align 8
  %incdec.ptr = getelementptr inbounds i32, i32* %tmp, i64 1
  store i32 %spacing, i32* %tmp, align 4
  store i32* %incdec.ptr, i32** %out, align 8
  ret void
}

define void @store16(i16** nocapture %out, i16 %index, i16 %spacing) nounwind noinline ssp {
; CHECK-LABEL: store16:
; CHECK: strh w{{[0-9+]}}, [x{{[0-9+]}}], #2
; CHECK: ret
  %tmp = load i16*, i16** %out, align 8
  %incdec.ptr = getelementptr inbounds i16, i16* %tmp, i64 1
  store i16 %spacing, i16* %tmp, align 4
  store i16* %incdec.ptr, i16** %out, align 8
  ret void
}

define void @store8(i8** nocapture %out, i8 %index, i8 %spacing) nounwind noinline ssp {
; CHECK-LABEL: store8:
; CHECK: strb w{{[0-9+]}}, [x{{[0-9+]}}], #1
; CHECK: ret
  %tmp = load i8*, i8** %out, align 8
  %incdec.ptr = getelementptr inbounds i8, i8* %tmp, i64 1
  store i8 %spacing, i8* %tmp, align 4
  store i8* %incdec.ptr, i8** %out, align 8
  ret void
}

define void @truncst64to32(i32** nocapture %out, i32 %index, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: truncst64to32:
; CHECK: str w{{[0-9+]}}, [x{{[0-9+]}}], #4
; CHECK: ret
  %tmp = load i32*, i32** %out, align 8
  %incdec.ptr = getelementptr inbounds i32, i32* %tmp, i64 1
  %trunc = trunc i64 %spacing to i32
  store i32 %trunc, i32* %tmp, align 4
  store i32* %incdec.ptr, i32** %out, align 8
  ret void
}

define void @truncst64to16(i16** nocapture %out, i16 %index, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: truncst64to16:
; CHECK: strh w{{[0-9+]}}, [x{{[0-9+]}}], #2
; CHECK: ret
  %tmp = load i16*, i16** %out, align 8
  %incdec.ptr = getelementptr inbounds i16, i16* %tmp, i64 1
  %trunc = trunc i64 %spacing to i16
  store i16 %trunc, i16* %tmp, align 4
  store i16* %incdec.ptr, i16** %out, align 8
  ret void
}

define void @truncst64to8(i8** nocapture %out, i8 %index, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: truncst64to8:
; CHECK: strb w{{[0-9+]}}, [x{{[0-9+]}}], #1
; CHECK: ret
  %tmp = load i8*, i8** %out, align 8
  %incdec.ptr = getelementptr inbounds i8, i8* %tmp, i64 1
  %trunc = trunc i64 %spacing to i8
  store i8 %trunc, i8* %tmp, align 4
  store i8* %incdec.ptr, i8** %out, align 8
  ret void
}


define void @storef16(half** %out, half %index, half %spacing) nounwind {
; CHECK-LABEL: storef16:
; CHECK: str h{{[0-9+]}}, [x{{[0-9+]}}], #2
; CHECK: ret
  %tmp = load half*, half** %out, align 2
  %incdec.ptr = getelementptr inbounds half, half* %tmp, i64 1
  store half %spacing, half* %tmp, align 2
  store half* %incdec.ptr, half** %out, align 2
  ret void
}

define void @storef32(float** nocapture %out, float %index, float %spacing) nounwind noinline ssp {
; CHECK-LABEL: storef32:
; CHECK: str s{{[0-9+]}}, [x{{[0-9+]}}], #4
; CHECK: ret
  %tmp = load float*, float** %out, align 8
  %incdec.ptr = getelementptr inbounds float, float* %tmp, i64 1
  store float %spacing, float* %tmp, align 4
  store float* %incdec.ptr, float** %out, align 8
  ret void
}

define void @storef64(double** nocapture %out, double %index, double %spacing) nounwind noinline ssp {
; CHECK-LABEL: storef64:
; CHECK: str d{{[0-9+]}}, [x{{[0-9+]}}], #8
; CHECK: ret
  %tmp = load double*, double** %out, align 8
  %incdec.ptr = getelementptr inbounds double, double* %tmp, i64 1
  store double %spacing, double* %tmp, align 4
  store double* %incdec.ptr, double** %out, align 8
  ret void
}

define double * @pref64(double** nocapture %out, double %spacing) nounwind noinline ssp {
; CHECK-LABEL: pref64:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: str     d0, [x0, #32]!
; CHECK-NEXT: ret
  %tmp = load double*, double** %out, align 8
  %ptr = getelementptr inbounds double, double* %tmp, i64 4
  store double %spacing, double* %ptr, align 4
  ret double *%ptr
}

define float * @pref32(float** nocapture %out, float %spacing) nounwind noinline ssp {
; CHECK-LABEL: pref32:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: str     s0, [x0, #12]!
; CHECK-NEXT: ret
  %tmp = load float*, float** %out, align 8
  %ptr = getelementptr inbounds float, float* %tmp, i64 3
  store float %spacing, float* %ptr, align 4
  ret float *%ptr
}

define half* @pref16(half** %out, half %spacing) nounwind {
; CHECK-LABEL: pref16:
; CHECK: ldr x0, [x0]
; CHECK-NEXT: str h0, [x0, #6]!
; CHECK-NEXT: ret
  %tmp = load half*, half** %out, align 2
  %ptr = getelementptr inbounds half, half* %tmp, i64 3
  store half %spacing, half* %ptr, align 2
  ret half *%ptr
}

define i64 * @pre64(i64** nocapture %out, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pre64:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: str     x1, [x0, #16]!
; CHECK-NEXT: ret
  %tmp = load i64*, i64** %out, align 8
  %ptr = getelementptr inbounds i64, i64* %tmp, i64 2
  store i64 %spacing, i64* %ptr, align 4
  ret i64 *%ptr
}

define i32 * @pre32(i32** nocapture %out, i32 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pre32:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: str     w1, [x0, #8]!
; CHECK-NEXT: ret
  %tmp = load i32*, i32** %out, align 8
  %ptr = getelementptr inbounds i32, i32* %tmp, i64 2
  store i32 %spacing, i32* %ptr, align 4
  ret i32 *%ptr
}

define i16 * @pre16(i16** nocapture %out, i16 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pre16:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: strh    w1, [x0, #4]!
; CHECK-NEXT: ret
  %tmp = load i16*, i16** %out, align 8
  %ptr = getelementptr inbounds i16, i16* %tmp, i64 2
  store i16 %spacing, i16* %ptr, align 4
  ret i16 *%ptr
}

define i8 * @pre8(i8** nocapture %out, i8 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pre8:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: strb    w1, [x0, #2]!
; CHECK-NEXT: ret
  %tmp = load i8*, i8** %out, align 8
  %ptr = getelementptr inbounds i8, i8* %tmp, i64 2
  store i8 %spacing, i8* %ptr, align 4
  ret i8 *%ptr
}

define i32 * @pretrunc64to32(i32** nocapture %out, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pretrunc64to32:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: str     w1, [x0, #8]!
; CHECK-NEXT: ret
  %tmp = load i32*, i32** %out, align 8
  %ptr = getelementptr inbounds i32, i32* %tmp, i64 2
  %trunc = trunc i64 %spacing to i32
  store i32 %trunc, i32* %ptr, align 4
  ret i32 *%ptr
}

define i16 * @pretrunc64to16(i16** nocapture %out, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pretrunc64to16:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: strh    w1, [x0, #4]!
; CHECK-NEXT: ret
  %tmp = load i16*, i16** %out, align 8
  %ptr = getelementptr inbounds i16, i16* %tmp, i64 2
  %trunc = trunc i64 %spacing to i16
  store i16 %trunc, i16* %ptr, align 4
  ret i16 *%ptr
}

define i8 * @pretrunc64to8(i8** nocapture %out, i64 %spacing) nounwind noinline ssp {
; CHECK-LABEL: pretrunc64to8:
; CHECK: ldr     x0, [x0]
; CHECK-NEXT: strb    w1, [x0, #2]!
; CHECK-NEXT: ret
  %tmp = load i8*, i8** %out, align 8
  %ptr = getelementptr inbounds i8, i8* %tmp, i64 2
  %trunc = trunc i64 %spacing to i8
  store i8 %trunc, i8* %ptr, align 4
  ret i8 *%ptr
}

;-----
; Pre-indexed loads
;-----
define double* @preidxf64(double* %src, double* %out) {
; CHECK-LABEL: preidxf64:
; CHECK: ldr     d0, [x0, #8]!
; CHECK: str     d0, [x1]
; CHECK: ret
  %ptr = getelementptr inbounds double, double* %src, i64 1
  %tmp = load double, double* %ptr, align 4
  store double %tmp, double* %out, align 4
  ret double* %ptr
}

define float* @preidxf32(float* %src, float* %out) {
; CHECK-LABEL: preidxf32:
; CHECK: ldr     s0, [x0, #4]!
; CHECK: str     s0, [x1]
; CHECK: ret
  %ptr = getelementptr inbounds float, float* %src, i64 1
  %tmp = load float, float* %ptr, align 4
  store float %tmp, float* %out, align 4
  ret float* %ptr
}

define half* @preidxf16(half* %src, half* %out) {
; CHECK-LABEL: preidxf16:
; CHECK: ldr     h0, [x0, #2]!
; CHECK: str     h0, [x1]
; CHECK: ret
  %ptr = getelementptr inbounds half, half* %src, i64 1
  %tmp = load half, half* %ptr, align 2
  store half %tmp, half* %out, align 2
  ret half* %ptr
}

define i64* @preidx64(i64* %src, i64* %out) {
; CHECK-LABEL: preidx64:
; CHECK: ldr     x[[REG:[0-9]+]], [x0, #8]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i64, i64* %src, i64 1
  %tmp = load i64, i64* %ptr, align 4
  store i64 %tmp, i64* %out, align 4
  ret i64* %ptr
}

define i32* @preidx32(i32* %src, i32* %out) {
; CHECK: ldr     w[[REG:[0-9]+]], [x0, #4]!
; CHECK: str     w[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i32, i32* %src, i64 1
  %tmp = load i32, i32* %ptr, align 4
  store i32 %tmp, i32* %out, align 4
  ret i32* %ptr
}

define i16* @preidx16zext32(i16* %src, i32* %out) {
; CHECK: ldrh    w[[REG:[0-9]+]], [x0, #2]!
; CHECK: str     w[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i16, i16* %src, i64 1
  %tmp = load i16, i16* %ptr, align 4
  %ext = zext i16 %tmp to i32
  store i32 %ext, i32* %out, align 4
  ret i16* %ptr
}

define i16* @preidx16zext64(i16* %src, i64* %out) {
; CHECK: ldrh    w[[REG:[0-9]+]], [x0, #2]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i16, i16* %src, i64 1
  %tmp = load i16, i16* %ptr, align 4
  %ext = zext i16 %tmp to i64
  store i64 %ext, i64* %out, align 4
  ret i16* %ptr
}

define i8* @preidx8zext32(i8* %src, i32* %out) {
; CHECK: ldrb    w[[REG:[0-9]+]], [x0, #1]!
; CHECK: str     w[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i8, i8* %src, i64 1
  %tmp = load i8, i8* %ptr, align 4
  %ext = zext i8 %tmp to i32
  store i32 %ext, i32* %out, align 4
  ret i8* %ptr
}

define i8* @preidx8zext64(i8* %src, i64* %out) {
; CHECK: ldrb    w[[REG:[0-9]+]], [x0, #1]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i8, i8* %src, i64 1
  %tmp = load i8, i8* %ptr, align 4
  %ext = zext i8 %tmp to i64
  store i64 %ext, i64* %out, align 4
  ret i8* %ptr
}

define i32* @preidx32sext64(i32* %src, i64* %out) {
; CHECK: ldrsw   x[[REG:[0-9]+]], [x0, #4]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i32, i32* %src, i64 1
  %tmp = load i32, i32* %ptr, align 4
  %ext = sext i32 %tmp to i64
  store i64 %ext, i64* %out, align 8
  ret i32* %ptr
}

define i16* @preidx16sext32(i16* %src, i32* %out) {
; CHECK: ldrsh   w[[REG:[0-9]+]], [x0, #2]!
; CHECK: str     w[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i16, i16* %src, i64 1
  %tmp = load i16, i16* %ptr, align 4
  %ext = sext i16 %tmp to i32
  store i32 %ext, i32* %out, align 4
  ret i16* %ptr
}

define i16* @preidx16sext64(i16* %src, i64* %out) {
; CHECK: ldrsh   x[[REG:[0-9]+]], [x0, #2]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i16, i16* %src, i64 1
  %tmp = load i16, i16* %ptr, align 4
  %ext = sext i16 %tmp to i64
  store i64 %ext, i64* %out, align 4
  ret i16* %ptr
}

define i8* @preidx8sext32(i8* %src, i32* %out) {
; CHECK: ldrsb   w[[REG:[0-9]+]], [x0, #1]!
; CHECK: str     w[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i8, i8* %src, i64 1
  %tmp = load i8, i8* %ptr, align 4
  %ext = sext i8 %tmp to i32
  store i32 %ext, i32* %out, align 4
  ret i8* %ptr
}

define i8* @preidx8sext64(i8* %src, i64* %out) {
; CHECK: ldrsb   x[[REG:[0-9]+]], [x0, #1]!
; CHECK: str     x[[REG]], [x1]
; CHECK: ret
  %ptr = getelementptr inbounds i8, i8* %src, i64 1
  %tmp = load i8, i8* %ptr, align 4
  %ext = sext i8 %tmp to i64
  store i64 %ext, i64* %out, align 4
  ret i8* %ptr
}

; This test checks if illegal post-index is generated

define i64* @postidx_clobber(i64* %addr) nounwind noinline ssp {
; CHECK-LABEL: postidx_clobber:
; CHECK-NOT: str     x0, [x0], #8
; ret
 %paddr = bitcast i64* %addr to i64**
 store i64* %addr, i64** %paddr
 %newaddr = getelementptr i64, i64* %addr, i32 1
 ret i64* %newaddr
}
