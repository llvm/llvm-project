; RUN: opt -S -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check that the interleaved-mem-access analysis identifies the access
; to array 'in' as interleaved, despite the possibly wrapping unsigned
; 'out_ix' index.
;
; In this test the interleave-groups are full (have no gaps), so no wrapping
; checks are necessary. We can call getPtrStride with Assume=false and
; ShouldCheckWrap=false to safely figure out that the stride is 2.

; #include <stdlib.h>
; class Complex {
; private:
;  float real_;
;  float imaginary_;
;
;public:
; Complex() : real_(0), imaginary_(0) { }
; Complex(float real, float imaginary) : real_(real), imaginary_(imaginary) { }
; Complex(const Complex &rhs) : real_(rhs.real()), imaginary_(rhs.imaginary()) { }
;
; inline float real() const { return real_; }
; inline float imaginary() const { return imaginary_; }
;};
;
;void test(Complex * __restrict__ out, Complex * __restrict__ in, size_t out_start, size_t size)
;{
;   for (size_t out_offset = 0; out_offset < size; ++out_offset)
;     {
;       size_t out_ix = out_start + out_offset;
;       Complex t0 = in[out_ix];
;       out[out_ix] = t0;
;     }
;}

; CHECK: vector.body:
; CHECK: %wide.vec = load <8 x i32>, <8 x i32>* {{.*}}, align 4
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK: shufflevector <8 x i32> %wide.vec, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>

%class.Complex = type { float, float }

define void @_Z4testP7ComplexS0_mm(%class.Complex* noalias nocapture %out, %class.Complex* noalias nocapture readonly %in, i64 %out_start, i64 %size) local_unnamed_addr {
entry:
  %cmp9 = icmp eq i64 %size, 0
  br i1 %cmp9, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %out_offset.010 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %add = add i64 %out_offset.010, %out_start
  %arrayidx = getelementptr inbounds %class.Complex, %class.Complex* %in, i64 %add
  %0 = bitcast %class.Complex* %arrayidx to i32*
  %1 = load i32, i32* %0, align 4
  %imaginary_.i.i = getelementptr inbounds %class.Complex, %class.Complex* %in, i64 %add, i32 1
  %2 = bitcast float* %imaginary_.i.i to i32*
  %3 = load i32, i32* %2, align 4
  %arrayidx1 = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %add
  %4 = bitcast %class.Complex* %arrayidx1 to i64*
  %t0.sroa.4.0.insert.ext = zext i32 %3 to i64
  %t0.sroa.4.0.insert.shift = shl nuw i64 %t0.sroa.4.0.insert.ext, 32
  %t0.sroa.0.0.insert.ext = zext i32 %1 to i64
  %t0.sroa.0.0.insert.insert = or i64 %t0.sroa.4.0.insert.shift, %t0.sroa.0.0.insert.ext
  store i64 %t0.sroa.0.0.insert.insert, i64* %4, align 4
  %inc = add nuw i64 %out_offset.010, 1
  %exitcond = icmp eq i64 %inc, %size
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
