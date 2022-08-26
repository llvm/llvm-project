; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check that the compile-time-unknown depenendece-distance is resolved 
; statically. Due to the non-unit stride of the accesses in this testcase
; we are currently not able to create runtime dependence checks, and therefore
; if we don't resolve the dependence statically we cannot vectorize the loop.
;
; Specifically in this example, during dependence analysis we get 6 unknown 
; dependence distances between the 8 real/imaginary accesses below: 
;    dist = 8*D, 4+8*D, -4+8*D, -8*D, 4-8*D, -4-8*D.
; At compile time we can prove for all of the above that |dist|>loopBound*step
; (where the step is 8bytes, and the loopBound is D-1), and thereby conclude 
; that there are no dependencies (without runtime tests):
; |8*D|>8*D-8, |4+8*D|>8*D-8, |-4+8*D|>8*D-8, etc.

; #include <stdlib.h>
; class Complex {
; private:
;   float real_;
;   float imaginary_;
;
; public:
;   Complex() : real_(0), imaginary_(0) { }
;   Complex(float real, float imaginary) : real_(real), imaginary_(imaginary) { }
;   Complex(const Complex &rhs) : real_(rhs.real()), imaginary_(rhs.imaginary()) { }
; 
;   inline float real() const { return real_; }
;   inline float imaginary() const { return imaginary_; }
; 
;   Complex operator+(const Complex& rhs) const
;   {
;    return Complex(real_ + rhs.real_, imaginary_ + rhs.imaginary_);
;   }
;
;   Complex operator-(const Complex& rhs) const
;  {
;     return Complex(real_ - rhs.real_, imaginary_ - rhs.imaginary_);
;   }
; };
;
; void Test(Complex *out, size_t size)
; {
;     size_t D = size / 2;
;     for (size_t offset = 0; offset < D; ++offset)
;     {
;         Complex t0 = out[offset];
;         Complex t1 = out[offset + D];
;         out[offset] = t1 + t0;
;         out[offset + D] = t0 - t1;
;     }
; }

; CHECK-LABEL: Test
; CHECK: Memory dependences are safe


%class.Complex = type { float, float }

define void @Test(%class.Complex* nocapture %out, i64 %size) local_unnamed_addr {
entry:
  %div = lshr i64 %size, 1
  %cmp47 = icmp eq i64 %div, 0
  br i1 %cmp47, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %offset.048 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %0 = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %offset.048, i32 0
  %1 = load float, float* %0, align 4
  %imaginary_.i.i = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %offset.048, i32 1
  %2 = load float, float* %imaginary_.i.i, align 4
  %add = add nuw i64 %offset.048, %div
  %3 = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %add, i32 0
  %4 = load float, float* %3, align 4
  %imaginary_.i.i28 = getelementptr inbounds %class.Complex, %class.Complex* %out, i64 %add, i32 1
  %5 = load float, float* %imaginary_.i.i28, align 4
  %add.i = fadd fast float %4, %1
  %add4.i = fadd fast float %5, %2
  store float %add.i, float* %0, align 4
  store float %add4.i, float* %imaginary_.i.i, align 4
  %sub.i = fsub fast float %1, %4
  %sub4.i = fsub fast float %2, %5
  store float %sub.i, float* %3, align 4
  store float %sub4.i, float* %imaginary_.i.i28, align 4
  %inc = add nuw nsw i64 %offset.048, 1
  %exitcond = icmp eq i64 %inc, %div
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
