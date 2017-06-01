; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: test_scalarize_phi
; CHECK: fmul fast float
define void @test_scalarize_phi(i32 * %n, float * %inout) {
entry:
  %0 = load volatile float, float * %inout, align 4
  %splat.splatinsert = insertelement <4 x float> undef, float %0, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  %splat.splatinsert1 = insertelement <4 x float> undef, float 3.0, i32 0
  br label %for.cond

for.cond:
  %x.0 = phi <4 x float> [ %splat.splat, %entry ], [ %mul, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %1 = load i32, i32 * %n, align 4
  %cmp = icmp ne i32 %i.0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %2 = extractelement <4 x float> %x.0, i32 1
  store volatile float %2, float * %inout, align 4
  %mul = fmul fast <4 x float> %x.0, <float 0x4002A3D700000000, float 0x4002A3D700000000, float 0x4002A3D700000000, float 0x4002A3D700000000>
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret void
}

; CHECK-LABEL: test_extract_element_fastmath
; CHECK: fadd fast float
define float @test_extract_element_fastmath(<4 x float> %x) #0 {
entry:
  %add = fadd fast <4 x float> %x, <float 0x4002A3D700000000, float 0x4002A3D700000000, float 0x4002A3D700000000, float 0x4002A3D700000000>
  %0 = extractelement <4 x float> %add, i32 2
  ret float %0
}

