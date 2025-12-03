; RUN: llc -march=hexagon -mattr=+hvx-length128b,+hvxv68 < %s

define void @_Z10range_flatIDF16bEvjT_S0_PS0_(i32 noundef %d, half noundef %start, half noundef %increm, ptr noundef %out) local_unnamed_addr {
entry:
  %d.ripple.bcast.splatinsert = insertelement <64 x i32> poison, i32 %d, i64 0
  %d.ripple.bcast.splat = shufflevector <64 x i32> %d.ripple.bcast.splatinsert, <64 x i32> poison, <64 x i32> zeroinitializer
  %0 = fpext half %increm to float
  %.ripple.bcast.splatinsert = insertelement <64 x float> poison, float %0, i64 0
  %.ripple.bcast.splat = shufflevector <64 x float> %.ripple.bcast.splatinsert, <64 x float> poison, <64 x i32> zeroinitializer
  %mul.ripple.vectorized = fmul <64 x float> %.ripple.bcast.splat, <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00, float 9.000000e+00, float 1.000000e+01, float 1.100000e+01, float 1.200000e+01, float 1.300000e+01, float 1.400000e+01, float 1.500000e+01, float 1.600000e+01, float 1.700000e+01, float 1.800000e+01, float 1.900000e+01, float 2.000000e+01, float 2.100000e+01, float 2.200000e+01, float 2.300000e+01, float 2.400000e+01, float 2.500000e+01, float 2.600000e+01, float 2.700000e+01, float 2.800000e+01, float 2.900000e+01, float 3.000000e+01, float 3.100000e+01, float 3.200000e+01, float 3.300000e+01, float 3.400000e+01, float 3.500000e+01, float 3.600000e+01, float 3.700000e+01, float 3.800000e+01, float 3.900000e+01, float 4.000000e+01, float 4.100000e+01, float 4.200000e+01, float 4.300000e+01, float 4.400000e+01, float 4.500000e+01, float 4.600000e+01, float 4.700000e+01, float 4.800000e+01, float 4.900000e+01, float 5.000000e+01, float 5.100000e+01, float 5.200000e+01, float 5.300000e+01, float 5.400000e+01, float 5.500000e+01, float 5.600000e+01, float 5.700000e+01, float 5.800000e+01, float 5.900000e+01, float 6.000000e+01, float 6.100000e+01, float 6.200000e+01, float 6.300000e+01, float 6.400000e+01>
  %arrayidx = getelementptr i8, ptr %out, i32 0
  %1 = fptrunc <64 x float> %mul.ripple.vectorized to <64 x half>
  store <64 x half> %1, ptr %arrayidx, align 2
  ret void
}
