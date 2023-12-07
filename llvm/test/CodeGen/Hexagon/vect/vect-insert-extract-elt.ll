; RUN: llc -march=hexagon < %s
; Used to fail with an infinite recursion in the insn selection.
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-linux-gnu"

%struct.elt = type { [2 x [4 x %struct.block]] }
%struct.block = type { [2 x i16] }

define void @foo(ptr noalias nocapture %p0, ptr noalias nocapture %p1) nounwind {
entry:
  %arrayidx1 = getelementptr inbounds %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 3
  %arrayidx4 = getelementptr inbounds %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 2
  %arrayidx7 = getelementptr inbounds %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 3
  %0 = load i32, ptr %arrayidx7, align 4
  store i32 %0, ptr %arrayidx4, align 4
  store i32 %0, ptr %arrayidx1, align 4
  %arrayidx10 = getelementptr inbounds %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 1
  %arrayidx16 = getelementptr inbounds %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 2
  %1 = load i32, ptr %arrayidx16, align 4
  store i32 %1, ptr %p1, align 4
  store i32 %1, ptr %arrayidx10, align 4
  %p_arrayidx26 = getelementptr %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %p_arrayidx2632 = getelementptr %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %p_arrayidx2633 = getelementptr %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 1
  %p_arrayidx2634 = getelementptr %struct.elt, ptr %p0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 1
  %p_arrayidx20 = getelementptr %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %p_arrayidx2035 = getelementptr %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %p_arrayidx2036 = getelementptr %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 2, i32 0, i32 1
  %p_arrayidx2037 = getelementptr %struct.elt, ptr %p1, i32 0, i32 0, i32 0, i32 3, i32 0, i32 1
  %2 = lshr i32 %1, 16
  %3 = trunc i32 %2 to i16
  %_p_vec_ = insertelement <4 x i16> undef, i16 %3, i32 0
  %_p_vec_39 = insertelement <4 x i16> %_p_vec_, i16 %3, i32 1
  %4 = lshr i32 %0, 16
  %5 = trunc i32 %4 to i16
  %_p_vec_41 = insertelement <4 x i16> %_p_vec_39, i16 %5, i32 2
  %_p_vec_43 = insertelement <4 x i16> %_p_vec_41, i16 %5, i32 3
  %shlp_vec = shl <4 x i16> %_p_vec_43, <i16 1, i16 1, i16 1, i16 1>
  %6 = extractelement <4 x i16> %shlp_vec, i32 0
  store i16 %6, ptr %p_arrayidx20, align 2
  %7 = extractelement <4 x i16> %shlp_vec, i32 1
  store i16 %7, ptr %p_arrayidx2035, align 2
  %8 = extractelement <4 x i16> %shlp_vec, i32 2
  store i16 %8, ptr %p_arrayidx2036, align 2
  %9 = extractelement <4 x i16> %shlp_vec, i32 3
  store i16 %9, ptr %p_arrayidx2037, align 2
  %_p_scalar_44 = load i16, ptr %p_arrayidx26, align 2
  %_p_vec_45 = insertelement <4 x i16> undef, i16 %_p_scalar_44, i32 0
  %_p_scalar_46 = load i16, ptr %p_arrayidx2632, align 2
  %_p_vec_47 = insertelement <4 x i16> %_p_vec_45, i16 %_p_scalar_46, i32 1
  %_p_scalar_48 = load i16, ptr %p_arrayidx2633, align 2
  %_p_vec_49 = insertelement <4 x i16> %_p_vec_47, i16 %_p_scalar_48, i32 2
  %_p_scalar_50 = load i16, ptr %p_arrayidx2634, align 2
  %_p_vec_51 = insertelement <4 x i16> %_p_vec_49, i16 %_p_scalar_50, i32 3
  %shl28p_vec = shl <4 x i16> %_p_vec_51, <i16 1, i16 1, i16 1, i16 1>
  %10 = extractelement <4 x i16> %shl28p_vec, i32 0
  store i16 %10, ptr %p_arrayidx26, align 2
  %11 = extractelement <4 x i16> %shl28p_vec, i32 1
  store i16 %11, ptr %p_arrayidx2632, align 2
  %12 = extractelement <4 x i16> %shl28p_vec, i32 2
  store i16 %12, ptr %p_arrayidx2633, align 2
  %13 = extractelement <4 x i16> %shl28p_vec, i32 3
  store i16 %13, ptr %p_arrayidx2634, align 2
  ret void
}
