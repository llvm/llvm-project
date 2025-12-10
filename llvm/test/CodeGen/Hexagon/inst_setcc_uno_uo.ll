;; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b %s -o - | FileCheck %s

define dso_local void @store_isnan_f32(ptr %a, ptr %b, ptr %isnan_cmp) local_unnamed_addr {
entry:
  %arrayidx_a = getelementptr inbounds nuw float, ptr %a, i32 0
  %arrayidx_b = getelementptr inbounds nuw float, ptr %b, i32 0
  %0 = load <32 x float>, ptr %arrayidx_a, align 4
  %1 = load <32 x float>, ptr %arrayidx_b, align 4
  %.vectorized = fcmp uno <32 x float> %0, %1
  %.LS.instance = zext <32 x i1> %.vectorized to <32 x i32>
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %isnan_cmp, i32 0
  store <32 x i32> %.LS.instance, ptr %arrayidx1, align 4
  ret void
}

; CHECK:      store_isnan_f32
; CHECK:      [[RONE32:r[0-9]+]] = #1
; CHECK:      [[VOP2_F32:v[0-9]+]] = vxor([[VOP2_F32]],[[VOP2_F32]])
; CHECK:      [[VOP1_F32:v[0-9]+]] = vmemu(r0+#0)
; CHECK:      [[VONES32:v[0-9]+]] = vsplat([[RONE32]])
; CHECK:      [[Q1_F32:q[0-9]+]] = vcmp.eq([[VOP1_F32]].w,[[VOP1_F32]].w)
; CHECK:      [[VOP3_F32:v[0-9]+]] = vmemu(r1+#0)
; CHECK:      [[Q1_F32]] &= vcmp.eq([[VOP3_F32]].w,[[VOP3_F32]].w)
; CHECK:      [[VOUT_F32:v[0-9]+]] = vmux([[Q1_F32]],[[VOP2_F32]],[[VONES32]])
; CHECK:      vmemu(r2+#0) = [[VOUT_F32]]

define dso_local void @store_isnan_f16(ptr %a, ptr %b, ptr %isnan_cmp) local_unnamed_addr {
entry:
  %arrayidx_a = getelementptr inbounds nuw half, ptr %a, i32 0
  %arrayidx_b = getelementptr inbounds nuw half, ptr %b, i32 0
  %0 = load <64 x half>, ptr %arrayidx_a, align 2
  %1 = load <64 x half>, ptr %arrayidx_b, align 2
  %.vectorized = fcmp uno <64 x half> %0, %1
  %conv.LS.instance = zext <64 x i1> %.vectorized to <64 x i16>
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %isnan_cmp, i32 0
  store <64 x i16> %conv.LS.instance, ptr %arrayidx1, align 2
  ret void
}
; CHECK-LABEL: store_isnan_f16
; CHECK:       [[RONE16:r[0-9]+]] = #1
; CHECK:       [[VOP2_F16:v[0-9]+]] = vxor([[VOP2_F16]],[[VOP2_F16]])
; CHECK:       [[VOP1_F16:v[0-9]+]] = vmemu(r0+#0)
; CHECK:       [[VONES16:v[0-9]+]].h = vsplat([[RONE16]])
; CHECK:       [[Q1_F16:q[0-9]+]] = vcmp.eq([[VOP1_F16]].h,[[VOP1_F16]].h)
; CHECK:       [[VOP3_F16:v[0-9]+]] = vmemu(r1+#0)
; CHECK:       [[Q1_F16]] &= vcmp.eq([[VOP3_F16]].h,[[VOP3_F16]].h)
; CHECK:       [[VOUT_F16:v[0-9]+]] = vmux([[Q1_F16]],[[VOP2_F16]],[[VONES16]])
; CHECK:       vmemu(r2+#0) = [[VOUT_F32]]

define dso_local void @store_isordered_f32(ptr %a, ptr %b, ptr %isordered_cmp) local_unnamed_addr {
entry:
  %arrayidx_a = getelementptr inbounds nuw float, ptr %a, i32 0
  %arrayidx_b = getelementptr inbounds nuw float, ptr %b, i32 0
  %0 = load <32 x float>, ptr %arrayidx_a, align 4
  %1 = load <32 x float>, ptr %arrayidx_b, align 4
  %.vectorized = fcmp ord <32 x float> %0, %1
  %.LS.instance = zext <32 x i1> %.vectorized to <32 x i32>
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %isordered_cmp, i32 0
  store <32 x i32> %.LS.instance, ptr %arrayidx1, align 4
  ret void
}
; CHECK-LABEL: store_isordered_f32
; CHECK:       [[VOP2_ORD_F32:v[0-9]+]] = vxor([[VOP2_ORD_F32]],[[VOP2_ORD_F32]])
; CHECK:       [[VOP1_ORD_F32:v[0-9]+]] = vmemu(r0+#0)
; CHECK:       [[VONES_ORD_F32:v[0-9]+]] = vsplat([[RONE32]])
; CHECK:       [[Q1_ORD_F32:q[0-9]+]] = vcmp.eq([[VOP1_ORD_F32]].w,[[VOP1_ORD_F32]].w)
; CHECK:       [[VOP3_ORD_F32:v[0-9]+]] = vmemu(r1+#0)
; CHECK:       [[Q1_ORD_F32]] &= vcmp.eq([[VOP3_ORD_F32]].w,[[VOP3_ORD_F32]].w)
; CHECK:       [[VOUT_ORD_F32:v[0-9]+]] = vmux([[Q1_ORD_F32]],[[VONES_ORD_F32]],[[VOP2_ORD_F32]])
; CHECK:       vmemu(r2+#0) = [[VOUT_ORD_F32]]


define dso_local void @store_isordered_f16(ptr %a, ptr %b, ptr %isordered_cmp) local_unnamed_addr {
entry:
  %arrayidx_a = getelementptr inbounds nuw half, ptr %a, i32 0
  %arrayidx_b = getelementptr inbounds nuw half, ptr %b, i32 0
  %0 = load <64 x half>, ptr %arrayidx_a, align 2
  %1 = load <64 x half>, ptr %arrayidx_b, align 2
  %.vectorized = fcmp ord <64 x half> %0, %1
  %conv.LS.instance = zext <64 x i1> %.vectorized to <64 x i16>
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %isordered_cmp, i32 0
  store <64 x i16> %conv.LS.instance, ptr %arrayidx1, align 2
  ret void
}
; CHECK-LABEL: store_isordered_f16
; CHECK:       [[VOP2_ORD_F16:v[0-9]+]] = vxor([[VOP2_ORD_F16]],[[VOP2_ORD_F16]])
; CHECK:       [[VOP1_ORD_F16:v[0-9]+]] = vmemu(r0+#0)
; CHECK:       [[VONES_ORD_F16:v[0-9]+]].h = vsplat([[RONE16]])
; CHECK:       [[Q1_ORD_F16:q[0-9]+]] = vcmp.eq([[VOP1_ORD_F16]].h,[[VOP1_ORD_F16]].h)
; CHECK:       [[VOP3_ORD_F16:v[0-9]+]] = vmemu(r1+#0)
; CHECK:       [[Q1_ORD_F16]] &= vcmp.eq([[VOP3_ORD_F16]].h,[[VOP3_ORD_F16]].h)
; CHECK:       [[VOUT_ORD_F16:v[0-9]+]] = vmux([[Q1_ORD_F16]],[[VONES_ORD_F16]],[[VOP2_ORD_F16]])
; CHECK:       vmemu(r2+#0) = [[VOUT_ORD_F16]]
