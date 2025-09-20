;; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b %s -o - | FileCheck %s

define dso_local void @store_isnan_f32(ptr %a, ptr %isnan_a) local_unnamed_addr {
entry:
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i32 0
  %0 = load <32 x float>, ptr %arrayidx, align 4
  %.vectorized = fcmp uno <32 x float> %0, zeroinitializer
  %.LS.instance = zext <32 x i1> %.vectorized to <32 x i32>
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %isnan_a, i32 0
  store <32 x i32> %.LS.instance, ptr %arrayidx1, align 4
  ret void
}
; CHECK:     store_isnan_f32
; CHECK:      [[VZERO32:v[0-9]+]] = vxor([[VZERO32]],[[VZERO32]])
; CHECK:      [[VLOAD32:v[0-9]+]] = vmemu(r0+#0)
; CHECK:      [[VONES32:v[0-9]+]] = vsplat([[RONE32:r[0-9]+]])
; CHECK:      {{q[0-9]+}} = vcmp.eq([[VLOAD32]].w,[[VLOAD32]].w)
; CHECK:      [[VOUT32:v[0-9]+]] = vmux({{q[0-9]+}},[[VZERO32]],[[VONES32]])
; CHECK:      vmemu(r1+#0) = [[VOUT32]]


define dso_local void @store_isnan_f16(ptr  %a, ptr %isnan_a) local_unnamed_addr {
entry:
  %arrayidx = getelementptr inbounds nuw half, ptr %a, i32 0
  %0 = load <64 x half>, ptr %arrayidx, align 2
  %.vectorized = fcmp uno <64 x half> %0, zeroinitializer
  %conv.LS.instance = zext <64 x i1> %.vectorized to <64 x i16>
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %isnan_a, i32 0
  store <64 x i16> %conv.LS.instance, ptr %arrayidx1, align 2
  ret void
}
; CHECK:      store_isnan_f16
; CHECK:      [[VZERO16:v[0-9]+]] = vxor([[VZERO16]],[[VZERO16]])
; CHECK:      [[VLOAD16:v[0-9]+]] = vmemu(r0+#0)
; CHECK:      [[VONES16:v[0-9]+]].h = vsplat([[RONE16:r[0-9]+]])
; CHECK:      {{q[0-9]+}} = vcmp.eq([[VLOAD16]].h,[[VLOAD16]].h)
; CHECK:      [[VOUT16:v[0-9]+]] = vmux({{q[0-9]+}},[[VZERO16]],[[VONES16]])
; CHECK:      vmemu(r1+#0) = [[VOUT16]]

define dso_local void @store_isordered_f32(ptr %a, ptr %isordered_a) local_unnamed_addr {
entry:
  %arrayidx = getelementptr inbounds nuw float, ptr %a, i32 0
  %0 = load <32 x float>, ptr %arrayidx, align 4
  %.vectorized = fcmp ord <32 x float> %0, zeroinitializer
  %.LS.instance = zext <32 x i1> %.vectorized to <32 x i32>
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %isordered_a, i32 0
  store <32 x i32> %.LS.instance, ptr %arrayidx1, align 4
  ret void
}
; CHECK:      store_isordered_f32
; CHECK:      [[V_ZERO32:v[0-9]+]] = vxor([[V_ZERO32]],[[V_ZERO32]])
; CHECK:      [[V_LOAD32:v[0-9]+]] = vmemu(r0+#0)
; CHECK:      [[V_ONES32:v[0-9]+]] = vsplat([[RO32:r[0-9]+]])
; CHECK:      {{q[0-9]+}} = vcmp.eq([[V_LOAD32]].w,[[V_LOAD32]].w)
; CHECK:      [[V_OUT32:v[0-9]+]] = vmux({{q[0-9]+}},[[V_ONES32]],[[V_ZERO32]])
; CHECK:      vmemu(r1+#0) = [[V_OUT32]]

define dso_local void @store_isordered_f16(ptr  %a, ptr %isordered_a) local_unnamed_addr {
entry:
  %arrayidx = getelementptr inbounds nuw half, ptr %a, i32 0
  %0 = load <64 x half>, ptr %arrayidx, align 2
  %.vectorized = fcmp ord <64 x half> %0, zeroinitializer
  %conv.LS.instance = zext <64 x i1> %.vectorized to <64 x i16>
  %arrayidx1 = getelementptr inbounds nuw i16, ptr %isordered_a, i32 0
  store <64 x i16> %conv.LS.instance, ptr %arrayidx1, align 2
  ret void
}
; CHECK:      store_isordered_f16
; CHECK:      [[V_ZERO16:v[0-9]+]] = vxor([[V_ZERO16]],[[V_ZERO16]])
; CHECK:      [[V_LOAD16:v[0-9]+]] = vmemu(r0+#0)
; CHECK:      [[V_ONES16:v[0-9]+]].h = vsplat([[RO16:r[0-9]+]])
; CHECK:      {{q[0-9]+}} = vcmp.eq([[V_LOAD16]].h,[[V_LOAD16]].h)
; CHECK:      [[V_OUT16:v[0-9]+]] = vmux({{q[0-9]+}},[[V_ONES16]],[[V_ZERO16]])
; CHECK:      vmemu(r1+#0) = [[V_OUT16]]
