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
;; CHECK: store_isnan_f32
;; CHECK: vcmp.eq({{v[0-9]+.w}},{{v[0-9]+.w}})

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

;; CHECK: store_isnan_f16
;; CHECK: vcmp.eq({{v[0-9]+.h}},{{v[0-9]+.h}})
