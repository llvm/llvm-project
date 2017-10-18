; RUN: opt -relocation-model pic -march=hexagon -mcpu=hexagonv60 -O2 -S < %s | llc -march=hexagon -mcpu=hexagonv60 -relocation-model pic

; CHECK: jumpr

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @hex_h.s0.__outermost(i32 %h.stride.114) #0 {
entry:
  br i1 undef, label %"for h.s0.y.preheader", label %call_destructor.exit, !prof !1

call_destructor.exit:                             ; preds = %entry
  ret void

"for h.s0.y.preheader":                           ; preds = %entry
  %tmp22.us = mul i32 undef, %h.stride.114
  br label %"for h.s0.x.x.us"

"for h.s0.x.x.us":                                ; preds = %"for h.s0.x.x.us", %"for h.s0.y.preheader"
  %h.s0.x.x.us = phi i32 [ %5, %"for h.s0.x.x.us" ], [ 0, %"for h.s0.y.preheader" ]
  %0 = shl nsw i32 %h.s0.x.x.us, 5
  %1 = add i32 %0, %tmp22.us
  %2 = add nsw i32 %1, 16
  %3 = getelementptr inbounds i32, i32* null, i32 %2
  %4 = bitcast i32* %3 to <16 x i32>*
  store <16 x i32> zeroinitializer, <16 x i32>* %4, align 4, !tbaa !2
  %5 = add nuw nsw i32 %h.s0.x.x.us, 1
  br label %"for h.s0.x.x.us"
}

attributes #0 = { nounwind }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}

!0 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR (based on LLVM 3.9.0)"}
!1 = !{!"branch_weights", i32 1073741824, i32 0}
!2 = !{!3, !3, i64 0}
!3 = !{!"h", !4}
!4 = !{!"Halide buffer"}
