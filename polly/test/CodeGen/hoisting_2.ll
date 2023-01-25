; RUN: opt %loadPolly -tbaa -polly-codegen -polly-allow-differing-element-types -disable-output %s
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hoge = type { ptr, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, double, i32, i32, i32, ptr, [4 x ptr], [4 x ptr], [4 x ptr], [16 x i8], [16 x i8], [16 x i8], i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i16, i16, i32, i32, i32, i32, i32, i32, i32, [4 x ptr], i32, i32, i32, [10 x i32], i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.widget = type { ptr, ptr, ptr, ptr, ptr, i32, %struct.hoge.0, i32, i64, ptr, i32, ptr, i32, i32 }
%struct.quux = type { ptr, ptr, ptr, i32, i32 }
%struct.hoge.0 = type { [8 x i32], [48 x i8] }
%struct.barney = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64 }
%struct.ham = type opaque
%struct.wombat = type opaque
%struct.foo = type { ptr, i64, i64, i32, i32 }
%struct.wibble = type { ptr, i64, ptr, ptr, ptr }
%struct.foo.1 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr }
%struct.hoge.2 = type { [64 x i16], i32 }
%struct.blam = type { [17 x i8], [256 x i8], i32 }
%struct.barney.3 = type { i32, [4 x i32], i32, i32, i32, i32 }
%struct.foo.4 = type { ptr, ptr, ptr, i32, i32 }
%struct.wombat.5 = type { ptr, ptr }
%struct.blam.6 = type { ptr, ptr }
%struct.foo.7 = type { ptr, ptr }
%struct.bar = type { ptr, ptr, ptr, ptr, ptr, ptr }
%struct.wibble.8 = type { ptr, ptr }
%struct.barney.9 = type { ptr, ptr, i32 }
%struct.hoge.10 = type { ptr, ptr }
%struct.bar.11 = type { ptr, ptr, ptr }
%struct.foo.12 = type { %struct.foo.4, i32, i32, i32, i32 }

; Function Attrs: nounwind uwtable
define void @eggs(ptr %arg) #0 {
bb:
  %tmp = load ptr, ptr undef, align 8, !tbaa !1
  br label %bb5

bb5:                                              ; preds = %bb
  %tmp6 = getelementptr inbounds %struct.hoge, ptr %arg, i32 0, i32 51
  %tmp7 = load ptr, ptr %tmp6, align 8, !tbaa !9
  %tmp9 = getelementptr inbounds %struct.foo.12, ptr %tmp7, i32 0, i32 4
  %tmp10 = load i32, ptr %tmp9, align 4, !tbaa !10
  %tmp151 = load i32, ptr %tmp, align 4, !tbaa !13
  %tmp162 = icmp slt i32 0, %tmp151
  br i1 %tmp162, label %bb17.lr.ph, label %bb22

bb17.lr.ph:                                       ; preds = %bb5
  br label %bb17

bb17:                                             ; preds = %bb17.lr.ph, %bb17
  %tmp143 = phi i32 [ 0, %bb17.lr.ph ], [ %tmp21, %bb17 ]
  %tmp18 = sext i32 %tmp143 to i64
  %tmp19 = getelementptr inbounds %struct.hoge, ptr %arg, i32 0, i32 42
  %tmp20 = getelementptr inbounds [4 x ptr], ptr %tmp19, i64 0, i64 %tmp18
  store ptr undef, ptr %tmp20, align 8, !tbaa !15
  %tmp21 = add nsw i32 %tmp143, 1
  %tmp15 = load i32, ptr %tmp, align 4, !tbaa !13
  %tmp16 = icmp slt i32 %tmp21, %tmp15
  br i1 %tmp16, label %bb17, label %bb13.bb22_crit_edge

bb13.bb22_crit_edge:                              ; preds = %bb17
  br label %bb22

bb22:                                             ; preds = %bb13.bb22_crit_edge, %bb5
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 259751) (llvm/trunk 259869)"}
!1 = !{!2, !3, i64 240}
!2 = !{!"jpeg_compress_struct", !3, i64 0, !3, i64 8, !3, i64 16, !6, i64 24, !6, i64 28, !3, i64 32, !6, i64 40, !6, i64 44, !6, i64 48, !4, i64 52, !7, i64 56, !6, i64 64, !6, i64 68, !4, i64 72, !3, i64 80, !4, i64 88, !4, i64 120, !4, i64 152, !4, i64 184, !4, i64 200, !4, i64 216, !6, i64 232, !3, i64 240, !6, i64 248, !6, i64 252, !6, i64 256, !6, i64 260, !6, i64 264, !4, i64 268, !6, i64 272, !6, i64 276, !6, i64 280, !4, i64 284, !8, i64 286, !8, i64 288, !6, i64 292, !6, i64 296, !6, i64 300, !6, i64 304, !6, i64 308, !6, i64 312, !6, i64 316, !4, i64 320, !6, i64 352, !6, i64 356, !6, i64 360, !4, i64 364, !6, i64 404, !6, i64 408, !6, i64 412, !6, i64 416, !3, i64 424, !3, i64 432, !3, i64 440, !3, i64 448, !3, i64 456, !3, i64 464, !3, i64 472, !3, i64 480, !3, i64 488}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"int", !4, i64 0}
!7 = !{!"double", !4, i64 0}
!8 = !{!"short", !4, i64 0}
!9 = !{!2, !3, i64 424}
!10 = !{!11, !6, i64 44}
!11 = !{!"", !12, i64 0, !4, i64 32, !6, i64 36, !6, i64 40, !6, i64 44}
!12 = !{!"jpeg_comp_master", !3, i64 0, !3, i64 8, !3, i64 16, !6, i64 24, !6, i64 28}
!13 = !{!14, !6, i64 0}
!14 = !{!"", !6, i64 0, !4, i64 4, !6, i64 20, !6, i64 24, !6, i64 28, !6, i64 32}
!15 = !{!3, !3, i64 0}
