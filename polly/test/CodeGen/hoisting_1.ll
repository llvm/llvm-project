; RUN: opt %loadNPMPolly -aa-pipeline=tbaa -passes=polly-codegen -polly-allow-differing-element-types -disable-output %s
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

; Function Attrs: nounwind uwtable
define void @foo(ptr %arg) #0 {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp3 = getelementptr inbounds %struct.hoge, ptr %arg, i32 0, i32 42
  %tmp = load ptr, ptr %tmp3, align 8, !tbaa !1
  %tmp5 = getelementptr inbounds %struct.foo.1, ptr %tmp, i32 0, i32 7
  %tmp6 = load i32, ptr %tmp5, align 4, !tbaa !5
  %tmp7 = getelementptr inbounds %struct.hoge, ptr %arg, i32 0, i32 43
  store i32 %tmp6, ptr %tmp7, align 8, !tbaa !8
  br i1 false, label %bb8, label %bb9

bb8:                                              ; preds = %bb2
  br label %bb9

bb9:                                              ; preds = %bb8, %bb2
  br label %bb10

bb10:                                             ; preds = %bb9
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 259751) (llvm/trunk 259869)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !7, i64 28}
!6 = !{!"", !7, i64 0, !7, i64 4, !7, i64 8, !7, i64 12, !7, i64 16, !7, i64 20, !7, i64 24, !7, i64 28, !7, i64 32, !7, i64 36, !7, i64 40, !7, i64 44, !7, i64 48, !7, i64 52, !7, i64 56, !7, i64 60, !7, i64 64, !7, i64 68, !7, i64 72, !2, i64 80, !2, i64 88}
!7 = !{!"int", !3, i64 0}
!8 = !{!9, !7, i64 352}
!9 = !{!"jpeg_compress_struct", !2, i64 0, !2, i64 8, !2, i64 16, !7, i64 24, !7, i64 28, !2, i64 32, !7, i64 40, !7, i64 44, !7, i64 48, !3, i64 52, !10, i64 56, !7, i64 64, !7, i64 68, !3, i64 72, !2, i64 80, !3, i64 88, !3, i64 120, !3, i64 152, !3, i64 184, !3, i64 200, !3, i64 216, !7, i64 232, !2, i64 240, !7, i64 248, !7, i64 252, !7, i64 256, !7, i64 260, !7, i64 264, !3, i64 268, !7, i64 272, !7, i64 276, !7, i64 280, !3, i64 284, !11, i64 286, !11, i64 288, !7, i64 292, !7, i64 296, !7, i64 300, !7, i64 304, !7, i64 308, !7, i64 312, !7, i64 316, !3, i64 320, !7, i64 352, !7, i64 356, !7, i64 360, !3, i64 364, !7, i64 404, !7, i64 408, !7, i64 412, !7, i64 416, !2, i64 424, !2, i64 432, !2, i64 440, !2, i64 448, !2, i64 456, !2, i64 464, !2, i64 472, !2, i64 480, !2, i64 488}
!10 = !{!"double", !3, i64 0}
!11 = !{!"short", !3, i64 0}
