; REQUIRES: asserts
; RUN: llc -O3 -mtriple=hexagon < %s -o /dev/null
; Make sure that this doesn't crash.
; This test validates that the compiler would not assert when analyzing the
; offset of V6_vS32b_pred_ai instruction

%struct.pluto = type <{ ptr, i16, ptr }>

@global = external hidden unnamed_addr constant [62 x i8], align 1
@global.1 = external hidden unnamed_addr constant [47 x i8], align 1
@global.2 = hidden local_unnamed_addr constant %struct.pluto <{ ptr @global, i16 892, ptr @global.1 }>, align 1
@global.3 = local_unnamed_addr constant [1 x i32] zeroinitializer

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1

; Function Attrs: noinline nounwind
declare hidden fastcc void @zot(i32, i32, i32, i32) unnamed_addr #2

; Function Attrs: noinline nounwind
define void @barney(ptr nocapture %arg, ptr nocapture readnone %arg1, i8 signext %arg2, i32 %arg3, ptr nocapture readnone %arg4, ptr nocapture readnone %arg5, i32 %arg6, i32 %arg7, ptr nocapture readnone %arg8, ptr nocapture readnone %arg9, ptr nocapture readnone %arg10, ptr nocapture readnone %arg11, ptr nocapture readnone %arg12, ptr nocapture readnone %arg13, ptr nocapture readnone %arg14, ptr nocapture readnone %arg15, ptr nocapture readnone %arg16, ptr nocapture readnone %arg17) local_unnamed_addr #2 {
bb:
  %icmp = icmp ult i32 %arg3, 4
  tail call void @llvm.assume(i1 %icmp) #3
  %call = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B() #3
  br label %bb18

bb18:                                             ; preds = %bb22, %bb
  %phi = phi i32 [ %and, %bb22 ], [ %arg3, %bb ]
  %phi19 = phi i32 [ %add23, %bb22 ], [ 4, %bb ]
  %icmp20 = icmp eq i32 %phi, 0
  br i1 %icmp20, label %bb21, label %bb22

bb21:                                             ; preds = %bb18
  %shl = shl i32 %phi19, 8
  %getelementptr = getelementptr inbounds i8, ptr %arg, i32 %shl
  %bitcast = bitcast ptr %getelementptr to ptr
  store <32 x i32> %call, ptr %bitcast, align 128
  br label %bb22

bb22:                                             ; preds = %bb21, %bb18
  %add = add nuw nsw i32 %phi, 1
  %and = and i32 %add, 3
  %add23 = add nuw nsw i32 %phi19, 1
  %icmp24 = icmp eq i32 %add23, 8
  br i1 %icmp24, label %bb25, label %bb18

bb25:                                             ; preds = %bb22
  tail call fastcc void @zot(i32 %arg6, i32 %arg7, i32 0, i32 %arg3)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { noinline nounwind "target-cpu"="hexagonv68" "target-features"="+hvx-length128b,+hvxv68,+v68,+hvx-ieee-fp,-long-calls,-small-data" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
