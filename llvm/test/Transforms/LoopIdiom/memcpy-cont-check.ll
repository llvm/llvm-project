; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memcpy-cont-check < %s | FileCheck --check-prefixes=CHECK0 %s
; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memcpy-cont-check=false < %s | FileCheck --check-prefixes=CHECK-NOCONT %s
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memcpy-cont-check < %s -o /dev/null
; RUN: opt -passes='loop-mssa(loop-idiom),dse' -S -verify-memoryssa -loop-idiom-enable-memcpy-cont-check < %s -o /dev/null
; RUN: opt -passes='loop-mssa(loop-idiom,licm),simplifycfg' -S -verify-each -loop-idiom-enable-memcpy-cont-check < %s -o /dev/null

; CHECK0-LABEL: @sub_(
; CHECK0: loop.idiom.cont.cond:
; CHECK0: br i1 {{.*}}, label %loop.idiom.cont.then, label %loop.idiom.cont.else, !unpredictable
; CHECK0: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[DST0:%.*]], ptr align 1 [[SRC0:%.*]], i64 16, i1 false)
; CHECK0: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[DST0]], ptr align 1 [[SRC0]], i64 [[NBYTES0:%.*]], i1 false)

; CHECK-NOCONT-LABEL: @sub_(
; CHECK-NOCONT-NOT: loop.idiom.cont.cond:
; CHECK-NOCONT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[DSTN:%.*]], ptr align 1 [[SRCN:%.*]], i64 [[NBYTESN:%.*]], i1 false)

; CHECK0-LABEL: @sub_tbaa_(
; CHECK0: loop.idiom.cont.then:
; CHECK0-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[DST1:%.*]], ptr align 1 [[SRC1:%.*]], i64 16, i1 false), !tbaa [[TBAA0:![0-9]+]]{{$}}
; CHECK0: loop.idiom.cont.else:
; CHECK0-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[DST1]], ptr align 1 [[SRC1]], i64 [[NBYTES1:%.*]], i1 false){{$}}

; CHECK0-LABEL: @sub_preheader_defs_(
; CHECK0: loop.idiom.cont.cond:
; CHECK0: br i1 {{.*}}, label %loop.idiom.cont.then, label %loop.idiom.cont.else, !unpredictable

;;;;;;;;;;;;;;;;;;;;
;;subroutine sub(arr, ubnd)
;;  integer(4), intent(out) :: arr(4,100)
;;  integer, intent(in) :: ubnd
;;  arr(1:ubnd,:) = 0
;;end subroutine sub
;;;;;;;;;;;;;;;;;;;;

define void @sub_(ptr noalias nocapture writeonly %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
  %Base = alloca i32, i32 1600
  br i1 %1, label %L.LB1_364, label %L.LB1_363.preheader

L.LB1_363.preheader:                              ; preds = %L.entry
  %2 = sext i32 %0 to i64
  %3 = getelementptr i8, ptr %arr, i64 -20
  br label %L.LB1_363

L.LB1_363:                                        ; preds = %L.LB1_363.preheader, %L.LB1_388
  %.dY0001_365.0 = phi i64 [ %11, %L.LB1_388 ], [ 100, %L.LB1_363.preheader ]
  %"i$a_358.0" = phi i64 [ %10, %L.LB1_388 ], [ 1, %L.LB1_363.preheader ]
  %4 = shl nsw i64 %"i$a_358.0", 2
  br label %L.LB1_366

L.LB1_366:                                        ; preds = %L.LB1_366, %L.LB1_363
  %.dY0002_368.0 = phi i64 [ %2, %L.LB1_363 ], [ %8, %L.LB1_366 ]
  %"i$b_359.0" = phi i64 [ 1, %L.LB1_363 ], [ %7, %L.LB1_366 ]
  %5 = add nuw nsw i64 %"i$b_359.0", %4
  %I.0 = getelementptr i32, ptr %Base, i64 %5
  %6 = getelementptr i32, ptr %3, i64 %5
  %V = load i32, ptr %I.0, align 1
  store i32 %V, ptr %6, align 4
  %7 = add nuw nsw i64 %"i$b_359.0", 1
  %8 = add nsw i64 %.dY0002_368.0, -1
  %9 = icmp sgt i64 %.dY0002_368.0, 1
  br i1 %9, label %L.LB1_366, label %L.LB1_388

L.LB1_388:                                        ; preds = %L.LB1_366
  %10 = add nuw nsw i64 %"i$a_358.0", 1
  %11 = add nsw i64 %.dY0001_365.0, -1
  %12 = icmp sgt i64 %.dY0001_365.0, 1
  br i1 %12, label %L.LB1_363, label %L.LB1_364.loopexit

L.LB1_364.loopexit:                               ; preds = %L.LB1_388
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.LB1_364.loopexit, %L.entry
  ret void
}

define void @sub_tbaa_(ptr noalias nocapture writeonly %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
  %Base = alloca i32, i32 1600
  br i1 %1, label %L.LB1_364, label %L.LB1_363.preheader

L.LB1_363.preheader:                              ; preds = %L.entry
  %2 = sext i32 %0 to i64
  %3 = getelementptr i8, ptr %arr, i64 -20
  br label %L.LB1_363

L.LB1_363:                                        ; preds = %L.LB1_363.preheader, %L.LB1_388
  %.dY0001_365.0 = phi i64 [ %11, %L.LB1_388 ], [ 100, %L.LB1_363.preheader ]
  %"i$a_358.0" = phi i64 [ %10, %L.LB1_388 ], [ 1, %L.LB1_363.preheader ]
  %4 = shl nsw i64 %"i$a_358.0", 2
  br label %L.LB1_366

L.LB1_366:                                        ; preds = %L.LB1_366, %L.LB1_363
  %.dY0002_368.0 = phi i64 [ %2, %L.LB1_363 ], [ %8, %L.LB1_366 ]
  %"i$b_359.0" = phi i64 [ 1, %L.LB1_363 ], [ %7, %L.LB1_366 ]
  %5 = add nuw nsw i64 %"i$b_359.0", %4
  %I.0 = getelementptr i32, ptr %Base, i64 %5
  %6 = getelementptr i32, ptr %3, i64 %5
  %V = load i32, ptr %I.0, align 1, !tbaa !0
  store i32 %V, ptr %6, align 4, !tbaa !0
  %7 = add nuw nsw i64 %"i$b_359.0", 1
  %8 = add nsw i64 %.dY0002_368.0, -1
  %9 = icmp sgt i64 %.dY0002_368.0, 1
  br i1 %9, label %L.LB1_366, label %L.LB1_388

L.LB1_388:                                        ; preds = %L.LB1_366
  %10 = add nuw nsw i64 %"i$a_358.0", 1
  %11 = add nsw i64 %.dY0001_365.0, -1
  %12 = icmp sgt i64 %.dY0001_365.0, 1
  br i1 %12, label %L.LB1_363, label %L.LB1_364.loopexit

L.LB1_364.loopexit:                               ; preds = %L.LB1_388
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.LB1_364.loopexit, %L.entry
  ret void
}

define void @sub_preheader_defs_(ptr noalias nocapture writeonly %arr,
                                 ptr noalias nocapture readonly %ubnd)
    local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
  %tmp = alloca i32, align 4
  %Base = alloca i32, i32 1600
  br i1 %1, label %L.exit, label %L.preheader

L.preheader:                                      ; preds = %L.entry
  store i32 1, ptr %tmp, align 4
  store i32 2, ptr %tmp, align 4
  %2 = sext i32 %0 to i64
  %3 = getelementptr i8, ptr %arr, i64 -20
  br label %L.outer

L.outer:                                          ; preds = %L.preheader, %L.outer.latch
  %.dY0001_365.0 = phi i64 [ %11, %L.outer.latch ], [ 100, %L.preheader ]
  %"i$a_358.0" = phi i64 [ %10, %L.outer.latch ], [ 1, %L.preheader ]
  %4 = shl nsw i64 %"i$a_358.0", 2
  br label %L.inner

L.inner:                                          ; preds = %L.inner, %L.outer
  %.dY0002_368.0 = phi i64 [ %2, %L.outer ], [ %8, %L.inner ]
  %"i$b_359.0" = phi i64 [ 1, %L.outer ], [ %7, %L.inner ]
  %5 = add nuw nsw i64 %"i$b_359.0", %4
  %I.0 = getelementptr i32, ptr %Base, i64 %5
  %6 = getelementptr i32, ptr %3, i64 %5
  %V = load i32, ptr %I.0, align 1
  store i32 %V, ptr %6, align 4
  %7 = add nuw nsw i64 %"i$b_359.0", 1
  %8 = add nsw i64 %.dY0002_368.0, -1
  %9 = icmp sgt i64 %.dY0002_368.0, 1
  br i1 %9, label %L.inner, label %L.outer.latch

L.outer.latch:                                    ; preds = %L.inner
  %10 = add nuw nsw i64 %"i$a_358.0", 1
  %11 = add nsw i64 %.dY0001_365.0, -1
  %12 = icmp sgt i64 %.dY0001_365.0, 1
  br i1 %12, label %L.outer, label %L.exit

L.exit:                                           ; preds = %L.outer.latch, %L.entry
  ret void
}

!0 = !{!1, !2, i64 0, i64 8}
!1 = !{!3, i64 32, !"_ZTS1A", !2, i64 0, i64 8, !2, i64 8, i64 8, !2, i64 16, i64 8, !2, i64 24, i64 8}
!2 = !{!4, i64 8, !"double"}
!3 = !{!4, i64 0, !"omnipotent char"}
!4 = !{!"Simple C++ TBAA"}
