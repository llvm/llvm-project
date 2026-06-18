; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memset-cont-check < %s | FileCheck --check-prefixes=CHECK-DIM1,CHECK-DIM2 %s
; RUN: opt -passes=loop-idiom -S -loop-idiom-enable-memset-cont-check=false < %s | FileCheck --check-prefixes=CHECK-DIM1-NOCONT,CHECK-DIM2-NOCONT %s
; RUN: opt -passes=loop-idiom -S -verify-each -loop-idiom-enable-memset-cont-check < %s -o /dev/null
; RUN: opt -O3 -S -loop-idiom-enable-memset-cont-check < %s | FileCheck --check-prefixes=CHECK-DIM1-O3,CHECK-DIM2-O3 %s

; CHECK-DIM1-LABEL: @sub_dim1(
; CHECK-DIM1: loop.idiom.cont.cond:
; CHECK-DIM1: br i1 {{.*}}, label %loop.idiom.cont.then, label %loop.idiom.cont.else, !unpredictable
; CHECK-DIM1: call void @llvm.memset.p0.i64(ptr align 4 [[BASE:%.*]], i8 0, i64 16, i1 false)
; CHECK-DIM1: call void @llvm.memset.p0.i64(ptr align 4 [[BASE]], i8 0, i64 [[NBYTES:%.*]], i1 false)

; CHECK-DIM1-NOCONT-LABEL: @sub_dim1(
; CHECK-DIM1-NOCONT-NOT: loop.idiom.cont.cond:
; CHECK-DIM1-NOCONT: call void @llvm.memset.p0.i64(ptr align 4 [[BASE:%.*]], i8 0, i64 [[NBYTES:%.*]], i1 false)

; CHECK-DIM1-O3: call void @llvm.memset.p0.i64
; CHECK-DIM1-O3-SAME: i64 %

; CHECK-DIM2-LABEL: @sub_dim2(
; CHECK-DIM2: loop.idiom.cont.cond:
; CHECK-DIM2: br i1 {{.*}}, label %loop.idiom.cont.then, label %loop.idiom.cont.else, !unpredictable
; CHECK-DIM2: call void @llvm.memset.p0.i64(ptr align 8 [[BASE:%.*]], i8 0, i64 3200, i1 false)
; CHECK-DIM2: call void @llvm.memset.p0.i64(ptr align 8 [[BASE]], i8 0, i64 [[NBYTES:%.*]], i1 false)

; CHECK-DIM2-NOCONT-LABEL: @sub_dim2(
; CHECK-DIM2-NOCONT-NOT: loop.idiom.cont.cond:
; CHECK-DIM2-NOCONT: call void @llvm.memset.p0.i64(ptr align 8 [[BASE:%.*]], i8 0, i64 [[NBYTES:%.*]], i1 false)

; CHECK-DIM2-O3: tail call void @llvm.memset.p0.i64
; CHECK-DIM2-O3-SAME: i64 %

;;;;;;;;;;;;;;;;;;;;
;;subroutine sub(arr, ubnd)
;;  integer(4), intent(out) :: arr(4,100)
;;  integer, intent(in) :: ubnd
;;  arr(1:ubnd,:) = 0
;;end subroutine sub
;;;;;;;;;;;;;;;;;;;;

define void @sub_dim1(ptr noalias nocapture writeonly %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr {
L.entry:
  %0 = load i32, ptr %ubnd, align 4
  %1 = icmp slt i32 %0, 1
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
  %6 = getelementptr i32, ptr %3, i64 %5
  store i32 0, ptr %6, align 4
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

;;;;;;;;;;;;;;;;;;;;
;;subroutine sub(arr, ubnd)
;;  integer(8), intent(out) :: arr(100,4,100)
;;  integer, intent(in) :: ubnd
;;  arr(:,1:ubnd,:) = 0
;;end subroutine sub
;;;;;;;;;;;;;;;;;;;;

define void @sub_dim2(ptr noalias nocapture %arr, ptr noalias nocapture readonly %ubnd) local_unnamed_addr !dbg !5 {
L.entry:
  %0 = load i32, ptr %ubnd, align 4, !dbg !14, !tbaa !15
  %1 = icmp slt i32 %0, 1, !dbg !14
  br i1 %1, label %L.LB1_367, label %L.LB1_366.preheader, !dbg !14

L.LB1_366.preheader:                              ; preds = %L.entry
  %2 = sext i32 %0 to i64
  %3 = getelementptr i8, ptr %arr, i64 -4008
  br label %L.LB1_366

L.LB1_366:                                        ; preds = %L.LB1_366.preheader, %L.LB1_397
  %"i$a_359.0" = phi i64 [ %13, %L.LB1_397 ], [ 1, %L.LB1_366.preheader ], !dbg !14
  %4 = mul nuw nsw i64 %"i$a_359.0", 400
  br label %L.LB1_369

L.LB1_369:                                        ; preds = %L.LB1_396, %L.LB1_366
  %"i$b_360.0" = phi i64 [ 1, %L.LB1_366 ], [ %11, %L.LB1_396 ], !dbg !14
  %5 = mul nuw nsw i64 %"i$b_360.0", 100
  %6 = add nuw nsw i64 %5, %4
  br label %L.LB1_372

L.LB1_372:                                        ; preds = %L.LB1_372, %L.LB1_369
  %"i$c_361.0" = phi i64 [ 1, %L.LB1_369 ], [ %9, %L.LB1_372 ], !dbg !14
  %7 = add nuw nsw i64 %6, %"i$c_361.0", !dbg !14
  %8 = getelementptr i64, ptr %3, i64 %7, !dbg !14
  store i64 0, ptr %8, align 8, !dbg !14, !tbaa !19
  %9 = add nuw nsw i64 %"i$c_361.0", 1, !dbg !14
  %10 = icmp eq i64 %"i$c_361.0", 100
  br i1 %10, label %L.LB1_396, label %L.LB1_372, !dbg !14

L.LB1_396:                                        ; preds = %L.LB1_372
  %11 = add nuw nsw i64 %"i$b_360.0", 1, !dbg !14
  %12 = icmp eq i64 %"i$b_360.0", %2
  br i1 %12, label %L.LB1_397, label %L.LB1_369, !dbg !14

L.LB1_397:                                        ; preds = %L.LB1_396
  %13 = add nuw nsw i64 %"i$a_359.0", 1, !dbg !14
  %14 = icmp eq i64 %"i$a_359.0", 100
  br i1 %14, label %L.LB1_367.loopexit, label %L.LB1_366, !dbg !14

L.LB1_367.loopexit:                               ; preds = %L.LB1_397
  br label %L.LB1_367, !dbg !21

L.LB1_367:                                        ; preds = %L.LB1_367.loopexit, %L.entry
  ret void, !dbg !21
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: "flang", isOptimized: true, flags: "", runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4, nameTableKind: None)
!3 = !DIFile(filename: "memset-cont-check.f90", directory: ".")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !13}
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 2560000, align: 64, elements: !10)
!9 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!10 = !{!11, !12, !11}
!11 = !DISubrange(lowerBound: 1, upperBound: 100)
!12 = !DISubrange(lowerBound: 1, upperBound: 4)
!13 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 4, column: 1, scope: !5)
!15 = !{!16, !16, i64 0}
!16 = !{!"t1.4", !17, i64 0}
!17 = !{!"unlimited ptr", !18, i64 0}
!18 = !{!"Flang FAA 1"}
!19 = !{!20, !20, i64 0}
!20 = !{!"t1.a", !17, i64 0}
!21 = !DILocation(line: 5, column: 1, scope: !5)
