; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck --check-prefix=DBG_VALUE --check-prefix=LOOPVEC_4_1 %s
; RUN: opt -S -loop-vectorize -force-vector-width=2 -force-vector-interleave=3 < %s | FileCheck --check-prefix=DBG_VALUE --check-prefix=LOOPVEC_2_3 %s
; RUN: opt -S -loop-unroll  -unroll-count=5 < %s | FileCheck --check-prefix=DBG_VALUE --check-prefix=LOOPUNROLL_5 %s
; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=4 -loop-unroll -unroll-count=2 < %s | FileCheck --check-prefix=DBG_VALUE --check-prefix=LOOPVEC_UNROLL %s

; Test if vectorization/unroll factor is recorded in discriminator.
;
; Original source code:
;  1 int *a;
;  2 int *b;
;  3 
;  4 void foo() {
;  5   for (int i = 0; i < 4096; i++)
;  6     a[i] += b[i];
;  7 }

@a = local_unnamed_addr global i32* null, align 8
@b = local_unnamed_addr global i32* null, align 8
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define void @_Z3foov() local_unnamed_addr #0 !dbg !6 {
  %1 = load i32*, i32** @b, align 8, !dbg !8, !tbaa !9
  %2 = load i32*, i32** @a, align 8, !dbg !13, !tbaa !9
  br label %3, !dbg !14

; <label>:3:                                      ; preds = %3, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %3 ]
  %4 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv, !dbg !8
  %5 = load i32, i32* %4, align 4, !dbg !8, !tbaa !15
  %6 = getelementptr inbounds i32, i32* %2, i64 %indvars.iv, !dbg !13
  %7 = load i32, i32* %6, align 4, !dbg !17, !tbaa !15
  %8 = add nsw i32 %7, %5, !dbg !17
;DBG_VALUE: call void @llvm.dbg.declare{{.*}}!dbg ![[DBG:[0-9]*]]
  call void @llvm.dbg.declare(metadata i32 %8, metadata !22, metadata !DIExpression()), !dbg !17
  store i32 %8, i32* %6, align 4, !dbg !17, !tbaa !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !18
  %exitcond = icmp eq i64 %indvars.iv.next, 4096, !dbg !19
  br i1 %exitcond, label %9, label %3, !dbg !14, !llvm.loop !20

; <label>:9:                                      ; preds = %3
  ret void, !dbg !21
}

;DBG_VALUE: ![[TOP:[0-9]*]] = distinct !DISubprogram(name: "foo"
;LOOPVEC_4_1: discriminator: 17
;LOOPVEC_2_3: discriminator: 25
;LOOPUNROLL_5: discriminator: 21
; When unrolling after loop vectorize, both vec_body and remainder loop
; are unrolled.
;LOOPVEC_UNROLL: discriminator: 9
;LOOPVEC_UNROLL: discriminator: 385
;DBG_VALUE: ![[DBG]] = {{.*}}, scope: ![[TOP]]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, debugInfoForProfiling: true)
!1 = !DIFile(filename: "a.cc", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, unit: !0)
!8 = !DILocation(line: 6, column: 13, scope: !6)
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C++ TBAA"}
!13 = !DILocation(line: 6, column: 5, scope: !6)
!14 = !DILocation(line: 5, column: 3, scope: !6)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !11, i64 0}
!17 = !DILocation(line: 6, column: 10, scope: !6)
!18 = !DILocation(line: 5, column: 30, scope: !6)
!19 = !DILocation(line: 5, column: 21, scope: !6)
!20 = distinct !{!20, !14}
!21 = !DILocation(line: 7, column: 1, scope: !6)
!22 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !1, line: 10)
