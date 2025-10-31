; RUN: opt < %s -passes=loop-vectorize -S -pass-remarks=loop-vectorize 2>%t
; RUN: cat %t | FileCheck --check-prefix=CHECK-REMARKS %s

; CHECK-REMARKS: remark: repro.f90:9:5: vectorized loop (vectorization width: vscale x 2, interleaved count: 2)

; ModuleID = 'FIRModule'
source_filename = "FIRModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) vscale_range(1,16)
define void @repro_(ptr noalias readonly captures(none) %0, ptr noalias readonly captures(none) %1, ptr noalias captures(none) %2) local_unnamed_addr #0 !dbg !6 {
    #dbg_declare(ptr %0, !14, !DIExpression(), !15)
    #dbg_declare(ptr %2, !16, !DIExpression(), !15)
    #dbg_declare(ptr %1, !17, !DIExpression(), !15)
  %4 = load i32, ptr %0, align 4, !dbg !18, !tbaa !19
  %5 = icmp sgt i32 %4, 0, !dbg !18
  br i1 %5, label %.preheader.preheader, label %._crit_edge, !dbg !18

.preheader.preheader:                             ; preds = %3
  %6 = zext nneg i32 %4 to i64, !dbg !18
  %.pre = load double, ptr %2, align 8, !dbg !25, !tbaa !26
  %.pre5 = load double, ptr %1, align 8, !dbg !25, !tbaa !28
  %.phi.trans.insert = getelementptr i8, ptr %2, i64 8
  %.pre6 = load double, ptr %.phi.trans.insert, align 8, !dbg !25, !tbaa !26
  %.phi.trans.insert7 = getelementptr i8, ptr %1, i64 8
  %.pre8 = load double, ptr %.phi.trans.insert7, align 8, !dbg !25, !tbaa !28
  %.phi.trans.insert9 = getelementptr i8, ptr %2, i64 16
  %.pre10 = load double, ptr %.phi.trans.insert9, align 8, !dbg !25, !tbaa !26
  %.phi.trans.insert11 = getelementptr i8, ptr %1, i64 16
  %.pre12 = load double, ptr %.phi.trans.insert11, align 8, !dbg !25, !tbaa !28
  %.phi.trans.insert13 = getelementptr i8, ptr %2, i64 24
  %.pre14 = load double, ptr %.phi.trans.insert13, align 8, !dbg !25, !tbaa !26
  %.phi.trans.insert15 = getelementptr i8, ptr %1, i64 24
  %.pre16 = load double, ptr %.phi.trans.insert15, align 8, !dbg !25, !tbaa !28
  %.phi.trans.insert17 = getelementptr i8, ptr %2, i64 32
  %.pre18 = load double, ptr %.phi.trans.insert17, align 8, !dbg !25, !tbaa !26
  %.phi.trans.insert19 = getelementptr i8, ptr %1, i64 32
  %.pre20 = load double, ptr %.phi.trans.insert19, align 8, !dbg !25, !tbaa !28
  br label %.preheader, !dbg !30

.preheader:                                       ; preds = %.preheader.preheader, %.preheader
  %7 = phi double [ %17, %.preheader ], [ %.pre18, %.preheader.preheader ], !dbg !25
  %8 = phi double [ %16, %.preheader ], [ %.pre14, %.preheader.preheader ], !dbg !25
  %9 = phi double [ %15, %.preheader ], [ %.pre10, %.preheader.preheader ], !dbg !25
  %10 = phi double [ %14, %.preheader ], [ %.pre6, %.preheader.preheader ], !dbg !25
  %11 = phi double [ %13, %.preheader ], [ %.pre, %.preheader.preheader ], !dbg !25
  %12 = phi i64 [ %18, %.preheader ], [ %6, %.preheader.preheader ]
    #dbg_value(i64 1, !31, !DIExpression(), !32)
  %13 = fadd contract double %11, %.pre5, !dbg !25
    #dbg_value(i64 2, !31, !DIExpression(), !32)
  %14 = fadd contract double %10, %.pre8, !dbg !25
    #dbg_value(i64 3, !31, !DIExpression(), !32)
  %15 = fadd contract double %9, %.pre12, !dbg !25
    #dbg_value(i64 4, !31, !DIExpression(), !32)
  %16 = fadd contract double %8, %.pre16, !dbg !25
    #dbg_value(i64 5, !31, !DIExpression(), !32)
  %17 = fadd contract double %7, %.pre20, !dbg !25
    #dbg_value(i32 poison, !31, !DIExpression(), !32)
  %18 = add nsw i64 %12, -1, !dbg !18
  %19 = icmp samesign ugt i64 %12, 1, !dbg !18
  br i1 %19, label %.preheader, label %._crit_edge.loopexit, !dbg !18

._crit_edge.loopexit:                             ; preds = %.preheader
  %.lcssa28 = phi double [ %13, %.preheader ], !dbg !25
  %.lcssa27 = phi double [ %14, %.preheader ], !dbg !25
  %.lcssa26 = phi double [ %15, %.preheader ], !dbg !25
  %.lcssa25 = phi double [ %16, %.preheader ], !dbg !25
  %.lcssa = phi double [ %17, %.preheader ], !dbg !25
  store double %.lcssa28, ptr %2, align 8, !dbg !25, !tbaa !26
  store double %.lcssa27, ptr %.phi.trans.insert, align 8, !dbg !25, !tbaa !26
  store double %.lcssa26, ptr %.phi.trans.insert9, align 8, !dbg !25, !tbaa !26
  store double %.lcssa25, ptr %.phi.trans.insert13, align 8, !dbg !25, !tbaa !26
  store double %.lcssa, ptr %.phi.trans.insert17, align 8, !dbg !25, !tbaa !26
  br label %._crit_edge, !dbg !33

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %3
    #dbg_value(i32 poison, !34, !DIExpression(), !32)
  ret void, !dbg !33
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) vscale_range(1,16) "frame-pointer"="non-leaf" "target-cpu"="generic" "target-features"="+outline-atomics,+v8a,+fp-armv8,+fullfp16,+neon,+sve" }

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !1, producer: "flang version 22.0.0 (https://github.com/llvm/llvm-project.git c8f5c602c897d2345c1cfd8d886c1325598dbdc6)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "repro.f90", directory: "/path/to")
!2 = !{!"flang version 22.0.0 (https://github.com/llvm/llvm-project.git c8f5c602c897d2345c1cfd8d886c1325598dbdc6)"}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = distinct !DISubprogram(name: "repro", linkageName: "repro_", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{null, !9, !10, !10}
!9 = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, elements: !12)
!11 = !DIBasicType(name: "real(kind=8)", size: 64, encoding: DW_ATE_float)
!12 = !{!13}
!13 = !DISubrange(count: 5)
!14 = !DILocalVariable(name: "iter", arg: 1, scope: !6, file: !1, line: 5, type: !9)
!15 = !DILocation(line: 2, column: 1, scope: !6)
!16 = !DILocalVariable(name: "sum", arg: 3, scope: !6, file: !1, line: 6, type: !10)
!17 = !DILocalVariable(name: "v", arg: 2, scope: !6, file: !1, line: 6, type: !10)
!18 = !DILocation(line: 8, column: 3, scope: !6)
!19 = !{!20, !20, i64 0}
!20 = !{!"dummy arg data/_QFreproEiter", !21, i64 0}
!21 = !{!"dummy arg data", !22, i64 0}
!22 = !{!"any data access", !23, i64 0}
!23 = !{!"any access", !24, i64 0}
!24 = !{!"Flang function root _QPrepro"}
!25 = !DILocation(line: 10, column: 7, scope: !6)
!26 = !{!27, !27, i64 0}
!27 = !{!"dummy arg data/_QFreproEsum", !21, i64 0}
!28 = !{!29, !29, i64 0}
!29 = !{!"dummy arg data/_QFreproEv", !21, i64 0}
!30 = !DILocation(line: 9, column: 5, scope: !6)
!31 = !DILocalVariable(name: "m", scope: !6, file: !1, line: 5, type: !9)
!32 = !DILocation(line: 0, scope: !6)
!33 = !DILocation(line: 13, column: 1, scope: !6)
!34 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 5, type: !9)
;.
; CHECK: [[META0:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: [[META1:![0-9]+]], producer: "{{.*}}flang version {{.*}}", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
; CHECK: [[META1]] = !DIFile(filename: "{{.*}}repro.f90", directory: {{.*}})
; CHECK: [[DBG6]] = distinct !DISubprogram(name: "repro", linkageName: "repro_", scope: [[META1]], file: [[META1]], line: 2, type: [[META7:![0-9]+]], scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: [[META0]])
; CHECK: [[META7]] = !DISubroutineType(cc: DW_CC_normal, types: [[META8:![0-9]+]])
; CHECK: [[META8]] = !{null, [[META9:![0-9]+]], [[META10:![0-9]+]], [[META10]]}
; CHECK: [[META9]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
; CHECK: [[META10]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[META11:![0-9]+]], elements: [[META12:![0-9]+]])
; CHECK: [[META11]] = !DIBasicType(name: "real(kind=8)", size: 64, encoding: DW_ATE_float)
; CHECK: [[META12]] = !{[[META13:![0-9]+]]}
; CHECK: [[META13]] = !DISubrange(count: 5)
; CHECK: [[META14]] = !DILocalVariable(name: "iter", arg: 1, scope: [[DBG6]], file: [[META1]], line: 5, type: [[META9]])
; CHECK: [[META15]] = !DILocation(line: 2, column: 1, scope: [[DBG6]])
; CHECK: [[META16]] = !DILocalVariable(name: "sum", arg: 3, scope: [[DBG6]], file: [[META1]], line: 6, type: [[META10]])
; CHECK: [[META17]] = !DILocalVariable(name: "v", arg: 2, scope: [[DBG6]], file: [[META1]], line: 6, type: [[META10]])
; CHECK: [[DBG18]] = !DILocation(line: 8, column: 3, scope: [[DBG6]])
; CHECK: [[DUMMY_ARG_DATA/_QFREPROEITER_TBAA19]] = !{[[META20:![0-9]+]], [[META20]], i64 0}
; CHECK: [[META20]] = !{!"dummy arg data/_QFreproEiter", [[META21:![0-9]+]], i64 0}
; CHECK: [[META21]] = !{!"dummy arg data", [[META22:![0-9]+]], i64 0}
; CHECK: [[META22]] = !{!"any data access", [[META23:![0-9]+]], i64 0}
; CHECK: [[META23]] = !{!"any access", [[META24:![0-9]+]], i64 0}
; CHECK: [[META24]] = !{!"Flang function root _QPrepro"}
; CHECK: [[DBG25]] = !DILocation(line: 10, column: 7, scope: [[DBG6]])
; CHECK: [[DUMMY_ARG_DATA/_QFREPROESUM_TBAA26]] = !{[[META27:![0-9]+]], [[META27]], i64 0}
; CHECK: [[META27]] = !{!"dummy arg data/_QFreproEsum", [[META21]], i64 0}
; CHECK: [[DUMMY_ARG_DATA/_QFREPROEV_TBAA28]] = !{[[META29:![0-9]+]], [[META29]], i64 0}
; CHECK: [[META29]] = !{!"dummy arg data/_QFreproEv", [[META21]], i64 0}
; CHECK: [[DBG30]] = !DILocation(line: 9, column: 5, scope: [[DBG6]])
; CHECK: [[LOOP31]] = distinct !{[[LOOP31]], [[META32:![0-9]+]], [[META33:![0-9]+]]}
; CHECK: [[META32]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META33]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[META34]] = !DILocalVariable(name: "m", scope: [[DBG6]], file: [[META1]], line: 5, type: [[META9]])
; CHECK: [[META35]] = !DILocation(line: 0, scope: [[DBG6]])
; CHECK: [[LOOP36]] = distinct !{[[LOOP36]], [[META33]], [[META32]]}
; CHECK: [[DBG37]] = !DILocation(line: 13, column: 1, scope: [[DBG6]])
; CHECK: [[META38]] = !DILocalVariable(name: "i", scope: [[DBG6]], file: [[META1]], line: 5, type: [[META9]])
;.
