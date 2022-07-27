; REQUIRES: x86_64-linux
; RUN: opt < %s -passes='pseudo-probe,loop-unroll-full' -verify-pseudo-probe -S -o %t 2>&1 | FileCheck %s --check-prefix=VERIFY
; RUN: FileCheck %s < %t

; VERIFY: *** Pseudo Probe Verification After LoopFullUnrollPass ***
; VERIFY: Function foo:
; VERIFY-DAG: Probe 6	previous factor 1.00	current factor 5.00
; VERIFY-DAG: Probe 4	previous factor 1.00	current factor 5.00

declare void @foo2() nounwind

define void @foo(i32 %x) {
bb:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
  %tmp = alloca [5 x ptr], align 16
  br label %bb7.preheader

bb3.loopexit:
  %spec.select.lcssa = phi i32 [ %spec.select, %bb10 ]
  %tmp5.not = icmp eq i32 %spec.select.lcssa, 0
  br i1 %tmp5.not, label %bb24, label %bb7.preheader

bb7.preheader:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 3, i32 0, i64 -1)
  %tmp1.06 = phi i32 [ 5, %bb ], [ %spec.select.lcssa, %bb3.loopexit ]
  br label %bb10

bb10:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; CHECK: call void @foo2(), !dbg ![[#PROBE6:]] 
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; CHECK: call void @foo2(), !dbg ![[#PROBE6:]] 
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; CHECK: call void @foo2(), !dbg ![[#PROBE6:]] 
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; CHECK: call void @foo2(), !dbg ![[#PROBE6:]] 
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; CHECK: call void @foo2(), !dbg ![[#PROBE6:]] 
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0, i64 -1)
  %indvars.iv = phi i64 [ 0, %bb7.preheader ], [ %indvars.iv.next, %bb10 ]
  %tmp1.14 = phi i32 [ %tmp1.06, %bb7.preheader ], [ %spec.select, %bb10 ]
  %tmp13 = getelementptr inbounds [5 x ptr], ptr %tmp, i64 0, i64 %indvars.iv
  %tmp14 = load ptr, ptr %tmp13, align 8
  %tmp15.not = icmp ne ptr %tmp14, null
  %tmp18 = sext i1 %tmp15.not to i32
  %spec.select = add nsw i32 %tmp1.14, %tmp18
  call void @foo2(), !dbg !12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 5
  br i1 %exitcond.not, label %bb3.loopexit, label %bb10, !llvm.loop !13

bb24:
; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 5, i32 0, i64 -1)
  ret void
}

;; A discriminator of 186646583 which is 0xb200037 in hexdecimal, stands for a direct call probe
;; with an index of 6 and a scale of -1%.
; CHECK: ![[#PROBE6]] = !DILocation(line: 2, column: 20, scope: ![[#SCOPE:]])
; CHECK: ![[#SCOPE]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 186646583)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.9.0"}
!12 = !DILocation(line: 2, column: 20, scope: !4)
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.unroll.full"}
