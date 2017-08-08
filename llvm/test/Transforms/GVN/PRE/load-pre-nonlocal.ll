; RUN: opt -S -o - -basicaa -domtree -gvn %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%struct.S1 = type { i32, i32 }

@a2 = common global i32* null, align 8
@a = common global i32* null, align 8
@s1 = common global %struct.S1 zeroinitializer, align 8

; Check that GVN doesn't determine %2 is partially redundant.

; CHECK-LABEL: define i32 @volatile_load
; CHECK: for.body:
; CHECK: %2 = load i32, i32*
; CHECK: %3 = load volatile i32, i32*
; CHECK: for.cond.for.end_crit_edge:

define i32 @volatile_load(i32 %n) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load i32*, i32** @a2, align 8, !tbaa !1
  %1 = load i32*, i32** @a, align 8, !tbaa !1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %s.09 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %p.08 = phi i32* [ %0, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %2 = load i32, i32* %p.08, align 4, !tbaa !5
  %arrayidx = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 %2, i32* %arrayidx, align 4, !tbaa !5
  %3 = load volatile i32, i32* %p.08, align 4, !tbaa !5
  %add = add nsw i32 %3, %s.09
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %incdec.ptr = getelementptr inbounds i32, i32* %p.08, i64 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end

for.end:
  %s.0.lcssa = phi i32 [ %add.lcssa, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %s.0.lcssa
}

; %1 is partially redundant if %0 can be widened to a 64-bit load.
; But we should not widen %0 to 64-bit load.

; CHECK-LABEL: define i32 @overaligned_load
; CHECK: if.then:
; CHECK-NOT:   %0 = load i64
; CHECK-NOT:   [[LSHR:%[0-9]+]] = lshr i64 %0, 32, !dbg [[LSHR_LOC:![0-9]+]]
; CHECK-NOT:   trunc i64 [[LSHR]] to i32
; CHECK: if.end:
; CHECK: %1 = load i32, i32*
; CHECK-NOT: [[LSHR_LOC]] = !DILocation(line: 101, column: 1, scope: !{{.*}})

define i32 @overaligned_load(i32 %a, i32* nocapture %b) !dbg !13 {
entry:
  %cmp = icmp sgt i32 %a, 0, !dbg !14
  br i1 %cmp, label %if.then, label %if.else, !dbg !14

if.then:
  %0 = load i32, i32* getelementptr inbounds (%struct.S1, %struct.S1* @s1, i64 0, i32 0), align 8, !tbaa !5, !dbg !15
  br label %if.end, !dbg !15

if.else:
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 2, !dbg !16
  store i32 10, i32* %arrayidx, align 4, !tbaa !5, !dbg !16
  br label %if.end, !dbg !16

if.end:
  %i.0 = phi i32 [ %0, %if.then ], [ 0, %if.else ]
  %p.0 = phi i32* [ getelementptr inbounds (%struct.S1, %struct.S1* @s1, i64 0, i32 0), %if.then ], [ %b, %if.else ]
  %add.ptr = getelementptr inbounds i32, i32* %p.0, i64 1, !dbg !17
  %1 = load i32, i32* %add.ptr, align 4, !tbaa !5, !dbg !17
  %add1 = add nsw i32 %1, %i.0, !dbg !17
  ret i32 %add1, !dbg !17
}

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}

!llvm.module.flags = !{!7, !8, !9}
!llvm.dbg.cu = !{!18}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}

!10 = !{}
!11 = !DISubroutineType(types: !10)
!12 = !DIFile(filename: "test.cpp", directory: "/tmp")
!13 = distinct !DISubprogram(name: "test", scope: !12, file: !12, line: 99, type: !11, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !18, variables: !10)
!14 = !DILocation(line: 100, column: 1, scope: !13)
!15 = !DILocation(line: 101, column: 1, scope: !13)
!16 = !DILocation(line: 102, column: 1, scope: !13)
!17 = !DILocation(line: 103, column: 1, scope: !13)
!18 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !12,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
