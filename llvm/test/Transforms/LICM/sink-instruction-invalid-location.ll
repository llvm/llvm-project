; RUN: opt --passes=licm -S -o - < %s | FileCheck %s

; Building the test case with `clang -O2 -g` the call to 'getInOrder'
; on line 16 is sunk out of the loop by LICM.
; According to 'HowToUpdateDebugInfo' the source location of the sunk
; call should be dropped.

;  1 __attribute__((noinline)) int getInOrder(int Idx) {
;  2   static int InOrder[] = { 0, 1 };
;  3   return InOrder[Idx];
;  4 }
;  5
;  6 __attribute__((noinline)) int getRandVar(int Idx) {
;  7   static int RandVars[] = { 4, 4 };
;  8   return RandVars[Idx];
;  9 }
; 10
; 11 int bounce() {
; 12   int Sum = 0;
; 13   int Extra = 0;
; 14   for (int I = 0; I < 2; ++I) {
; 15     Sum += getRandVar(I);
; 16     Extra = getInOrder(I);
; 17     Sum %= 4;
; 18   }
; 19   return Sum + Extra;
; 20 }

; CHECK-LABEL: for.end:
; CHECK: tail call noundef i32 @getInOrder({{.*}}), !dbg ![[DBG:[0-9]+]]
; CHECK-DAG: ![[DBG]] = !DILocation(line: 0, scope: ![[SCOPE:[0-9]+]]
; CHECK-DAG: ![[SCOPE]] = distinct !DISubprogram(name: "bounce",

define noundef i32 @getInOrder(i32 noundef %Idx) #0 !dbg !9 {
entry:
  ret i32 2
}

define noundef i32 @getRandVar(i32 noundef %Idx) #0 !dbg !15 {
entry:
  ret i32 4
}

define noundef i32 @bounce() !dbg !17 {
entry:
  br label %for.body, !dbg !20

for.body:                                         ; preds = %entry, %for.body
  %I.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %Sum.06 = phi i32 [ 0, %entry ], [ %rem, %for.body ]
  %call = tail call noundef i32 @getRandVar(i32 noundef %I.07), !dbg !21
  %add = add nsw i32 %call, %Sum.06, !dbg !21
  %call1 = tail call noundef i32 @getInOrder(i32 noundef %I.07), !dbg !22
  %rem = srem i32 %add, 4, !dbg !23
  %inc = add nuw nsw i32 %I.07, 1, !dbg !20
  %cmp = icmp ult i32 %inc, 2, !dbg !20
  br i1 %cmp, label %for.body, label %for.end, !dbg !20, !llvm.loop !24

for.end:                                          ; preds = %for.body
  %Sum.0.lcssa = phi i32 [ %rem, %for.body ], !dbg !27
  %Extra.0.lcssa = phi i32 [ %call1, %for.body ], !dbg !27
  %add2 = add nsw i32 %Extra.0.lcssa, %Sum.0.lcssa, !dbg !28
  ret i32 %add2, !dbg !28
}

attributes #0 = { noinline nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "", checksumkind: CSK_MD5, checksum: "cce1594f8ccf16528cd91ee35b0c1fea")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 17.0.0"}
!9 = distinct !DISubprogram(name: "getInOrder", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocation(line: 3, scope: !9)
!15 = distinct !DISubprogram(name: "getRandVar", scope: !1, file: !1, line: 6, type: !10, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!16 = !DILocation(line: 8, scope: !15)
!17 = distinct !DISubprogram(name: "bounce", scope: !1, file: !1, line: 11, type: !18, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!18 = !DISubroutineType(types: !19)
!19 = !{!12}
!20 = !DILocation(line: 14, scope: !17)
!21 = !DILocation(line: 15, scope: !17)
!22 = !DILocation(line: 16, scope: !17)
!23 = !DILocation(line: 17, scope: !17)
!24 = distinct !{!24, !20, !25, !26}
!25 = !DILocation(line: 18, scope: !17)
!26 = !{!"llvm.loop.mustprogress"}
!27 = !DILocation(line: 0, scope: !17)
!28 = !DILocation(line: 19, scope: !17)
