; RUN: opt -passes=mldst-motion -S %s -o - | FileCheck %s

;; $ cat test.cpp -n
;;      1	void fun(int *a, int cond) {
;;      2	  if (cond)
;;      3	    a[1] = 1;
;;      4	  else
;;      5	    a[1] = 2;
;;      6	}
;;
;; mldst-motion will merge and sink the stores in if.then and if.else into
;; if.end. The resultant PHI, gep and store should be attributed line zero
;; with the innermost common scope rather than picking a debug location from
;; one of the original stores.

; CHECK: if.end:
; CHECK-NEXT: %.sink = phi i32 [ 2, %if.else ], [ 1, %if.then ], !dbg ![[dbg:[0-9]+]]
; CHECK-NEXT: %0 = getelementptr inbounds i32, i32* %a, i64 1, !dbg ![[dbg:[0-9]+]]
; CHECK-NEXT: store i32 %.sink, i32* %0, align 4, !dbg ![[dbg:[0-9]+]]

; CHECK-DAG: ![[dbg]] = !DILocation(line: 0, scope: ![[scp:[0-9]+]]
; CHECK-DAG: ![[scp]] = distinct !DILexicalBlock(scope: ![[fun:[0-9]+]]
; CHECK-DAG: ![[fun]] = distinct !DISubprogram(name: "fun",

define void @_Z3funPii(i32* nocapture noundef writeonly %a, i32 noundef %cond) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32* %a, metadata !13, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 %cond, metadata !14, metadata !DIExpression()), !dbg !15
  %tobool.not = icmp eq i32 %cond, 0, !dbg !16
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !18

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 1, !dbg !19
  store i32 1, i32* %arrayidx, align 4, !dbg !20
  br label %if.end, !dbg !19

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 1, !dbg !25
  store i32 2, i32* %arrayidx1, align 4, !dbg !26
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !27
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funPii", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "cond", arg: 2, scope: !7, file: !1, line: 1, type: !11)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 2, column: 7, scope: !17)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 2, column: 7)
!18 = !DILocation(line: 2, column: 7, scope: !7)
!19 = !DILocation(line: 3, column: 5, scope: !17)
!20 = !DILocation(line: 3, column: 10, scope: !17)
!25 = !DILocation(line: 5, column: 5, scope: !17)
!26 = !DILocation(line: 5, column: 10, scope: !17)
!27 = !DILocation(line: 6, column: 1, scope: !7)
