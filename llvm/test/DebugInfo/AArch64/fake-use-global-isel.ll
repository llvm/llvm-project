; REQUIRES: object-emission

; Make sure the fake use of 'b' at the end of 'foo' causes location information for 'b'
; to extend all the way to the end of the function.
; Duplicates `DebugInfo/X86/fake-use.ll` for global-isel.

; RUN: %llc_dwarf -O2 --global-isel=1 -mtriple=aarch64--linux-gnu -filetype=obj -dwarf-linkage-names=Abstract < %s | llvm-dwarfdump --debug-info --debug-line -v - -o %t
; RUN: %python %p/../Inputs/check-fake-use.py %t
; RUN: sed -e 's,call void (...) @llvm.fake.use,;,' %s \
; RUN:   | %llc_dwarf - -O2 --global-isel=1 -mtriple=aarch64--linux-gnu -filetype=obj -dwarf-linkage-names=Abstract \
; RUN:   | llvm-dwarfdump --debug-info --debug-line -v - -o %t
; RUN: not %python %p/../Inputs/check-fake-use.py %t

; Generated with:
; clang -O2 -g -S -emit-llvm -fextend-this-ptr fake-use.c
;
; int glob[10];
; extern void bar();
;
; int foo(int b, int i)
; {
;    int loc = glob[i] * 2;
;    if (b) {
;      glob[2] = loc;
;      bar();
;    }
;    return loc;
; }
;
; ModuleID = 't2.c'
source_filename = "t2.c"

@glob = common local_unnamed_addr global [10 x i32] zeroinitializer, align 16, !dbg !0

; Function Attrs: nounwind sspstrong uwtable
define i32 @foo(i32 %b, i32 %i) local_unnamed_addr optdebug !dbg !13 {
entry:
    #dbg_value(i32 %b, !17, !20, !21)
  %c = add i32 %b, 42
  %tobool = icmp sgt i32 %c, 2, !dbg !27
  tail call void (...) @bar() #2, !dbg !32
  %idxprom = sext i32 %i to i64, !dbg !22
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* @glob, i64 0, i64 %idxprom, !dbg !22
  %0 = load i32, i32* %arrayidx, align 4, !dbg !22, !tbaa !23
  %mul = shl nsw i32 %0, 1, !dbg !22
  br i1 %tobool, label %if.end, label %if.then, !dbg !29

if.then:                                          ; preds = %entry
  store i32 %mul, i32* getelementptr inbounds ([10 x i32], [10 x i32]* @glob, i64 0, i64 2), align 8, !dbg !30, !tbaa !23
  tail call void (...) @bar() #2, !dbg !32
  br label %if.end, !dbg !33

if.end:                                           ; preds = %entry, %if.then
  call void (...) @llvm.fake.use(i32 %b), !dbg !34
  ret i32 %mul, !dbg !35
}

declare void @bar(...) local_unnamed_addr

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "glob", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true), expr: !DIExpression())
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "t2.c", directory: "/")
!3 = !{}
!4 = !{!0}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 320, align: 32, elements: !7)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{!8}
!8 = !DISubrange(count: 10)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 4.0.0"}
!13 = distinct !DISubprogram(name: "foo", scope: !2, file: !2, line: 4, type: !14, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !1, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!6, !6, !6}
!16 = !{!17, !19}
!17 = !DILocalVariable(name: "b", arg: 1, scope: !13, file: !2, line: 4, type: !6)
!19 = !DILocalVariable(name: "loc", scope: !13, file: !2, line: 6, type: !6)
!20 = !DIExpression()
!21 = !DILocation(line: 4, scope: !13)
!22 = !DILocation(line: 6, scope: !13)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 7, scope: !28)
!28 = distinct !DILexicalBlock(scope: !13, file: !2, line: 7)
!29 = !DILocation(line: 7, scope: !13)
!30 = !DILocation(line: 8, scope: !31)
!31 = distinct !DILexicalBlock(scope: !28, file: !2, line: 7)
!32 = !DILocation(line: 9, scope: !31)
!33 = !DILocation(line: 10, scope: !31)
!34 = !DILocation(line: 12, scope: !13)
!35 = !DILocation(line: 11, scope: !13)
