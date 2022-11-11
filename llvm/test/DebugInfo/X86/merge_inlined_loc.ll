; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - | FileCheck %s

; Generated with "clang -g -c -emit-llvm -S -O3"

; This will test several features of merging debug locations. Importantly,
; locations with the same source line but different scopes should be merged to
; a line zero location at the nearest common scope and inlining. The location
; of the single call to "common" (the two calls are collapsed together by
; BranchFolding) should be attributed to line zero inside the wrapper2 inlined
; scope within f1.

; void common();
; inline void wrapper() { common(); }
; extern bool b;
; void sink();
; inline void wrapper2() {
;   if (b) {
;     sink();
;     wrapper();
;   } else
;     wrapper();
; }
; void f1() { wrapper2(); }

; Ensure there is only one inlined_subroutine (for wrapper2, none for wrapper)
; & that its address range includes the call to 'common'.

; CHECK: jmp _Z6commonv
; CHECK-NEXT: [[LABEL:.*]]:

; CHECK: .section .debug_info
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG\|End Of Children}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG\|End Of Children}}
; CHECK:     [[LABEL]]-{{.*}} DW_AT_high_pc
; CHECK-NOT: DW_TAG



@b = external dso_local local_unnamed_addr global i8, align 1

; Function Attrs: uwtable
define dso_local void @_Z2f1v() local_unnamed_addr !dbg !7 {
entry:
  %0 = load i8, ptr @b, align 1, !dbg !10, !tbaa !14, !range !18
  %tobool.i = icmp eq i8 %0, 0, !dbg !10
  br i1 %tobool.i, label %if.else.i, label %if.then.i, !dbg !19

if.then.i:                                        ; preds = %entry
  tail call void @_Z4sinkv(), !dbg !20
  tail call void @_Z6commonv(), !dbg !22
  br label %_Z8wrapper2v.exit, !dbg !25

if.else.i:                                        ; preds = %entry
  tail call void @_Z6commonv(), !dbg !26
  br label %_Z8wrapper2v.exit

_Z8wrapper2v.exit:                                ; preds = %if.then.i, %if.else.i
  ret void, !dbg !28
}

declare dso_local void @_Z4sinkv() local_unnamed_addr

declare dso_local void @_Z6commonv() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 340559) (llvm/trunk 340572)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "merge_loc.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 340559) (llvm/trunk 340572)"}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 12, type: !8, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 6, column: 7, scope: !11, inlinedAt: !13)
!11 = distinct !DILexicalBlock(scope: !12, file: !1, line: 6, column: 7)
!12 = distinct !DISubprogram(name: "wrapper2", linkageName: "_Z8wrapper2v", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!13 = distinct !DILocation(line: 13, column: 3, scope: !7)
!14 = !{!15, !15, i64 0}
!15 = !{!"bool", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
!18 = !{i8 0, i8 2}
!19 = !DILocation(line: 6, column: 7, scope: !12, inlinedAt: !13)
!20 = !DILocation(line: 7, column: 5, scope: !21, inlinedAt: !13)
!21 = distinct !DILexicalBlock(scope: !11, file: !1, line: 6, column: 10)
!22 = !DILocation(line: 2, column: 25, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "wrapper", linkageName: "_Z7wrapperv", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!24 = distinct !DILocation(line: 8, column: 5, scope: !21, inlinedAt: !13)
!25 = !DILocation(line: 9, column: 3, scope: !21, inlinedAt: !13)
!26 = !DILocation(line: 2, column: 25, scope: !23, inlinedAt: !27)
!27 = distinct !DILocation(line: 10, column: 5, scope: !11, inlinedAt: !13)
!28 = !DILocation(line: 14, column: 1, scope: !7)
