; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo -enable-changed-func-dbinfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;  // clang -O2 -S -emit-llvm -g test.c
;  struct t { int a; };
;  char *tar(struct t *a, struct t *d);
;  __attribute__((noinline)) static char * foo(struct t *a, struct t *d, int b)
;  {
;    return tar(a, d);
;  }
;  char *bar(struct t *a, struct t *d)
;  {
;    return foo(a, d, 1);
;  }

; Function Attrs: nounwind uwtable
define dso_local ptr @bar(ptr noundef %0, ptr noundef %1) local_unnamed_addr #0 !dbg !10 {
    #dbg_value(ptr %0, !21, !DIExpression(), !23)
    #dbg_value(ptr %1, !22, !DIExpression(), !23)
  %3 = tail call fastcc ptr @foo(ptr noundef %0, ptr noundef %1), !dbg !24
  ret ptr %3, !dbg !25
}

; Function Attrs: noinline nounwind uwtable
define internal fastcc ptr @foo(ptr noundef %0, ptr noundef %1) unnamed_addr #1 !dbg !26 {
    #dbg_value(ptr %0, !30, !DIExpression(), !33)
    #dbg_value(ptr %1, !31, !DIExpression(), !33)
    #dbg_value(i32 poison, !32, !DIExpression(), !33)
  %3 = tail call ptr @tar(ptr noundef %0, ptr noundef %1) #3, !dbg !34
  ret ptr %3, !dbg !35
}

declare !dbg !36 ptr @tar(ptr noundef, ptr noundef) local_unnamed_addr #2

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/tests/sig-change/deadarg", checksumkind: CSK_MD5, checksum: "54bc89245cb23f69a8eb94fe2fb50a09")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)"}
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !11, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20, keyInstructions: true)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !15, !15}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 32, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !1, line: 1, baseType: !19, size: 32)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 7, type: !15)
!22 = !DILocalVariable(name: "d", arg: 2, scope: !10, file: !1, line: 7, type: !15)
!23 = !DILocation(line: 0, scope: !10)
!24 = !DILocation(line: 9, column: 10, scope: !10, atomGroup: 1, atomRank: 2)
!25 = !DILocation(line: 9, column: 3, scope: !10, atomGroup: 1, atomRank: 1)
!26 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !27, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29, keyInstructions: true)
!27 = !DISubroutineType(cc: DW_CC_nocall, types: !28)
!28 = !{!13, !15, !15, !19}
!29 = !{!30, !31, !32}
!30 = !DILocalVariable(name: "a", arg: 1, scope: !26, file: !1, line: 3, type: !15)
!31 = !DILocalVariable(name: "d", arg: 2, scope: !26, file: !1, line: 3, type: !15)
!32 = !DILocalVariable(name: "b", arg: 3, scope: !26, file: !1, line: 3, type: !19)
!33 = !DILocation(line: 0, scope: !26)
!34 = !DILocation(line: 5, column: 10, scope: !26, atomGroup: 1, atomRank: 2)
!35 = !DILocation(line: 5, column: 3, scope: !26, atomGroup: 1, atomRank: 1)
!36 = !DISubprogram(name: "tar", scope: !1, file: !1, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

; DWARF:        DW_TAG_inlined_subroutine
; DWARF-NEXT:     DW_AT_name      ("foo")
; DWARF-NEXT:     DW_AT_type
; DWARF-SAME:     "char *"
; DWARF-NEXT:     DW_AT_artificial        (true)
; DWARF-NEXT:     DW_AT_specification
; DWARF-SAME:     "foo"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("a")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "t *"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("d")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "t *"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
