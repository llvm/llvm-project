; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;   // clang -O2 -S -emit-llvm -g test2.c -mllvm -disable-changed-func-dbinfo
;   struct t { long a; long b; long c;};
;   long foo(struct t arg) {
;     return arg.a * arg.c;
;   }

%struct.t = type { i64, i64, i64 }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local i64 @foo(ptr noundef readonly byval(%struct.t) align 8 captures(none) %0) local_unnamed_addr #0 !dbg !9 {
    #dbg_declare(ptr %0, !19, !DIExpression(), !20)
  %2 = load i64, ptr %0, align 8, !dbg !21, !tbaa !22
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16, !dbg !27
  %4 = load i64, ptr %3, align 8, !dbg !27, !tbaa !28
  %5 = mul nsw i64 %4, %2, !dbg !29
  ret i64 %5, !dbg !30
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test2.c", directory: "/tmp/home/yhs/tests/sig-change/struct", checksumkind: CSK_MD5, checksum: "d58648f18d3fa35e3d95b364b4a95c4c")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18, keyInstructions: true)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 192, elements: !14)
!14 = !{!15, !16, !17}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 1, baseType: !12, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 1, baseType: !12, size: 64, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !1, line: 1, baseType: !12, size: 64, offset: 128)
!18 = !{!19}
!19 = !DILocalVariable(name: "arg", arg: 1, scope: !9, file: !1, line: 2, type: !13)
!20 = !DILocation(line: 2, column: 19, scope: !9)
!21 = !DILocation(line: 3, column: 14, scope: !9)
!22 = !{!23, !24, i64 0}
!23 = !{!"t", !24, i64 0, !24, i64 8, !24, i64 16}
!24 = !{!"long", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 3, column: 22, scope: !9)
!28 = !{!23, !24, i64 16}
!29 = !DILocation(line: 3, column: 16, scope: !9, atomGroup: 1, atomRank: 2)
!30 = !DILocation(line: 3, column: 3, scope: !9, atomGroup: 1, atomRank: 1)

; DWARF:        DW_TAG_inlined_subroutine
; DWARF-NEXT:     DW_AT_name      ("foo")
; DWARF-NEXT:     DW_AT_type
; DWARF-SAME:     "long"
; DWARF-NEXT:     DW_AT_artificial        (true)
; DWARF-NEXT:     DW_AT_specification
; DWARF-SAME:     "foo"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("arg")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "t *"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
