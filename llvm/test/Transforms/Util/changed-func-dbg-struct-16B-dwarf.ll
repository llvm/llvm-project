; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo -enable-changed-func-dbinfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;   // clang -O2 -S -emit-llvm -g test1.c
;   struct t { long a; long b; };
;   __attribute__((noinline)) static int foo(struct t arg, int a) { return arg.a * arg.b; }
;   int bar(struct t arg) {
;     return foo(arg, 1);
;   }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @bar(i64 %0, i64 %1) local_unnamed_addr #0 !dbg !14 {
    #dbg_value(i64 %0, !24, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !25)
    #dbg_value(i64 %1, !24, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !25)
  %3 = tail call fastcc i32 @foo(i64 %0, i64 %1), !dbg !26
  ret i32 %3, !dbg !27
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc i32 @foo(i64 %0, i64 %1) unnamed_addr #1 !dbg !28 {
    #dbg_value(i64 %0, !32, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !34)
    #dbg_value(i64 %1, !32, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !34)
    #dbg_value(i32 poison, !33, !DIExpression(), !34)
  %3 = mul nsw i64 %1, %0, !dbg !35
  %4 = trunc i64 %3 to i32, !dbg !36
  ret i32 %4, !dbg !37
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!llvm.errno.tbaa = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 2bb68bb783927bdc2b54e64aea1b78ba598a3349)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test1.c", directory: "/home/yhs/tests/sig-change/struct16B", checksumkind: CSK_MD5, checksum: "c01b6b6ca539b790114bf7472cfb761a")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 2bb68bb783927bdc2b54e64aea1b78ba598a3349)"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23, keyInstructions: true)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 128, elements: !19)
!19 = !{!20, !22}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !18, file: !1, line: 1, baseType: !21, size: 64)
!21 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !18, file: !1, line: 1, baseType: !21, size: 64, offset: 64)
!23 = !{!24}
!24 = !DILocalVariable(name: "arg", arg: 1, scope: !14, file: !1, line: 3, type: !18)
!25 = !DILocation(line: 0, scope: !14)
!26 = !DILocation(line: 4, column: 10, scope: !14, atomGroup: 1, atomRank: 2)
!27 = !DILocation(line: 4, column: 3, scope: !14, atomGroup: 1, atomRank: 1)
!28 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !29, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31, keyInstructions: true)
!29 = !DISubroutineType(cc: DW_CC_nocall, types: !30)
!30 = !{!17, !18, !17}
!31 = !{!32, !33}
!32 = !DILocalVariable(name: "arg", arg: 1, scope: !28, file: !1, line: 2, type: !18)
!33 = !DILocalVariable(name: "a", arg: 2, scope: !28, file: !1, line: 2, type: !17)
!34 = !DILocation(line: 0, scope: !28)
!35 = !DILocation(line: 2, column: 78, scope: !28, atomGroup: 1, atomRank: 3)
!36 = !DILocation(line: 2, column: 72, scope: !28, atomGroup: 1, atomRank: 2)
!37 = !DILocation(line: 2, column: 65, scope: !28, atomGroup: 1, atomRank: 1)

; DWARF:        DW_TAG_inlined_subroutine
; DWARF-NEXT:     DW_AT_name      ("foo")
; DWARF-NEXT:     DW_AT_type
; DWARF-SAME:     "int"
; DWARF-NEXT:     DW_AT_artificial        (true)
; DWARF-NEXT:     DW_AT_specification
; DWARF-SAME:     "foo"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("a")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "long"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("b")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "long"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
