; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;   // clang -O2 -S -emit-llvm -g test1.c -mllvm -disable-changed-func-dbinfo
;   struct t { long a; long b; };
;   long foo(struct t arg) {
;     return arg.a * arg.b;
;   }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @foo(i64 %0, i64 %1) local_unnamed_addr #0 !dbg !10 {
    #dbg_value(i64 %0, !19, !DIExpression(DW_OP_LLVM_fragment, 0, 64), !20)
    #dbg_value(i64 %1, !19, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !20)
  %3 = mul nsw i64 %1, %0, !dbg !21
  ret i64 %3, !dbg !22
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test1.c", directory: "/tmp/home/yhs/tests/sig-change/struct", checksumkind: CSK_MD5, checksum: "bd0d0ce5cc67e004962a79d888a27468")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)"}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18, keyInstructions: true)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 128, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 1, baseType: !13, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 1, baseType: !13, size: 64, offset: 64)
!18 = !{!19}
!19 = !DILocalVariable(name: "arg", arg: 1, scope: !10, file: !1, line: 2, type: !14)
!20 = !DILocation(line: 0, scope: !10)
!21 = !DILocation(line: 3, column: 16, scope: !10, atomGroup: 1, atomRank: 2)
!22 = !DILocation(line: 3, column: 3, scope: !10, atomGroup: 1, atomRank: 1)

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
; DWARF-SAME:       "long"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("arg__1")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "long"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
