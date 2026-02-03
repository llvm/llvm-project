; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo -enable-changed-func-dbinfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;   // clang -S -emit-llvm -O3 -g test.c
;   __attribute__((noinline)) static int callee(const int *p) { return *p + 42; }
;   int caller(void) {
;     int x = 100;
;     return callee(&x);
;   }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 -2147483606, -2147483648) i32 @caller() local_unnamed_addr #0 !dbg !10 {
    #dbg_value(i32 100, !15, !DIExpression(), !16)
  %1 = tail call fastcc i32 @callee(i32 100), !dbg !17
  ret i32 %1, !dbg !18
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define internal fastcc range(i32 -2147483606, -2147483648) i32 @callee(i32 %0) unnamed_addr #1 !dbg !19 {
    #dbg_value(ptr poison, !25, !DIExpression(), !26)
  %2 = add nsw i32 %0, 42, !dbg !27
  ret i32 %2, !dbg !28
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/tests/sig-change/prom", checksumkind: CSK_MD5, checksum: "f42f3fd1477418a2e17b444f656351ff")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 8e5d24efc7dac78e8ba568dfe2fc6cfbe9663b13)"}
!10 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14, keyInstructions: true)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 3, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 4, column: 10, scope: !10, atomGroup: 3, atomRank: 2)
!18 = !DILocation(line: 4, column: 3, scope: !10, atomGroup: 3, atomRank: 1)
!19 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 1, type: !20, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !24, keyInstructions: true)
!20 = !DISubroutineType(cc: DW_CC_nocall, types: !21)
!21 = !{!13, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!24 = !{!25}
!25 = !DILocalVariable(name: "p", arg: 1, scope: !19, file: !1, line: 1, type: !22)
!26 = !DILocation(line: 0, scope: !19)
!27 = !DILocation(line: 1, column: 71, scope: !19, atomGroup: 1, atomRank: 2)
!28 = !DILocation(line: 1, column: 61, scope: !19, atomGroup: 1, atomRank: 1)

; DWARF:        DW_TAG_inlined_subroutine
; DWARF-NEXT:     DW_AT_name      ("callee")
; DWARF-NEXT:     DW_AT_type
; DWARF-SAME:     "int"
; DWARF-NEXT:     DW_AT_artificial        (true)
; DWARF-NEXT:     DW_AT_specification
; DWARF-SAME:     "callee"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("__0")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "int"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
