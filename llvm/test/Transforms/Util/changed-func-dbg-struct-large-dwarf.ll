; Check the generated DWARF debug info:
; RUN: opt -S -mtriple=x86_64-unknown-unknown -passes=emit-changed-func-debuginfo -enable-changed-func-dbinfo < %s \
; RUN:   | %llc_dwarf -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF
;
; REQUIRES: debug_frame
; REQUIRES: object-emission

; Source code:
;   // clang -O2 -S -emit-llvm -g test2.c
;   struct t { long a; long b; long c; };
;   __attribute__((noinline)) static int foo(struct t arg, int a) { return arg.a * arg.b * arg.c; }
;   int bar(struct t arg) {
;     return foo(arg, 1);
;   }

%struct.t = type { i64, i64, i64 }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local i32 @bar(ptr noundef readonly byval(%struct.t) align 8 captures(none) %0) local_unnamed_addr #0 !dbg !14 {
    #dbg_declare(ptr %0, !25, !DIExpression(), !26)
  %2 = tail call fastcc i32 @foo(ptr noundef nonnull byval(%struct.t) align 8 %0), !dbg !27
  ret i32 %2, !dbg !28
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define internal fastcc i32 @foo(ptr noundef readonly byval(%struct.t) align 8 captures(none) %0) unnamed_addr #1 !dbg !29 {
    #dbg_declare(ptr %0, !33, !DIExpression(), !35)
    #dbg_value(i32 poison, !34, !DIExpression(), !36)
  %2 = load i64, ptr %0, align 8, !dbg !37, !tbaa !38
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8, !dbg !41
  %4 = load i64, ptr %3, align 8, !dbg !41, !tbaa !42
  %5 = mul nsw i64 %4, %2, !dbg !43
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16, !dbg !44
  %7 = load i64, ptr %6, align 8, !dbg !44, !tbaa !45
  %8 = mul nsw i64 %5, %7, !dbg !46
  %9 = trunc i64 %8 to i32, !dbg !47
  ret i32 %9, !dbg !48
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!llvm.errno.tbaa = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:yonghong-song/llvm-project.git 2bb68bb783927bdc2b54e64aea1b78ba598a3349)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test2.c", directory: "/home/yhs/tests/sig-change/struct16B", checksumkind: CSK_MD5, checksum: "70265912cbcea4a3b09ee4c21d26b33e")
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
!14 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !24, keyInstructions: true)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 192, elements: !19)
!19 = !{!20, !22, !23}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !18, file: !1, line: 1, baseType: !21, size: 64)
!21 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !18, file: !1, line: 1, baseType: !21, size: 64, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !18, file: !1, line: 1, baseType: !21, size: 64, offset: 128)
!24 = !{!25}
!25 = !DILocalVariable(name: "arg", arg: 1, scope: !14, file: !1, line: 3, type: !18)
!26 = !DILocation(line: 3, column: 18, scope: !14)
!27 = !DILocation(line: 4, column: 10, scope: !14, atomGroup: 1, atomRank: 2)
!28 = !DILocation(line: 4, column: 3, scope: !14, atomGroup: 1, atomRank: 1)
!29 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !30, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32, keyInstructions: true)
!30 = !DISubroutineType(cc: DW_CC_nocall, types: !31)
!31 = !{!17, !18, !17}
!32 = !{!33, !34}
!33 = !DILocalVariable(name: "arg", arg: 1, scope: !29, file: !1, line: 2, type: !18)
!34 = !DILocalVariable(name: "a", arg: 2, scope: !29, file: !1, line: 2, type: !17)
!35 = !DILocation(line: 2, column: 51, scope: !29)
!36 = !DILocation(line: 0, scope: !29)
!37 = !DILocation(line: 2, column: 76, scope: !29)
!38 = !{!39, !40, i64 0}
!39 = !{!"t", !40, i64 0, !40, i64 8, !40, i64 16}
!40 = !{!"long", !12, i64 0}
!41 = !DILocation(line: 2, column: 84, scope: !29)
!42 = !{!39, !40, i64 8}
!43 = !DILocation(line: 2, column: 78, scope: !29)
!44 = !DILocation(line: 2, column: 92, scope: !29)
!45 = !{!39, !40, i64 16}
!46 = !DILocation(line: 2, column: 86, scope: !29, atomGroup: 1, atomRank: 3)
!47 = !DILocation(line: 2, column: 72, scope: !29, atomGroup: 1, atomRank: 2)
!48 = !DILocation(line: 2, column: 65, scope: !29, atomGroup: 1, atomRank: 1)

; DWARF:        DW_TAG_inlined_subroutine
; DWARF-NEXT:     DW_AT_name      ("foo")
; DWARF-NEXT:     DW_AT_type
; DWARF-SAME:     "int"
; DWARF-NEXT:     DW_AT_artificial        (true)
; DWARF-NEXT:     DW_AT_specification
; DWARF-SAME:     "foo"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     DW_TAG_formal_parameter
; DWARF-NEXT:       DW_AT_name    ("arg")
; DWARF-NEXT:       DW_AT_type
; DWARF-SAME:       "t"
; DWARF-NEXT: {{^$}}
; DWARF-NEXT:     NULL
