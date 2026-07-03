; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefix=SECTIONS %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck --check-prefix=SECTIONS %s

; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_location
; CHECK-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +7, DW_OP_stack_value
; CHECK-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +8, DW_OP_stack_value
; CHECK-NEXT: DW_AT_name	("i")

; SECTIONS:      DW_TAG_variable
; SECTIONS-NEXT: DW_AT_location
; SECTIONS-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +7, DW_OP_stack_value
; SECTIONS-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +8, DW_OP_stack_value
; SECTIONS-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +8, DW_OP_stack_value
; SECTIONS-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_consts +8, DW_OP_stack_value
; SECTIONS-NEXT: DW_AT_name	("i")

; Source to generate the IR below:
; void f1();
; extern bool b;
; void test() {
;     // i is not a const throughout the whole scope and should
;     // not use DW_AT_const_value
;     int i = 7;
;     f1();
;     i = 8;
;     if (b)
;       f1();
; }
; $ clang++ -S loclist_section.cc -O2 -g  -emit-llvm

@b = external local_unnamed_addr global i8, align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z4testv() local_unnamed_addr #0 !dbg !10 {
entry:
    #dbg_value(i32 7, !14, !DIExpression(), !16)
  tail call void @_Z2f1v(), !dbg !17
    #dbg_value(i32 8, !14, !DIExpression(), !16)
  %0 = load i8, ptr @b, align 1, !dbg !18, !tbaa !20, !range !24, !noundef !25
  %loadedv = trunc nuw i8 %0 to i1, !dbg !18
  br i1 %loadedv, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
  tail call void @_Z2f1v(), !dbg !27
  br label %if.end, !dbg !27

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !28
}

declare !dbg !29 void @_Z2f1v() local_unnamed_addr #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git (git@github.com:)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "loclist_section.cc", directory: "Examples/debug_loc", checksumkind: CSK_MD5, checksum: "67769a94389681c8a6da481e2f358abb")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 20.0.0git (git@github.com:.../llvm-project.git 7c3256280a78b0505ae4d43985c4d3239451a151)"}
!10 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{!14}
!14 = !DILocalVariable(name: "i", scope: !10, file: !1, line: 6, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 7, column: 5, scope: !10)
!18 = !DILocation(line: 9, column: 9, scope: !19)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 9)
!20 = !{!21, !21, i64 0}
!21 = !{!"bool", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !{i8 0, i8 2}
!25 = !{}
!26 = !DILocation(line: 9, column: 9, scope: !10)
!27 = !DILocation(line: 10, column: 7, scope: !19)
!28 = !DILocation(line: 11, column: 1, scope: !10)
!29 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
