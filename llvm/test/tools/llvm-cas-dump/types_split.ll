; RUN: rm -rf %t.casdb
; RUN: llc -O0 --filetype=obj --cas-backend --cas=%t.casdb --mccas-casid -o %t_DIE.casid %s
; RUN: llvm-cas-dump --cas=%t.casdb --casid-file %t_DIE.casid  --die-refs | FileCheck %s --check-prefix=DWARF-DIE

; DWARF-DIE:        mc:debug_DIE_top_level llvmcas://{{.*}}
; DWARF-DIE-NEXT:   Header = [CD 0 0 0 4 0 0 0 0 0 8]
; DWARF-DIE-NEXT:   CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_compile_unit  AbbrevIdx = 2
; DWARF-DIE:        CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_subprogram    AbbrevIdx = 3
; DWARF-DIE:        CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_structure_type    AbbrevIdx = 6
; DWARF-DIE-NEXT:     DW_AT_calling_convention       DW_FORM_data1              [dedups]   [5]
; DWARF-DIE-NEXT:     DW_AT_name                     DW_FORM_strp_cas           [distinct] [92 1]
; DWARF-DIE-NEXT:     DW_AT_byte_size                DW_FORM_data1              [dedups]   [20]
; DWARF-DIE-NEXT:     DW_AT_decl_file                DW_FORM_data1              [distinct] [1]
; DWARF-DIE-NEXT:     DW_AT_decl_line                DW_FORM_data1              [dedups]   [2]
; DWARF-DIE-NEXT:     DW_TAG_member             AbbrevIdx = 7
; DWARF-DIE-NEXT:       DW_AT_name                     DW_FORM_strp_cas           [distinct] [9B 1]
; DWARF-DIE-NEXT:       DW_AT_type                     DW_FORM_ref4_cas           [distinct] [5B]
; DWARF-DIE-NEXT:       DW_AT_decl_file                DW_FORM_data1              [distinct] [1]
; DWARF-DIE-NEXT:       DW_AT_decl_line                DW_FORM_data1              [dedups]   [4]
; DWARF-DIE-NEXT:       DW_AT_data_member_location     DW_FORM_data1              [dedups]   [0]
; DWARF-DIE-NEXT:       DW_AT_accessibility            DW_FORM_data1              [dedups]   [3]
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 7
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 7
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 7
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 8
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 8
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 8
; DWARF-DIE:          DW_TAG_member             AbbrevIdx = 8


target triple = "arm64-apple-macosx13.0.0"

%struct.MyStruct = type { i32, i32, i32, i32, i32, i32, i32, i32 }

define i32 @foo(ptr %S) !dbg !9 {
entry:
  call void @llvm.dbg.declare(metadata ptr %S, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load i32, ptr %S, align 4, !dbg !26
  ret i32 %0
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (git@github.com:apple/llvm-project.git)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "test.cpp", directory: "/Users/piovezan/workspace/apple-llvm-project/worktrees/cas_main")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"clang version 16.0.0 (git@github.com:apple/llvm-project.git)"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, line: 2, size: 256, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS8MyStruct")
!14 = !{!15, !16, !17, !18, !19, !20, !21, !22}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 4, baseType: !12, size: 32, flags: DIFlagPrivate)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !1, line: 4, baseType: !12, size: 32, offset: 32, flags: DIFlagPrivate)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !1, line: 4, baseType: !12, size: 32, offset: 64, flags: DIFlagPrivate)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !13, file: !1, line: 4, baseType: !12, size: 32, offset: 96, flags: DIFlagPrivate)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !13, file: !1, line: 6, baseType: !12, size: 32, offset: 128)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !13, file: !1, line: 6, baseType: !12, size: 32, offset: 160)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !13, file: !1, line: 6, baseType: !12, size: 32, offset: 192)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "h", scope: !13, file: !1, line: 6, baseType: !12, size: 32, offset: 224)
!23 = !{}
!24 = !DILocalVariable(name: "S", arg: 1, scope: !9, file: !1, line: 9, type: !13)
!25 = !DILocation(line: 9, column: 18, scope: !9)
!26 = !DILocation(line: 9, column: 32, scope: !9)
