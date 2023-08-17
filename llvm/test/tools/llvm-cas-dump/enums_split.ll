; RUN: rm -rf %t.casdb
; RUN: llc -O0 --filetype=obj --cas-backend --cas=%t.casdb --mccas-casid -o %t_DIE.casid %s
; RUN: llvm-cas-dump --cas=%t.casdb --casid-file %t_DIE.casid  --die-refs | FileCheck %s --check-prefix=DWARF-DIE

; DWARF-DIE:        mc:debug_DIE_top_level llvmcas://{{.*}}
; DWARF-DIE-NEXT:   Header = [68 0 0 0 4 0 0 0 0 0 8]
; DWARF-DIE-NEXT:   CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_compile_unit  AbbrevIdx = 2
; DWARF-DIE:        CAS Block: llvmcas://{{.*}}
; DWARF-DIE-NEXT:   DW_TAG_enumeration_type     AbbrevIdx = 3
; DWARF-DIE-NEXT:     DW_AT_type                     DW_FORM_ref4_cas           [distinct] [47]
; DWARF-DIE-NEXT:     DW_AT_enum_class               DW_FORM_flag_present       [dedups]   []
; DWARF-DIE-NEXT:     DW_AT_name                     DW_FORM_strp_cas           [distinct] [88 1]
; DWARF-DIE-NEXT:     DW_AT_byte_size                DW_FORM_data1              [dedups]   [4]
; DWARF-DIE-NEXT:     DW_AT_decl_file                DW_FORM_data1              [distinct] [1]
; DWARF-DIE-NEXT:     DW_AT_decl_line                DW_FORM_data1              [dedups]   [3]
; DWARF-DIE-NEXT:     DW_TAG_enumerator         AbbrevIdx = 4
; DWARF-DIE-NEXT:       DW_AT_name                     DW_FORM_strp_cas           [distinct] [96 1]
; DWARF-DIE-NEXT:       DW_AT_const_value              DW_FORM_sdata              [dedups]   [0]
; DWARF-DIE-NEXT:     DW_TAG_enumerator         AbbrevIdx = 4
; DWARF-DIE-NEXT:       DW_AT_name                     DW_FORM_strp_cas           [distinct] [9B 1]
; DWARF-DIE-NEXT:       DW_AT_const_value              DW_FORM_sdata              [dedups]   [1]

target triple = "arm64-apple-macosx13.0.0"

; enum class ColorEnum {
;  Blue = 0,
;  Red = 1
; };
; ColorEnum foo(ColorEnum x) { return ColorEnum::Blue;}

define i32 @foo(i32 %x) !dbg !15 {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0 (git@github.com:apple/llvm-project.git)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "test.cpp", directory: "/Users/piovezan/workspace/apple-llvm-project/worktrees/cas_main")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "ColorEnum", file: !1, line: 3, baseType: !4, size: 32, flags: DIFlagEnumClass, elements: !5, identifier: "_ZTS9ColorEnum")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "Blue", value: 0)
!7 = !DIEnumerator(name: "Red", value: 1)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 1}
!13 = !{i32 7, !"frame-pointer", i32 1}
!14 = !{!"clang version 16.0.0 (git@github.com:apple/llvm-project.git)"}
!15 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 8, type: !16, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!3, !3}
!18 = !{}
