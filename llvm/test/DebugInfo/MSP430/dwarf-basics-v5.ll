; RUN: llc -generate-arange-section -minimize-addr-in-v5=Ranges --filetype=obj -o %t < %s
; RUN: llvm-dwarfdump --debug-info -debug-aranges -debug-addr %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t

; This file was based on output of
;
;   clang -target msp430 -S -emit-llvm -gdwarf-5 -Os dwarf-basics-v5.c
;
; for the following dwarf-basics-v5.c
;
;   struct X {
;     void *a;
;   };
;
;   int f(long y, struct X *p)
;   {
;     return 42;
;   }
;

; CHECK: file format elf32-msp430

; CHECK: .debug_info contents:
; CHECK: Compile Unit: length = 0x{{.*}}, format = DWARF32, version = 0x0005, unit_type = DW_UT_compile, abbr_offset = 0x0000, addr_size = 0x02 (next unit at 0x{{.*}})

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_producer    ("clang version 14.0.0 (git@...)")
; CHECK:   DW_AT_language    (DW_LANG_C99)
; CHECK:   DW_AT_name        ("dwarf-basics-v5.c")
; CHECK:   DW_AT_str_offsets_base    (0x00000008)
; CHECK:   DW_AT_stmt_list   (0x{{.*}})
; CHECK:   DW_AT_comp_dir    ("/tmp")
; CHECK:   DW_AT_low_pc      (0x{{.*}})
; CHECK:   DW_AT_high_pc     (0x{{.*}})
; CHECK:   DW_AT_addr_base   (0x00000008)
; CHECK:   DW_AT_loclists_base       (0x0000000c)

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc    (0x{{.*}})
; CHECK:     DW_AT_high_pc   (0x{{.*}})
; CHECK:     DW_AT_frame_base        (DW_OP_reg1 SP)
; CHECK:     DW_AT_call_all_calls    (true)
; CHECK:     DW_AT_name      ("f")
; CHECK:     DW_AT_decl_file ("/tmp{{[/\\]}}dwarf-basics-v5.c")
; CHECK:     DW_AT_decl_line (5)
; CHECK:     DW_AT_prototyped (true)
; CHECK:     DW_AT_type      (0x{{.*}} "int")
; CHECK:     DW_AT_external  (true)

; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_location        (indexed (0x0) loclist = 0x{{.*}}:
; CHECK:            [0x0000, 0x0004): DW_OP_reg12 R12)
; CHECK:         DW_AT_name    ("y")
; CHECK:         DW_AT_decl_file       ("/tmp{{[/\\]}}dwarf-basics-v5.c")
; CHECK:         DW_AT_decl_line       (5)
; CHECK:         DW_AT_type    (0x{{.*}} "long")

; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_location        (DW_OP_reg14 R14)
; CHECK:         DW_AT_name    ("p")
; CHECK:         DW_AT_decl_file       ("/tmp{{[/\\]}}dwarf-basics-v5.c")
; CHECK:         DW_AT_decl_line       (5)
; CHECK:         DW_AT_type    (0x{{.*}} "X *")

; CHECK:       NULL

; CHECK:     DW_TAG_base_type
; CHECK:       DW_AT_name      ("int")
; CHECK:       DW_AT_encoding  (DW_ATE_signed)
; CHECK:       DW_AT_byte_size (0x02)

; CHECK:     DW_TAG_base_type
; CHECK:       DW_AT_name      ("long")
; CHECK:       DW_AT_encoding  (DW_ATE_signed)
; CHECK:       DW_AT_byte_size (0x04)

; CHECK:     DW_TAG_pointer_type
; CHECK:       DW_AT_type      (0x{{.*}} "X")

; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name      ("X")
; CHECK:       DW_AT_byte_size (0x02)
; CHECK:       DW_AT_decl_file ("/tmp{{[/\\]}}dwarf-basics-v5.c")
; CHECK:       DW_AT_decl_line (1)

; CHECK:       DW_TAG_member
; CHECK:         DW_AT_name    ("a")
; CHECK:         DW_AT_type    (0x{{.*}} "void *")
; CHECK:         DW_AT_decl_file       ("/tmp{{[/\\]}}dwarf-basics-v5.c")
; CHECK:         DW_AT_decl_line       (2)
; CHECK:         DW_AT_data_member_location    (0x00)

; CHECK:       NULL

; CHECK:     DW_TAG_pointer_type

; CHECK:     NULL

; CHECK:      .debug_aranges contents:
; CHECK-NEXT: Address Range Header: length = 0x{{.*}}, format = DWARF32, version = 0x0002, cu_offset = 0x00000000, addr_size = 0x02, seg_size = 0x00
; CHECK-NEXT: [0x0000, 0x0006)

; CHECK:      .debug_addr contents:
; CHECK-NEXT: Address table header: length = 0x{{.*}}, format = DWARF32, version = 0x0005, addr_size = 0x02, seg_size = 0x00
; CHECK-NEXT: Addrs: [
; CHECK-NEXT: 0x0000
; CHECK-NEXT: ]

; ModuleID = 'dwarf-basics-v5.c'
source_filename = "dwarf-basics-v5.c"
target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

%struct.X = type { i8* }

; Function Attrs: mustprogress nofree norecurse nosync nounwind optsize readnone willreturn
define dso_local i16 @f(i32 noundef %y, %struct.X* nocapture noundef readnone %p) local_unnamed_addr #0 !dbg !6 {
entry:
  call void @llvm.dbg.value(metadata i32 %y, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata %struct.X* %p, metadata !18, metadata !DIExpression()), !dbg !19
  ret i16 42, !dbg !20
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind optsize readnone willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (git@...)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dwarf-basics-v5.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "ead340d457001e2ce340630cfa3a9cb8")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{!"clang version 14.0.0 (git@...)"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 5, type: !7, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11}
!9 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!10 = !DIBasicType(name: "long", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 16)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !1, line: 1, size: 16, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 2, baseType: !15, size: 16)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 16)
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "y", arg: 1, scope: !6, file: !1, line: 5, type: !10)
!18 = !DILocalVariable(name: "p", arg: 2, scope: !6, file: !1, line: 5, type: !11)
!19 = !DILocation(line: 0, scope: !6)
!20 = !DILocation(line: 7, column: 3, scope: !6)
