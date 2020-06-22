; RUN: llc --filetype=obj -o %t < %s
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump --verify %t

; This file was based on output of
;
;   clang -target msp430 -S -emit-llvm -gdwarf-3 -Os dwarf-basics.c
;
; for the following dwarf-basics.c
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
; CHECK: Compile Unit: length = 0x{{.*}}, format = DWARF32, version = 0x0003, abbr_offset = 0x0000, addr_size = 0x02 (next unit at 0x{{.*}})

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_producer    ("clang version 11.0.0 (git@...)")
; CHECK:   DW_AT_language    (DW_LANG_C99)
; CHECK:   DW_AT_name        ("dwarf-basics.c")
; CHECK:   DW_AT_stmt_list   (0x{{.*}})
; CHECK:   DW_AT_comp_dir    ("/tmp")
; CHECK:   DW_AT_low_pc      (0x{{.*}})
; CHECK:   DW_AT_high_pc     (0x{{.*}})

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_low_pc    (0x{{.*}})
; CHECK:     DW_AT_high_pc   (0x{{.*}})
; CHECK:     DW_AT_frame_base        (DW_OP_reg1 SPB)
; CHECK:     DW_AT_name      ("f")
; CHECK:     DW_AT_decl_file ("/tmp/dwarf-basics.c")
; CHECK:     DW_AT_decl_line (5)
; CHECK:     DW_AT_prototyped        (0x01)
; CHECK:     DW_AT_type      (0x{{.*}} "int")
; CHECK:     DW_AT_external  (0x01)

; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_location        (0x{{.*}}:
; CHECK:            [0x0000, 0x0004): DW_OP_reg12 R12B)
; CHECK:         DW_AT_name    ("y")
; CHECK:         DW_AT_decl_file       ("/tmp/dwarf-basics.c")
; CHECK:         DW_AT_decl_line       (5)
; CHECK:         DW_AT_type    (0x{{.*}} "long int")

; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_location        (DW_OP_reg14 R14B)
; CHECK:         DW_AT_name    ("p")
; CHECK:         DW_AT_decl_file       ("/tmp/dwarf-basics.c")
; CHECK:         DW_AT_decl_line       (5)
; CHECK:         DW_AT_type    (0x{{.*}} "X*")

; CHECK:       NULL

; CHECK:     DW_TAG_base_type
; CHECK:       DW_AT_name      ("int")
; CHECK:       DW_AT_encoding  (DW_ATE_signed)
; CHECK:       DW_AT_byte_size (0x02)

; CHECK:     DW_TAG_base_type
; CHECK:       DW_AT_name      ("long int")
; CHECK:       DW_AT_encoding  (DW_ATE_signed)
; CHECK:       DW_AT_byte_size (0x04)

; CHECK:     DW_TAG_pointer_type
; CHECK:       DW_AT_type      (0x{{.*}} "X")

; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name      ("X")
; CHECK:       DW_AT_byte_size (0x02)
; CHECK:       DW_AT_decl_file ("/tmp/dwarf-basics.c")
; CHECK:       DW_AT_decl_line (1)

; CHECK:       DW_TAG_member
; CHECK:         DW_AT_name    ("a")
; CHECK:         DW_AT_type    (0x{{.*}} "*")
; CHECK:         DW_AT_decl_file       ("/tmp/dwarf-basics.c")
; CHECK:         DW_AT_decl_line       (2)
; CHECK:         DW_AT_data_member_location    (0x00)

; CHECK:       NULL

; CHECK:     DW_TAG_pointer_type

; CHECK:     NULL


source_filename = "dwarf-basics.c"
target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430"

%struct.X = type { i8* }

define i16 @f(i32 %y, %struct.X* %p) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %y, metadata !18, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata %struct.X* %p, metadata !19, metadata !DIExpression()), !dbg !20
  ret i16 42, !dbg !21
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (git@...)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dwarf-basics.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 3}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 11.0.0 (git@...)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 5, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !12}
!10 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!11 = !DIBasicType(name: "long int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 16)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !1, line: 1, size: 16, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 2, baseType: !16, size: 16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 16)
!17 = !{!18, !19}
!18 = !DILocalVariable(name: "y", arg: 1, scope: !7, file: !1, line: 5, type: !11)
!19 = !DILocalVariable(name: "p", arg: 2, scope: !7, file: !1, line: 5, type: !12)
!20 = !DILocation(line: 0, scope: !7)
!21 = !DILocation(line: 7, column: 3, scope: !7)
