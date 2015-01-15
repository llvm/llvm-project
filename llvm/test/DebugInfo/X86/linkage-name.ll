; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_subprogram [9] *
; CHECK-NOT: DW_AT_MIPS_linkage_name
; CHECK: DW_AT_specification

%class.A = type { i8 }

@a = global %class.A zeroinitializer, align 1

define i32 @_ZN1A1aEi(%class.A* %this, i32 %b) nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %b.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !21, metadata !{!"0x102"}), !dbg !23
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !24, metadata !{!"0x102"}), !dbg !25
  %this1 = load %class.A** %this.addr
  %0 = load i32* %b.addr, align 4, !dbg !26
  ret i32 %0, !dbg !26
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = !{!"0x11\004\00clang version 3.1 (trunk 152691) (llvm/trunk 152692)\000\00\000\00\000", !28, !1, !1, !3, !18,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00a\00a\00_ZN1A1aEi\005\000\001\000\006\00256\000\005", !6, null, !7, null, i32 (%class.A*, i32)* @_ZN1A1aEi, null, !13, null} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !28} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x2\00A\001\008\008\000\000\000", !28, null, null, !12, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!12 = !{!13}
!13 = !{!"0x2e\00a\00a\00_ZN1A1aEi\002\000\000\000\006\00257\000\000", !6, !11, !7, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!18 = !{!20}
!20 = !{!"0x34\00a\00a\00\009\000\001", null, !6, !11, %class.A* @a, null} ; [ DW_TAG_variable ]
!21 = !{!"0x101\00this\0016777221\0064", !5, !6, !22} ; [ DW_TAG_arg_variable ]
!22 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!23 = !MDLocation(line: 5, column: 8, scope: !5)
!24 = !{!"0x101\00b\0033554437\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!25 = !MDLocation(line: 5, column: 14, scope: !5)
!26 = !MDLocation(line: 6, column: 4, scope: !27)
!27 = !{!"0xb\005\0017\000", !6, !5} ; [ DW_TAG_lexical_block ]
!28 = !{!"foo.cpp", !"/Users/echristo"}
!29 = !{i32 1, !"Debug Info Version", i32 2}
