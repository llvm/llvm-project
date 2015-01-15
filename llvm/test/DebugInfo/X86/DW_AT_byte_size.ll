; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Checks that we don't emit a size for a pointer type.
; CHECK: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-NOT: DW_AT_byte_size
; CHECK: DW_TAG
; CHECK: .debug_info contents

%struct.A = type { i32 }

define i32 @_Z3fooP1A(%struct.A* %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca %struct.A*, align 8
  store %struct.A* %a, %struct.A** %a.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %a.addr, metadata !16, metadata !{!"0x102"}), !dbg !17
  %0 = load %struct.A** %a.addr, align 8, !dbg !18
  %b = getelementptr inbounds %struct.A* %0, i32 0, i32 0, !dbg !18
  %1 = load i32* %b, align 4, !dbg !18
  ret i32 %1, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = !{!"0x11\004\00clang version 3.1 (trunk 150996)\000\00\000\00\000", !20, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00_Z3fooP1A\003\000\001\000\006\00256\000\003", !20, !6, !7, null, i32 (%struct.A*)* @_Z3fooP1A, null, null, null} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !20} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !10}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x2\00A\001\0032\0032\000\000\000", !20, null, null, !12, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!12 = !{!13}
!13 = !{!"0xd\00b\001\0032\0032\000\000", !20, !11, !9} ; [ DW_TAG_member ]
!16 = !{!"0x101\00a\0016777219\000", !5, !6, !10} ; [ DW_TAG_arg_variable ]
!17 = !MDLocation(line: 3, column: 13, scope: !5)
!18 = !MDLocation(line: 4, column: 3, scope: !19)
!19 = !{!"0xb\003\0016\000", !20, !5} ; [ DW_TAG_lexical_block ]
!20 = !{!"foo.cpp", !"/Users/echristo"}
!21 = !{i32 1, !"Debug Info Version", i32 2}
