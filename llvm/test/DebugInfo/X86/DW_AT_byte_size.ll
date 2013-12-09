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
  call void @llvm.dbg.declare(metadata !{%struct.A** %a.addr}, metadata !16), !dbg !17
  %0 = load %struct.A** %a.addr, align 8, !dbg !18
  %b = getelementptr inbounds %struct.A* %0, i32 0, i32 0, !dbg !18
  %1 = load i32* %b, align 4, !dbg !18
  ret i32 %1, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{i32 786449, metadata !20, i32 4, metadata !"clang version 3.1 (trunk 150996)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !20, metadata !6, metadata !"foo", metadata !"foo", metadata !"_Z3fooP1A", i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%struct.A*)* @_Z3fooP1A, null, null, metadata !14, i32 3} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786434, metadata !20, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786445, metadata !20, metadata !11, metadata !"b", i32 1, i64 32, i64 32, i64 0, i32 0, metadata !9} ; [ DW_TAG_member ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!16 = metadata !{i32 786689, metadata !5, metadata !"a", metadata !6, i32 16777219, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 3, i32 13, metadata !5, null}
!18 = metadata !{i32 4, i32 3, metadata !19, null}
!19 = metadata !{i32 786443, metadata !20, metadata !5, i32 3, i32 16, i32 0} ; [ DW_TAG_lexical_block ]
!20 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
