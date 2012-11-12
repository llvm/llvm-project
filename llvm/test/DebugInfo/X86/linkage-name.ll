; RUN: llc -mtriple=x86_64-macosx -darwin-gdb-compat=Disable %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

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
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !21), !dbg !23
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %b.addr}, metadata !24), !dbg !25
  %this1 = load %class.A** %this.addr
  %0 = load i32* %b.addr, align 4, !dbg !26
  ret i32 %0, !dbg !26
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !"foo.cpp", metadata !"/Users/echristo", metadata !"clang version 3.1 (trunk 152691) (llvm/trunk 152692)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !18} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, null, metadata !"a", metadata !"a", metadata !"_ZN1A1aEi", metadata !6, i32 5, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%class.A*, i32)* @_ZN1A1aEi, null, metadata !13, metadata !16} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !"foo.cpp", metadata !"/Users/echristo", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9, metadata !10, metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{i32 786447, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786434, null, metadata !"A", metadata !6, i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !12, i32 0, null, null} ; [ DW_TAG_class_type ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786478, i32 0, metadata !11, metadata !"a", metadata !"a", metadata !"_ZN1A1aEi", metadata !6, i32 2, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 257, i1 false, null, null, i32 0, metadata !14} ; [ DW_TAG_subprogram ]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !19}
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 9, metadata !11, i32 0, i32 1, %class.A* @a} ; [ DW_TAG_variable ]
!21 = metadata !{i32 786689, metadata !5, metadata !"this", metadata !6, i32 16777221, metadata !22, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 786447, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!23 = metadata !{i32 5, i32 8, metadata !5, null}
!24 = metadata !{i32 786689, metadata !5, metadata !"b", metadata !6, i32 33554437, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 5, i32 14, metadata !5, null}
!26 = metadata !{i32 6, i32 4, metadata !27, null}
!27 = metadata !{i32 786443, metadata !5, i32 5, i32 17, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
