; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; CHECK: 0x0000005c:     DW_TAG_subprogram [5]
; CHECK: 0x0000007c:     DW_AT_specification [DW_FORM_ref4]      (cu + 0x005c => {0x0000005c})

%struct.foo = type { i8 }

define void @_Z3zedP3foo(%struct.foo* %x) uwtable {
entry:
  %x.addr = alloca %struct.foo*, align 8
  store %struct.foo* %x, %struct.foo** %x.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.foo** %x.addr}, metadata !23), !dbg !24
  %0 = load %struct.foo** %x.addr, align 8, !dbg !25
  call void @_ZN3foo3barEv(%struct.foo* %0), !dbg !25
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3foo3barEv(%struct.foo* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr}, metadata !28), !dbg !29
  %this1 = load %struct.foo** %this.addr
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"/home/espindola/llvm/test.cc", metadata !"/home/espindola/tmpfs/build", metadata !"clang version 3.0 ()", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5, metadata !20}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"zed", metadata !"zed", metadata !"_Z3zedP3foo", metadata !6, i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.foo*)* @_Z3zedP3foo, null, null, metadata !18} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"/home/espindola/llvm/test.cc", metadata !"/home/espindola/tmpfs/build", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 720911, null, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 720898, null, metadata !"foo", metadata !6, i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !11, i32 0, null, null} ; [ DW_TAG_class_type ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 720942, i32 0, metadata !10, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !6, i32 2, metadata !13, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !16} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !14, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!14 = metadata !{null, metadata !15}
!15 = metadata !{i32 720911, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !10} ; [ DW_TAG_pointer_type ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!20 = metadata !{i32 720942, i32 0, null, metadata !"bar", metadata !"bar", metadata !"_ZN3foo3barEv", metadata !6, i32 2, metadata !13, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, void (%struct.foo*)* @_ZN3foo3barEv, null, metadata !12, metadata !21} ; [ DW_TAG_subprogram ]
!21 = metadata !{metadata !22}
!22 = metadata !{i32 720932}                      ; [ DW_TAG_base_type ]
!23 = metadata !{i32 721153, metadata !5, metadata !"x", metadata !6, i32 16777220, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 4, i32 15, metadata !5, null}
!25 = metadata !{i32 4, i32 20, metadata !26, null}
!26 = metadata !{i32 720907, metadata !5, i32 4, i32 18, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 4, i32 30, metadata !26, null}
!28 = metadata !{i32 721153, metadata !20, metadata !"this", metadata !6, i32 16777218, metadata !15, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!29 = metadata !{i32 2, i32 8, metadata !20, null}
!30 = metadata !{i32 2, i32 15, metadata !31, null}
!31 = metadata !{i32 720907, metadata !20, i32 2, i32 14, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
