; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; test that the DW_AT_specification is a back edge in the file.

; Skip the definition of zed(foo*)
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_class_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_MIPS_linkage_name {{.*}} "_ZN3foo3barEv"
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_specification {{.*}} "_ZN3foo3barEv"

%struct.foo = type { i8 }

define void @_Z3zedP3foo(%struct.foo* %x) uwtable {
entry:
  %x.addr = alloca %struct.foo*, align 8
  store %struct.foo* %x, %struct.foo** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %x.addr, metadata !23, metadata !{!"0x102"}), !dbg !24
  %0 = load %struct.foo** %x.addr, align 8, !dbg !25
  call void @_ZN3foo3barEv(%struct.foo* %0), !dbg !25
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3foo3barEv(%struct.foo* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !28, metadata !{!"0x102"}), !dbg !29
  %this1 = load %struct.foo** %this.addr
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = !{!"0x11\004\00clang version 3.0 ()\000\00\000\00\000", !32, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5, !20}
!5 = !{!"0x2e\00zed\00zed\00_Z3zedP3foo\004\000\001\000\006\00256\000\004", !6, !6, !7, null, void (%struct.foo*)* @_Z3zedP3foo, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [zed]
!6 = !{!"0x29", !32} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9}
!9 = !{!"0xf\00\000\0064\0064\000\000", null, null, !10} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x2\00foo\001\008\008\000\000\000", !32, null, null, !11, null, null, null} ; [ DW_TAG_class_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!11 = !{!12}
!12 = !{!"0x2e\00bar\00bar\00_ZN3foo3barEv\002\000\000\000\006\00256\000\002", !6, !10, !13, null, null, null, i32 0, !16} ; [ DW_TAG_subprogram ]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{null, !15}
!15 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !10} ; [ DW_TAG_pointer_type ]
!16 = !{!17}
!17 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!18 = !{!19}
!19 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!20 = !{!"0x2e\00bar\00bar\00_ZN3foo3barEv\002\000\001\000\006\00256\000\002", !6, null, !13, null, void (%struct.foo*)* @_ZN3foo3barEv, null, !12, null} ; [ DW_TAG_subprogram ] [line 2] [def] [bar]
!23 = !{!"0x101\00x\0016777220\000", !5, !6, !9} ; [ DW_TAG_arg_variable ]
!24 = !MDLocation(line: 4, column: 15, scope: !5)
!25 = !MDLocation(line: 4, column: 20, scope: !26)
!26 = !{!"0xb\004\0018\000", !6, !5} ; [ DW_TAG_lexical_block ]
!27 = !MDLocation(line: 4, column: 30, scope: !26)
!28 = !{!"0x101\00this\0016777218\0064", !20, !6, !15} ; [ DW_TAG_arg_variable ]
!29 = !MDLocation(line: 2, column: 8, scope: !20)
!30 = !MDLocation(line: 2, column: 15, scope: !31)
!31 = !{!"0xb\002\0014\001", !6, !20} ; [ DW_TAG_lexical_block ]
!32 = !{!"/home/espindola/llvm/test.cc", !"/home/espindola/tmpfs/build"}
!33 = !{i32 1, !"Debug Info Version", i32 2}
