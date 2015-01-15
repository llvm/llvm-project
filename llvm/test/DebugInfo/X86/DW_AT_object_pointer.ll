; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_formal_parameter [
; CHECK-NOT: ""
; CHECK: DW_TAG
; CHECK: DW_TAG_class_type
; CHECK: DW_AT_object_pointer [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]*}} => {[[PARAM:0x[0-9a-f]*]]})
; CHECK: [[PARAM]]:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x{{[0-9a-f]*}}] = "this")

%class.A = type { i32 }

define i32 @_Z3fooi(i32) nounwind uwtable ssp {
entry:
  %.addr = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !36, metadata !{!"0x102"}), !dbg !35
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !21, metadata !{!"0x102"}), !dbg !23
  call void @_ZN1AC1Ev(%class.A* %a), !dbg !24
  %m_a = getelementptr inbounds %class.A* %a, i32 0, i32 0, !dbg !25
  %1 = load i32* %m_a, align 4, !dbg !25
  ret i32 %1, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !26, metadata !{!"0x102"}), !dbg !28
  %this1 = load %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1), !dbg !29
  ret void, !dbg !29
}

define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !30, metadata !{!"0x102"}), !dbg !31
  %this1 = load %class.A** %this.addr
  %m_a = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !32
  store i32 0, i32* %m_a, align 4, !dbg !32
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38}

!0 = !{!"0x11\004\00clang version 3.2 (trunk 163586) (llvm/trunk 163570)\000\00\000\00\000", !37, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ] [/Users/echristo/debug-tests/bar.cpp] [DW_LANG_C_plus_plus]
!1 = !{}
!3 = !{!5, !10, !20}
!5 = !{!"0x2e\00foo\00foo\00_Z3fooi\007\000\001\000\006\00256\000\007", !6, !6, !7, null, i32 (i32)* @_Z3fooi, null, null, !1} ; [ DW_TAG_subprogram ] [line 7] [def] [foo]
!6 = !{!"0x29", !37} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x2e\00A\00A\00_ZN1AC1Ev\003\000\001\000\006\00256\000\003", !6, null, !11, null, void (%class.A*)* @_ZN1AC1Ev, null, !17, !1} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{null, !13}
!13 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!14 = !{!"0x2\00A\001\0032\0032\000\000\000", !37, null, null, !15, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!15 = !{!16, !17}
!16 = !{!"0xd\00m_a\004\0032\0032\000\000", !37, !14, !9} ; [ DW_TAG_member ] [m_a] [line 4, size 32, align 32, offset 0] [from int]
!17 = !{!"0x2e\00A\00A\00\003\000\000\000\006\00256\000\003", !6, !14, !11, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ] [line 3] [A]
!18 = !{!19}
!19 = !{!"0x24"}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = !{!"0x2e\00A\00A\00_ZN1AC2Ev\003\000\001\000\006\00256\000\003", !6, null, !11, null, void (%class.A*)* @_ZN1AC2Ev, null, !17, !1} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!21 = !{!"0x100\00a\008\000", !22, !6, !14} ; [ DW_TAG_auto_variable ] [a] [line 8]
!22 = !{!"0xb\007\0011\000", !6, !5} ; [ DW_TAG_lexical_block ] [/Users/echristo/debug-tests/bar.cpp]
!23 = !MDLocation(line: 8, column: 5, scope: !22)
!24 = !MDLocation(line: 8, column: 6, scope: !22)
!25 = !MDLocation(line: 9, column: 3, scope: !22)
!26 = !{!"0x101\00this\0016777219\001088", !10, !6, !27} ; [ DW_TAG_arg_variable ] [this] [line 3]
!27 = !{!"0xf\00\000\0064\0064\000\000", null, null, !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!28 = !MDLocation(line: 3, column: 3, scope: !10)
!29 = !MDLocation(line: 3, column: 18, scope: !10)
!30 = !{!"0x101\00this\0016777219\001088", !20, !6, !27} ; [ DW_TAG_arg_variable ] [this] [line 3]
!31 = !MDLocation(line: 3, column: 3, scope: !20)
!32 = !MDLocation(line: 3, column: 9, scope: !33)
!33 = !{!"0xb\003\007\001", !6, !20} ; [ DW_TAG_lexical_block ] [/Users/echristo/debug-tests/bar.cpp]
!34 = !MDLocation(line: 3, column: 18, scope: !33)
!35 = !MDLocation(line: 7, scope: !5)
!36 = !{!"0x101\00\0016777223\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [line 7]
!37 = !{!"bar.cpp", !"/Users/echristo/debug-tests"}
!38 = !{i32 1, !"Debug Info Version", i32 2}
