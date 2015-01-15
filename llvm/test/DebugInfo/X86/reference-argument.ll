; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=asm %s -o - | FileCheck %s
; ModuleID = 'aggregate-indirect-arg.cpp'
; extracted from debuginfo-tests/aggregate-indirect-arg.cpp

; v should not be a pointer.
; CHECK: ##DEBUG_VALUE: foo:v <- RSI
; rdar://problem/13658587

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%class.SVal = type { i8*, i32 }
%class.A = type { i8 }

declare void @_Z3barR4SVal(%class.SVal* %v)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare i32 @main()
; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1A3fooE4SVal(%class.A* %this, %class.SVal* %v) nounwind ssp uwtable align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !59, metadata !{!"0x102"}), !dbg !61
  call void @llvm.dbg.declare(metadata %class.SVal* %v, metadata !62, metadata !{!"0x102"}), !dbg !61
  %this1 = load %class.A** %this.addr
  call void @_Z3barR4SVal(%class.SVal* %v), !dbg !61
  ret void, !dbg !61
}
declare void @_ZN4SValD1Ev(%class.SVal* %this)
declare void @_ZN4SValD2Ev(%class.SVal* %this)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !68}

!0 = !{!"0x11\004\00clang version 3.4 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [aggregate-indirect-arg.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"aggregate-indirect-arg.cpp", !""}
!2 = !{}
!3 = !{!4, !29, !33, !34, !35}
!4 = !{!"0x2e\00bar\00bar\00_Z3barR4SVal\0019\000\001\000\006\00256\000\0019", !1, !5, !6, null, void (%class.SVal*)* @_Z3barR4SVal, null, null, !2} ; [ DW_TAG_subprogram ] [line 19] [def] [bar]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [aggregate-indirect-arg.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0x10\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!9 = !{!"0x2\00SVal\0012\00128\0064\000\000\000", !1, null, null, !10, null, null, null} ; [ DW_TAG_class_type ] [SVal] [line 12, size 128, align 64, offset 0] [def] [from ]
!10 = !{!11, !14, !16, !21, !23}
!11 = !{!"0xd\00Data\0015\0064\0064\000\000", !1, !9, !12} ; [ DW_TAG_member ] [Data] [line 15, size 64, align 64, offset 0] [from ]
!12 = !{!"0xf\00\000\0064\0064\000\000", null, null, !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = !{!"0x26\00\000\000\000\000\000", null, null, null} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!"0xd\00Kind\0016\0032\0032\0064\000", !1, !9, !15} ; [ DW_TAG_member ] [Kind] [line 16, size 32, align 32, offset 64] [from unsigned int]
!15 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!16 = !{!"0x2e\00~SVal\00~SVal\00\0014\000\000\000\006\00256\000\0014", !1, !9, !17, null, null, null, i32 0, !20} ; [ DW_TAG_subprogram ] [line 14] [~SVal]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19}
!19 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from SVal]
!20 = !{i32 786468}
!21 = !{!"0x2e\00SVal\00SVal\00\0012\000\000\000\006\00320\000\0012", !1, !9, !17, null, null, null, i32 0, !22} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!22 = !{i32 786468}
!23 = !{!"0x2e\00SVal\00SVal\00\0012\000\000\000\006\00320\000\0012", !1, !9, !24, null, null, null, i32 0, !28} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!24 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = !{null, !19, !26}
!26 = !{!"0x10\00\000\000\000\000\000", null, null, !27} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{!"0x26\00\000\000\000\000\000", null, null, !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!28 = !{i32 786468}
!29 = !{!"0x2e\00main\00main\00\0025\000\001\000\006\00256\000\0025", !1, !5, !30, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 25] [def] [main]
!30 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !31, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = !{!32}
!32 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!33 = !{!"0x2e\00~SVal\00~SVal\00_ZN4SValD1Ev\0014\000\001\000\006\00256\000\0014", !1, null, !17, null, void (%class.SVal*)* @_ZN4SValD1Ev, null, !16, !2} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!34 = !{!"0x2e\00~SVal\00~SVal\00_ZN4SValD2Ev\0014\000\001\000\006\00256\000\0014", !1, null, !17, null, void (%class.SVal*)* @_ZN4SValD2Ev, null, !16, !2} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!35 = !{!"0x2e\00foo\00foo\00_ZN1A3fooE4SVal\0022\000\001\000\006\00256\000\0022", !1, null, !36, null, void (%class.A*, %class.SVal*)* @_ZN1A3fooE4SVal, null, !41, !2} ; [ DW_TAG_subprogram ] [line 22] [def] [foo]
!36 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !37, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = !{null, !38, !9}
!38 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!39 = !{!"0x2\00A\0020\008\008\000\000\000", !1, null, null, !40, null, null, null} ; [ DW_TAG_class_type ] [A] [line 20, size 8, align 8, offset 0] [def] [from ]
!40 = !{!41, !43}
!41 = !{!"0x2e\00foo\00foo\00_ZN1A3fooE4SVal\0022\000\000\000\006\00256\000\0022", !1, !39, !36, null, null, null, i32 0, !42} ; [ DW_TAG_subprogram ] [line 22] [foo]
!42 = !{i32 786468}
!43 = !{!"0x2e\00A\00A\00\0020\000\000\000\006\00320\000\0020", !1, !39, !44, null, null, null, i32 0, !46} ; [ DW_TAG_subprogram ] [line 20] [A]
!44 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !45, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = !{null, !38}
!46 = !{i32 786468}
!47 = !{i32 2, !"Dwarf Version", i32 3}
!48 = !{!"0x101\00v\0016777235\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [v] [line 19]
!49 = !MDLocation(line: 19, scope: !4)
!50 = !{!"0x100\00v\0026\000", !29, !5, !9} ; [ DW_TAG_auto_variable ] [v] [line 26]
!51 = !MDLocation(line: 26, scope: !29)
!52 = !MDLocation(line: 27, scope: !29)
!53 = !MDLocation(line: 28, scope: !29)
!54 = !{!"0x100\00a\0029\000", !29, !5, !39} ; [ DW_TAG_auto_variable ] [a] [line 29]
!55 = !MDLocation(line: 29, scope: !29)
!56 = !MDLocation(line: 30, scope: !29)
!57 = !MDLocation(line: 31, scope: !29)
!58 = !MDLocation(line: 32, scope: !29)
!59 = !{!"0x101\00this\0016777238\001088", !35, !5, !60} ; [ DW_TAG_arg_variable ] [this] [line 22]
!60 = !{!"0xf\00\000\0064\0064\000\000", null, null, !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!61 = !MDLocation(line: 22, scope: !35)
!62 = !{!"0x101\00v\0033554454\008192", !35, !5, !9} ; [ DW_TAG_arg_variable ] [v] [line 22]
!63 = !{!"0x101\00this\0016777230\001088", !33, !5, !64} ; [ DW_TAG_arg_variable ] [this] [line 14]
!64 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from SVal]
!65 = !MDLocation(line: 14, scope: !33)
!66 = !{!"0x101\00this\0016777230\001088", !34, !5, !64} ; [ DW_TAG_arg_variable ] [this] [line 14]
!67 = !MDLocation(line: 14, scope: !34)
!68 = !{i32 1, !"Debug Info Version", i32 2}
