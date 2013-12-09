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
declare void @llvm.dbg.declare(metadata, metadata) #1
declare i32 @main()
; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1A3fooE4SVal(%class.A* %this, %class.SVal* %v) nounwind ssp uwtable align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !59), !dbg !61
  call void @llvm.dbg.declare(metadata !{%class.SVal* %v}, metadata !62), !dbg !61
  %this1 = load %class.A** %this.addr
  call void @_Z3barR4SVal(%class.SVal* %v), !dbg !61
  ret void, !dbg !61
}
declare void @_ZN4SValD1Ev(%class.SVal* %this)
declare void @_ZN4SValD2Ev(%class.SVal* %this)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !68}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [aggregate-indirect-arg.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"aggregate-indirect-arg.cpp", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !29, metadata !33, metadata !34, metadata !35}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"_Z3barR4SVal", i32 19, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.SVal*)* @_Z3barR4SVal, null, null, metadata !2, i32 19} ; [ DW_TAG_subprogram ] [line 19] [def] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [aggregate-indirect-arg.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!9 = metadata !{i32 786434, metadata !1, null, metadata !"SVal", i32 12, i64 128, i64 64, i32 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_class_type ] [SVal] [line 12, size 128, align 64, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !14, metadata !16, metadata !21, metadata !23}
!11 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"Data", i32 15, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ] [Data] [line 15, size 64, align 64, offset 0] [from ]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"Kind", i32 16, i64 32, i64 32, i64 64, i32 0, metadata !15} ; [ DW_TAG_member ] [Kind] [line 16, size 32, align 32, offset 64] [from unsigned int]
!15 = metadata !{i32 786468, null, null, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!16 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"~SVal", metadata !"~SVal", metadata !"", i32 14, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !20, i32 14} ; [ DW_TAG_subprogram ] [line 14] [~SVal]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from SVal]
!20 = metadata !{i32 786468}
!21 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"SVal", metadata !"SVal", metadata !"", i32 12, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !22, i32 12} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!22 = metadata !{i32 786468}
!23 = metadata !{i32 786478, metadata !1, metadata !9, metadata !"SVal", metadata !"SVal", metadata !"", i32 12, metadata !24, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !28, i32 12} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!24 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null, metadata !19, metadata !26}
!26 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!28 = metadata !{i32 786468}
!29 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 25, metadata !30, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 25} ; [ DW_TAG_subprogram ] [line 25] [def] [main]
!30 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !31, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = metadata !{metadata !32}
!32 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!33 = metadata !{i32 786478, metadata !1, null, metadata !"~SVal", metadata !"~SVal", metadata !"_ZN4SValD1Ev", i32 14, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.SVal*)* @_ZN4SValD1Ev, null, metadata !16, metadata !2, i32 14} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!34 = metadata !{i32 786478, metadata !1, null, metadata !"~SVal", metadata !"~SVal", metadata !"_ZN4SValD2Ev", i32 14, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.SVal*)* @_ZN4SValD2Ev, null, metadata !16, metadata !2, i32 14} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!35 = metadata !{i32 786478, metadata !1, null, metadata !"foo", metadata !"foo", metadata !"_ZN1A3fooE4SVal", i32 22, metadata !36, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, %class.SVal*)* @_ZN1A3fooE4SVal, null, metadata !41, metadata !2, i32 22} ; [ DW_TAG_subprogram ] [line 22] [def] [foo]
!36 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !37, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = metadata !{null, metadata !38, metadata !9}
!38 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!39 = metadata !{i32 786434, metadata !1, null, metadata !"A", i32 20, i64 8, i64 8, i32 0, i32 0, null, metadata !40, i32 0, null, null, null} ; [ DW_TAG_class_type ] [A] [line 20, size 8, align 8, offset 0] [def] [from ]
!40 = metadata !{metadata !41, metadata !43}
!41 = metadata !{i32 786478, metadata !1, metadata !39, metadata !"foo", metadata !"foo", metadata !"_ZN1A3fooE4SVal", i32 22, metadata !36, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !42, i32 22} ; [ DW_TAG_subprogram ] [line 22] [foo]
!42 = metadata !{i32 786468}
!43 = metadata !{i32 786478, metadata !1, metadata !39, metadata !"A", metadata !"A", metadata !"", i32 20, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !46, i32 20} ; [ DW_TAG_subprogram ] [line 20] [A]
!44 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !45, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{null, metadata !38}
!46 = metadata !{i32 786468}
!47 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!48 = metadata !{i32 786689, metadata !4, metadata !"v", metadata !5, i32 16777235, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [v] [line 19]
!49 = metadata !{i32 19, i32 0, metadata !4, null}
!50 = metadata !{i32 786688, metadata !29, metadata !"v", metadata !5, i32 26, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [v] [line 26]
!51 = metadata !{i32 26, i32 0, metadata !29, null}
!52 = metadata !{i32 27, i32 0, metadata !29, null}
!53 = metadata !{i32 28, i32 0, metadata !29, null}
!54 = metadata !{i32 786688, metadata !29, metadata !"a", metadata !5, i32 29, metadata !39, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 29]
!55 = metadata !{i32 29, i32 0, metadata !29, null}
!56 = metadata !{i32 30, i32 0, metadata !29, null}
!57 = metadata !{i32 31, i32 0, metadata !29, null}
!58 = metadata !{i32 32, i32 0, metadata !29, null}
!59 = metadata !{i32 786689, metadata !35, metadata !"this", metadata !5, i32 16777238, metadata !60, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 22]
!60 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!61 = metadata !{i32 22, i32 0, metadata !35, null}
!62 = metadata !{i32 786689, metadata !35, metadata !"v", metadata !5, i32 33554454, metadata !9, i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [v] [line 22]
!63 = metadata !{i32 786689, metadata !33, metadata !"this", metadata !5, i32 16777230, metadata !64, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 14]
!64 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from SVal]
!65 = metadata !{i32 14, i32 0, metadata !33, null}
!66 = metadata !{i32 786689, metadata !34, metadata !"this", metadata !5, i32 16777230, metadata !64, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 14]
!67 = metadata !{i32 14, i32 0, metadata !34, null}
!68 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
