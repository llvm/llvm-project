; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.function = type { i8 }
%class.BPLFunctionWriter = type { %struct.BPLModuleWriter* }
%struct.BPLModuleWriter = type { i8 }
%class.anon = type { i8 }
%class.anon.0 = type { i8 }

@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_" = alias internal void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"
@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_" = alias internal void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"

define void @_ZN17BPLFunctionWriter9writeExprEv(%class.BPLFunctionWriter* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.BPLFunctionWriter*, align 8
  %agg.tmp = alloca %class.function, align 1
  %agg.tmp2 = alloca %class.anon, align 1
  %agg.tmp4 = alloca %class.function, align 1
  %agg.tmp5 = alloca %class.anon.0, align 1
  store %class.BPLFunctionWriter* %this, %class.BPLFunctionWriter** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.BPLFunctionWriter** %this.addr}, metadata !133), !dbg !135
  %this1 = load %class.BPLFunctionWriter** %this.addr
  %MW = getelementptr inbounds %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !136
  %0 = load %struct.BPLModuleWriter** %MW, align 8, !dbg !136
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %agg.tmp), !dbg !136
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %0), !dbg !136
  %MW3 = getelementptr inbounds %class.BPLFunctionWriter* %this1, i32 0, i32 0, !dbg !138
  %1 = load %struct.BPLModuleWriter** %MW3, align 8, !dbg !138
  call void @"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %agg.tmp4), !dbg !138
  call void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter* %1), !dbg !138
  ret void, !dbg !139
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter*)

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon.0, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.function** %this.addr}, metadata !140), !dbg !142
  call void @llvm.dbg.declare(metadata !{%class.anon.0* %__f}, metadata !143), !dbg !144
  %this1 = load %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0* %__f), !dbg !145
  ret void, !dbg !147
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_"(%class.anon.0*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon.0*, align 8
  store %class.anon.0* %0, %class.anon.0** %.addr, align 8
  ret void, !dbg !148
}

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.function** %this.addr}, metadata !150), !dbg !151
  call void @llvm.dbg.declare(metadata !{%class.anon* %__f}, metadata !152), !dbg !153
  %this1 = load %class.function** %this.addr
  call void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon* %__f), !dbg !154
  ret void, !dbg !156
}

define internal void @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_"(%class.anon*) nounwind uwtable align 2 {
entry:
  %.addr = alloca %class.anon*, align 8
  store %class.anon* %0, %class.anon** %.addr, align 8
  ret void, !dbg !157
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!162}

!0 = metadata !{i32 786449, metadata !161, i32 4, metadata !"clang version 3.2 ", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !128, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !106, metadata !107, metadata !126, metadata !127}
!5 = metadata !{i32 786478, metadata !6, null, metadata !"writeExpr", metadata !"writeExpr", metadata !"_ZN17BPLFunctionWriter9writeExprEv", i32 19, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.BPLFunctionWriter*)* @_ZN17BPLFunctionWriter9writeExprEv, null, metadata !103, metadata !1, i32 19} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !160} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 786434, metadata !160, null, metadata !"BPLFunctionWriter", i32 15, i64 64, i64 64, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_class_type ] [BPLFunctionWriter] [line 15, size 64, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !103}
!12 = metadata !{i32 786445, metadata !160, metadata !10, metadata !"MW", i32 16, i64 64, i64 64, i64 0, i32 1, metadata !13} ; [ DW_TAG_member ]
!13 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 786434, metadata !160, null, metadata !"BPLModuleWriter", i32 12, i64 8, i64 8, i32 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_class_type ] [BPLModuleWriter] [line 12, size 8, align 8, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786478, metadata !6, metadata !14, metadata !"writeIntrinsic", metadata !"writeIntrinsic", metadata !"_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE", i32 13, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !101, i32 13} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19, metadata !20}
!19 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !14} ; [ DW_TAG_pointer_type ]
!20 = metadata !{i32 786434, metadata !160, null, metadata !"function<void ()>", i32 6, i64 8, i64 8, i32 0, i32 0, null, metadata !21, i32 0, null, metadata !97, null} ; [ DW_TAG_class_type ] [function<void ()>] [line 6, size 8, align 8, offset 0] [def] [from ]
!21 = metadata !{metadata !22, metadata !51, metadata !58, metadata !86, metadata !92}
!22 = metadata !{i32 786478, metadata !6, metadata !20, metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"", i32 8, metadata !23, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, metadata !47, i32 0, metadata !49, i32 8} ; [ DW_TAG_subprogram ]
!23 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !24, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null, metadata !25, metadata !26}
!25 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !20} ; [ DW_TAG_pointer_type ]
!26 = metadata !{i32 786434, metadata !160, metadata !5, metadata !"", i32 20, i64 8, i64 8, i32 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_class_type ] [line 20, size 8, align 8, offset 0] [def] [from ]
!27 = metadata !{metadata !28, metadata !35, metadata !41}
!28 = metadata !{i32 786478, metadata !6, metadata !26, metadata !"operator()", metadata !"operator()", metadata !"", i32 20, metadata !29, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !33, i32 20} ; [ DW_TAG_subprogram ]
!29 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{null, metadata !31}
!31 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !32} ; [ DW_TAG_pointer_type ]
!32 = metadata !{i32 786470, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !26} ; [ DW_TAG_const_type ]
!33 = metadata !{metadata !34}
!34 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!35 = metadata !{i32 786478, metadata !6, metadata !26, metadata !"~", metadata !"~", metadata !"", i32 20, metadata !36, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !39, i32 20} ; [ DW_TAG_subprogram ]
!36 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !37, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = metadata !{null, metadata !38}
!38 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !26} ; [ DW_TAG_pointer_type ]
!39 = metadata !{metadata !40}
!40 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!41 = metadata !{i32 786478, metadata !6, metadata !26, metadata !"", metadata !"", metadata !"", i32 20, metadata !42, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !45, i32 20} ; [ DW_TAG_subprogram ]
!42 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !43, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = metadata !{null, metadata !38, metadata !44}
!44 = metadata !{i32 786498, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !26} ; [ DW_TAG_rvalue_reference_type ]
!45 = metadata !{metadata !46}
!46 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!47 = metadata !{metadata !48}
!48 = metadata !{i32 786479, null, metadata !"_Functor", metadata !26, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!49 = metadata !{metadata !50}
!50 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!51 = metadata !{i32 786478, metadata !6, metadata !20, metadata !"function<function<void ()> >", metadata !"function<function<void ()> >", metadata !"", i32 8, metadata !52, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, metadata !54, i32 0, metadata !56, i32 8} ; [ DW_TAG_subprogram ]
!52 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !53, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!53 = metadata !{null, metadata !25, metadata !20}
!54 = metadata !{metadata !55}
!55 = metadata !{i32 786479, null, metadata !"_Functor", metadata !20, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!56 = metadata !{metadata !57}
!57 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!58 = metadata !{i32 786478, metadata !6, metadata !20, metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"", i32 8, metadata !59, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, metadata !82, i32 0, metadata !84, i32 8} ; [ DW_TAG_subprogram ]
!59 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !60, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!60 = metadata !{null, metadata !25, metadata !61}
!61 = metadata !{i32 786434, metadata !160, metadata !5, metadata !"", i32 23, i64 8, i64 8, i32 0, i32 0, null, metadata !62, i32 0, null, null, null} ; [ DW_TAG_class_type ] [line 23, size 8, align 8, offset 0] [def] [from ]
!62 = metadata !{metadata !63, metadata !70, metadata !76}
!63 = metadata !{i32 786478, metadata !6, metadata !61, metadata !"operator()", metadata !"operator()", metadata !"", i32 23, metadata !64, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !68, i32 23} ; [ DW_TAG_subprogram ]
!64 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !65, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = metadata !{null, metadata !66}
!66 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !67} ; [ DW_TAG_pointer_type ]
!67 = metadata !{i32 786470, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !61} ; [ DW_TAG_const_type ]
!68 = metadata !{metadata !69}
!69 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!70 = metadata !{i32 786478, metadata !6, metadata !61, metadata !"~", metadata !"~", metadata !"", i32 23, metadata !71, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !74, i32 23} ; [ DW_TAG_subprogram ]
!71 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !72, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!72 = metadata !{null, metadata !73}
!73 = metadata !{i32 786447, i32 0, null, i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !61} ; [ DW_TAG_pointer_type ]
!74 = metadata !{metadata !75}
!75 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!76 = metadata !{i32 786478, metadata !6, metadata !61, metadata !"", metadata !"", metadata !"", i32 23, metadata !77, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !80, i32 23} ; [ DW_TAG_subprogram ]
!77 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !78, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!78 = metadata !{null, metadata !73, metadata !79}
!79 = metadata !{i32 786498, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !61} ; [ DW_TAG_rvalue_reference_type ]
!80 = metadata !{metadata !81}
!81 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!82 = metadata !{metadata !83}
!83 = metadata !{i32 786479, null, metadata !"_Functor", metadata !61, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!84 = metadata !{metadata !85}
!85 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!86 = metadata !{i32 786478, metadata !6, metadata !20, metadata !"function", metadata !"function", metadata !"", i32 6, metadata !87, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !90, i32 6} ; [ DW_TAG_subprogram ]
!87 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !88, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!88 = metadata !{null, metadata !25, metadata !89}
!89 = metadata !{i32 786498, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_rvalue_reference_type ]
!90 = metadata !{metadata !91}
!91 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!92 = metadata !{i32 786478, metadata !6, metadata !20, metadata !"~function", metadata !"~function", metadata !"", i32 6, metadata !93, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !95, i32 6} ; [ DW_TAG_subprogram ]
!93 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !94, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!94 = metadata !{null, metadata !25}
!95 = metadata !{metadata !96}
!96 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!97 = metadata !{metadata !98}
!98 = metadata !{i32 786479, null, metadata !"T", metadata !99, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!99 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !100, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!100 = metadata !{null}
!101 = metadata !{metadata !102}
!102 = metadata !{i32 786468}                     ; [ DW_TAG_base_type ]
!103 = metadata !{i32 786478, metadata !6, metadata !10, metadata !"writeExpr", metadata !"writeExpr", metadata !"_ZN17BPLFunctionWriter9writeExprEv", i32 17, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 257, i1 false, null, null, i32 0, metadata !104, i32 17} ; [ DW_TAG_subprogram ]
!104 = metadata !{metadata !105}
!105 = metadata !{i32 786468}                     ; [ DW_TAG_base_type ]
!106 = metadata !{i32 786478, metadata !6, null, metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_", i32 8, metadata !59, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_", metadata !82, metadata !58, metadata !1, i32 8} ; [ DW_TAG_subprogram ]
!107 = metadata !{i32 786478, metadata !6, null, metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", i32 3, metadata !108, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.anon.0*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", metadata !111, metadata !113, metadata !1, i32 3} ; [ DW_TAG_subprogram ]
!108 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !109, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!109 = metadata !{null, metadata !110}
!110 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !61} ; [ DW_TAG_reference_type ]
!111 = metadata !{metadata !112}
!112 = metadata !{i32 786479, null, metadata !"_Tp", metadata !61, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!113 = metadata !{i32 786478, metadata !6, metadata !114, metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >", metadata !"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", i32 3, metadata !108, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, metadata !111, i32 0, metadata !124, i32 3} ; [ DW_TAG_subprogram ]
!114 = metadata !{i32 786434, metadata !160, null, metadata !"_Base_manager", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !115, i32 0, null, null, null} ; [ DW_TAG_class_type ] [_Base_manager] [line 1, size 8, align 8, offset 0] [def] [from ]
!115 = metadata !{metadata !116, metadata !113}
!116 = metadata !{i32 786478, metadata !6, metadata !114, metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", i32 3, metadata !117, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, metadata !120, i32 0, metadata !122, i32 3} ; [ DW_TAG_subprogram ]
!117 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !118, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!118 = metadata !{null, metadata !119}
!119 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !26} ; [ DW_TAG_reference_type ]
!120 = metadata !{metadata !121}
!121 = metadata !{i32 786479, null, metadata !"_Tp", metadata !26, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!122 = metadata !{metadata !123}
!123 = metadata !{i32 786468}                     ; [ DW_TAG_base_type ]
!124 = metadata !{metadata !125}
!125 = metadata !{i32 786468}                     ; [ DW_TAG_base_type ]
!126 = metadata !{i32 786478, metadata !6, null, metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_", i32 8, metadata !23, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_", metadata !47, metadata !22, metadata !1, i32 8} ; [ DW_TAG_subprogram ]
!127 = metadata !{i32 786478, metadata !6, null, metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >", metadata !"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", i32 3, metadata !117, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.anon*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", metadata !120, metadata !116, metadata !1, i32 3} ; [ DW_TAG_subprogram ]
!128 = metadata !{metadata !130}
!130 = metadata !{i32 786484, i32 0, metadata !114, metadata !"__stored_locally", metadata !"__stored_locally", metadata !"__stored_locally", metadata !6, i32 2, metadata !131, i32 1, i32 1, i1 true, null} ; [ DW_TAG_variable ]
!131 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !132} ; [ DW_TAG_const_type ]
!132 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!133 = metadata !{i32 786689, metadata !5, metadata !"this", metadata !6, i32 16777235, metadata !134, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!134 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ]
!135 = metadata !{i32 19, i32 39, metadata !5, null}
!136 = metadata !{i32 20, i32 17, metadata !137, null}
!137 = metadata !{i32 786443, metadata !5, i32 19, i32 51, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
!138 = metadata !{i32 23, i32 17, metadata !137, null}
!139 = metadata !{i32 26, i32 15, metadata !137, null}
!140 = metadata !{i32 786689, metadata !106, metadata !"this", metadata !6, i32 16777224, metadata !141, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!141 = metadata !{i32 786447, null, null, null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !20} ; [ DW_TAG_pointer_type ]
!142 = metadata !{i32 8, i32 45, metadata !106, null}
!143 = metadata !{i32 786689, metadata !106, metadata !"__f", metadata !6, i32 33554440, metadata !61, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!144 = metadata !{i32 8, i32 63, metadata !106, null}
!145 = metadata !{i32 9, i32 9, metadata !146, null}
!146 = metadata !{i32 786443, metadata !106, i32 8, i32 81, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!147 = metadata !{i32 10, i32 13, metadata !146, null}
!148 = metadata !{i32 4, i32 5, metadata !149, null}
!149 = metadata !{i32 786443, metadata !107, i32 3, i32 105, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!150 = metadata !{i32 786689, metadata !126, metadata !"this", metadata !6, i32 16777224, metadata !141, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!151 = metadata !{i32 8, i32 45, metadata !126, null}
!152 = metadata !{i32 786689, metadata !126, metadata !"__f", metadata !6, i32 33554440, metadata !26, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!153 = metadata !{i32 8, i32 63, metadata !126, null}
!154 = metadata !{i32 9, i32 9, metadata !155, null}
!155 = metadata !{i32 786443, metadata !126, i32 8, i32 81, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!156 = metadata !{i32 10, i32 13, metadata !155, null}
!157 = metadata !{i32 4, i32 5, metadata !158, null}
!158 = metadata !{i32 786443, metadata !127, i32 3, i32 105, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!159 = metadata !{i32 786473, metadata !161} ; [ DW_TAG_file_type ]
!160 = metadata !{metadata !"BPLFunctionWriter2.ii", metadata !"/home/peter/crashdelta"}
!161 = metadata !{metadata !"BPLFunctionWriter.cpp", metadata !"/home/peter/crashdelta"}
!162 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
