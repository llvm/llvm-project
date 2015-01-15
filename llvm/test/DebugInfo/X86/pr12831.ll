; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.function = type { i8 }
%class.BPLFunctionWriter = type { %struct.BPLModuleWriter* }
%struct.BPLModuleWriter = type { i8 }
%class.anon = type { i8 }
%class.anon.0 = type { i8 }

@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_" = internal alias void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"
@"_ZN8functionIFvvEEC1IZN17BPLFunctionWriter9writeExprEvE3$_0EET_" = internal alias void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_"

define void @_ZN17BPLFunctionWriter9writeExprEv(%class.BPLFunctionWriter* %this) nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.BPLFunctionWriter*, align 8
  %agg.tmp = alloca %class.function, align 1
  %agg.tmp2 = alloca %class.anon, align 1
  %agg.tmp4 = alloca %class.function, align 1
  %agg.tmp5 = alloca %class.anon.0, align 1
  store %class.BPLFunctionWriter* %this, %class.BPLFunctionWriter** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.BPLFunctionWriter** %this.addr, metadata !133, metadata !{!"0x102"}), !dbg !135
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE(%struct.BPLModuleWriter*)

define internal void @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_"(%class.function* %this) unnamed_addr nounwind uwtable align 2 {
entry:
  %this.addr = alloca %class.function*, align 8
  %__f = alloca %class.anon.0, align 1
  store %class.function* %this, %class.function** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.function** %this.addr, metadata !140, metadata !{!"0x102"}), !dbg !142
  call void @llvm.dbg.declare(metadata %class.anon.0* %__f, metadata !143, metadata !{!"0x102"}), !dbg !144
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
  call void @llvm.dbg.declare(metadata %class.function** %this.addr, metadata !150, metadata !{!"0x102"}), !dbg !151
  call void @llvm.dbg.declare(metadata %class.anon* %__f, metadata !152, metadata !{!"0x102"}), !dbg !153
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

!0 = !{!"0x11\004\00clang version 3.2 \000\00\000\00\000", !161, !1, !1, !3, !128, null} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5, !106, !107, !126, !127}
!5 = !{!"0x2e\00writeExpr\00writeExpr\00_ZN17BPLFunctionWriter9writeExprEv\0019\000\001\000\006\00256\000\0019", !6, null, !7, null, void (%class.BPLFunctionWriter*)* @_ZN17BPLFunctionWriter9writeExprEv, null, !103, !1} ; [ DW_TAG_subprogram ]
!6 = !{!"0x29", !160} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9}
!9 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !10} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x2\00BPLFunctionWriter\0015\0064\0064\000\000\000", !160, null, null, !11, null, null, null} ; [ DW_TAG_class_type ] [BPLFunctionWriter] [line 15, size 64, align 64, offset 0] [def] [from ]
!11 = !{!12, !103}
!12 = !{!"0xd\00MW\0016\0064\0064\000\001", !160, !10, !13} ; [ DW_TAG_member ]
!13 = !{!"0xf\00\000\0064\0064\000\000", null, null, !14} ; [ DW_TAG_pointer_type ]
!14 = !{!"0x2\00BPLModuleWriter\0012\008\008\000\000\000", !160, null, null, !15, null, null, null} ; [ DW_TAG_class_type ] [BPLModuleWriter] [line 12, size 8, align 8, offset 0] [def] [from ]
!15 = !{!16}
!16 = !{!"0x2e\00writeIntrinsic\00writeIntrinsic\00_ZN15BPLModuleWriter14writeIntrinsicE8functionIFvvEE\0013\000\000\000\006\00256\000\0013", !6, !14, !17, null, null, null, i32 0, !101} ; [ DW_TAG_subprogram ]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19, !20}
!19 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !14} ; [ DW_TAG_pointer_type ]
!20 = !{!"0x2\00function<void ()>\006\008\008\000\000\000", !160, null, null, !21, null, !97, null} ; [ DW_TAG_class_type ] [function<void ()>] [line 6, size 8, align 8, offset 0] [def] [from ]
!21 = !{!22, !51, !58, !86, !92}
!22 = !{!"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00\008\000\000\000\006\00256\000\008", !6, !20, !23, null, null, !47, i32 0, !49} ; [ DW_TAG_subprogram ]
!23 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = !{null, !25, !26}
!25 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !20} ; [ DW_TAG_pointer_type ]
!26 = !{!"0x2\00\0020\008\008\000\000\000", !160, !5, null, !27, null, null, null} ; [ DW_TAG_class_type ] [line 20, size 8, align 8, offset 0] [def] [from ]
!27 = !{!28, !35, !41}
!28 = !{!"0x2e\00operator()\00operator()\00\0020\000\000\000\006\00256\000\0020", !6, !26, !29, null, null, null, i32 0, !33} ; [ DW_TAG_subprogram ]
!29 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !30, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{null, !31}
!31 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !32} ; [ DW_TAG_pointer_type ]
!32 = !{!"0x26\00\000\000\000\000\000", null, null, !26} ; [ DW_TAG_const_type ]
!33 = !{!34}
!34 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!35 = !{!"0x2e\00~\00~\00\0020\000\000\000\006\00320\000\0020", !6, !26, !36, null, null, null, i32 0, !39} ; [ DW_TAG_subprogram ]
!36 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !37, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = !{null, !38}
!38 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !26} ; [ DW_TAG_pointer_type ]
!39 = !{!40}
!40 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!41 = !{!"0x2e\00\00\00\0020\000\000\000\006\00320\000\0020", !6, !26, !42, null, null, null, i32 0, !45} ; [ DW_TAG_subprogram ]
!42 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !43, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!43 = !{null, !38, !44}
!44 = !{!"0x42\00\000\000\000\000\000", null, null, !26} ; [ DW_TAG_rvalue_reference_type ]
!45 = !{!46}
!46 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!47 = !{!48}
!48 = !{!"0x2f\00_Functor\000\000", null, !26, null} ; [ DW_TAG_template_type_parameter ]
!49 = !{!50}
!50 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!51 = !{!"0x2e\00function<function<void ()> >\00function<function<void ()> >\00\008\000\000\000\006\00256\000\008", !6, !20, !52, null, null, !54, i32 0, !56} ; [ DW_TAG_subprogram ]
!52 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !53, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!53 = !{null, !25, !20}
!54 = !{!55}
!55 = !{!"0x2f\00_Functor\000\000", null, !20, null} ; [ DW_TAG_template_type_parameter ]
!56 = !{!57}
!57 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!58 = !{!"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00\008\000\000\000\006\00256\000\008", !6, !20, !59, null, null, !82, i32 0, !84} ; [ DW_TAG_subprogram ]
!59 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !60, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!60 = !{null, !25, !61}
!61 = !{!"0x2\00\0023\008\008\000\000\000", !160, !5, null, !62, null, null, null} ; [ DW_TAG_class_type ] [line 23, size 8, align 8, offset 0] [def] [from ]
!62 = !{!63, !70, !76}
!63 = !{!"0x2e\00operator()\00operator()\00\0023\000\000\000\006\00256\000\0023", !6, !61, !64, null, null, null, i32 0, !68} ; [ DW_TAG_subprogram ]
!64 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !65, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = !{null, !66}
!66 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !67} ; [ DW_TAG_pointer_type ]
!67 = !{!"0x26\00\000\000\000\000\000", null, null, !61} ; [ DW_TAG_const_type ]
!68 = !{!69}
!69 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!70 = !{!"0x2e\00~\00~\00\0023\000\000\000\006\00320\000\0023", !6, !61, !71, null, null, null, i32 0, !74} ; [ DW_TAG_subprogram ]
!71 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !72, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!72 = !{null, !73}
!73 = !{!"0xf\00\000\0064\0064\000\0064", i32 0, null, !61} ; [ DW_TAG_pointer_type ]
!74 = !{!75}
!75 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!76 = !{!"0x2e\00\00\00\0023\000\000\000\006\00320\000\0023", !6, !61, !77, null, null, null, i32 0, !80} ; [ DW_TAG_subprogram ]
!77 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !78, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!78 = !{null, !73, !79}
!79 = !{!"0x42\00\000\000\000\000\000", null, null, !61} ; [ DW_TAG_rvalue_reference_type ]
!80 = !{!81}
!81 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!82 = !{!83}
!83 = !{!"0x2f\00_Functor\000\000", null, !61, null} ; [ DW_TAG_template_type_parameter ]
!84 = !{!85}
!85 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!86 = !{!"0x2e\00function\00function\00\006\000\000\000\006\00320\000\006", !6, !20, !87, null, null, null, i32 0, !90} ; [ DW_TAG_subprogram ]
!87 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !88, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!88 = !{null, !25, !89}
!89 = !{!"0x42\00\000\000\000\000\000", null, null, !20} ; [ DW_TAG_rvalue_reference_type ]
!90 = !{!91}
!91 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!92 = !{!"0x2e\00~function\00~function\00\006\000\000\000\006\00320\000\006", !6, !20, !93, null, null, null, i32 0, !95} ; [ DW_TAG_subprogram ]
!93 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !94, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!94 = !{null, !25}
!95 = !{!96}
!96 = !{!"0x24"}                      ; [ DW_TAG_base_type ]
!97 = !{!98}
!98 = !{!"0x2f\00T\000\000", null, !99, null} ; [ DW_TAG_template_type_parameter ]
!99 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !100, i32 0} ; [ DW_TAG_subroutine_type ]
!100 = !{null}
!101 = !{!102}
!102 = !{!"0x24"}                     ; [ DW_TAG_base_type ]
!103 = !{!"0x2e\00writeExpr\00writeExpr\00_ZN17BPLFunctionWriter9writeExprEv\0017\000\000\000\006\00257\000\0017", !6, !10, !7, null, null, null, i32 0, !104} ; [ DW_TAG_subprogram ]
!104 = !{!105}
!105 = !{!"0x24"}                     ; [ DW_TAG_base_type ]
!106 = !{!"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_\008\001\001\000\006\00256\000\008", !6, null, !59, null, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_1_0EET_", !82, !58, !1} ; [ DW_TAG_subprogram ]
!107 = !{!"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_\003\001\001\000\006\00256\000\003", !6, null, !108, null, void (%class.anon.0*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_", !111, !113, !1} ; [ DW_TAG_subprogram ]
!108 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !109, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!109 = !{null, !110}
!110 = !{!"0x10\00\000\000\000\000\000", null, null, !61} ; [ DW_TAG_reference_type ]
!111 = !{!112}
!112 = !{!"0x2f\00_Tp\000\000", null, !61, null} ; [ DW_TAG_template_type_parameter ]
!113 = !{!"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:23:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_1_0EEvRKT_\003\000\000\000\006\00256\000\003", !6, !114, !108, null, null, !111, i32 0, !124} ; [ DW_TAG_subprogram ]
!114 = !{!"0x2\00_Base_manager\001\008\008\000\000\000", !160, null, null, !115, null, null, null} ; [ DW_TAG_class_type ] [_Base_manager] [line 1, size 8, align 8, offset 0] [def] [from ]
!115 = !{!116, !113}
!116 = !{!"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_\003\000\000\000\006\00256\000\003", !6, !114, !117, null, null, !120, i32 0, !122} ; [ DW_TAG_subprogram ]
!117 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !118, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!118 = !{null, !119}
!119 = !{!"0x10\00\000\000\000\000\000", null, null, !26} ; [ DW_TAG_reference_type ]
!120 = !{!121}
!121 = !{!"0x2f\00_Tp\000\000", null, !26, null} ; [ DW_TAG_template_type_parameter ]
!122 = !{!123}
!123 = !{!"0x24"}                     ; [ DW_TAG_base_type ]
!124 = !{!125}
!125 = !{!"0x24"}                     ; [ DW_TAG_base_type ]
!126 = !{!"0x2e\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_\008\001\001\000\006\00256\000\008", !6, null, !23, null, void (%class.function*)* @"_ZN8functionIFvvEEC2IZN17BPLFunctionWriter9writeExprEvE3$_0EET_", !47, !22, !1} ; [ DW_TAG_subprogram ]
!127 = !{!"0x2e\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_M_not_empty_function<BPLFunctionWriter::<lambda at BPLFunctionWriter2.ii:20:36> >\00_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_\003\001\001\000\006\00256\000\003", !6, null, !117, null, void (%class.anon*)* @"_ZN13_Base_manager21_M_not_empty_functionIZN17BPLFunctionWriter9writeExprEvE3$_0EEvRKT_", !120, !116, !1} ; [ DW_TAG_subprogram ]
!128 = !{!130}
!130 = !{!"0x34\00__stored_locally\00__stored_locally\00__stored_locally\002\001\001", !114, !6, !131, i1 1, null} ; [ DW_TAG_variable ]
!131 = !{!"0x26\00\000\000\000\000\000", null, null, !132} ; [ DW_TAG_const_type ]
!132 = !{!"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ]
!133 = !{!"0x101\00this\0016777235\0064", !5, !6, !134} ; [ DW_TAG_arg_variable ]
!134 = !{!"0xf\00\000\0064\0064\000\000", null, null, !10} ; [ DW_TAG_pointer_type ]
!135 = !MDLocation(line: 19, column: 39, scope: !5)
!136 = !MDLocation(line: 20, column: 17, scope: !137)
!137 = !{!"0xb\0019\0051\000", !6, !5} ; [ DW_TAG_lexical_block ]
!138 = !MDLocation(line: 23, column: 17, scope: !137)
!139 = !MDLocation(line: 26, column: 15, scope: !137)
!140 = !{!"0x101\00this\0016777224\0064", !106, !6, !141} ; [ DW_TAG_arg_variable ]
!141 = !{!"0xf\00\000\0064\0064\000\000", null, null, !20} ; [ DW_TAG_pointer_type ]
!142 = !MDLocation(line: 8, column: 45, scope: !106)
!143 = !{!"0x101\00__f\0033554440\000", !106, !6, !61} ; [ DW_TAG_arg_variable ]
!144 = !MDLocation(line: 8, column: 63, scope: !106)
!145 = !MDLocation(line: 9, column: 9, scope: !146)
!146 = !{!"0xb\008\0081\001", !6, !106} ; [ DW_TAG_lexical_block ]
!147 = !MDLocation(line: 10, column: 13, scope: !146)
!148 = !MDLocation(line: 4, column: 5, scope: !149)
!149 = !{!"0xb\003\00105\002", !6, !107} ; [ DW_TAG_lexical_block ]
!150 = !{!"0x101\00this\0016777224\0064", !126, !6, !141} ; [ DW_TAG_arg_variable ]
!151 = !MDLocation(line: 8, column: 45, scope: !126)
!152 = !{!"0x101\00__f\0033554440\000", !126, !6, !26} ; [ DW_TAG_arg_variable ]
!153 = !MDLocation(line: 8, column: 63, scope: !126)
!154 = !MDLocation(line: 9, column: 9, scope: !155)
!155 = !{!"0xb\008\0081\003", !6, !126} ; [ DW_TAG_lexical_block ]
!156 = !MDLocation(line: 10, column: 13, scope: !155)
!157 = !MDLocation(line: 4, column: 5, scope: !158)
!158 = !{!"0xb\003\00105\004", !6, !127} ; [ DW_TAG_lexical_block ]
!159 = !{!"0x29", !161} ; [ DW_TAG_file_type ]
!160 = !{!"BPLFunctionWriter2.ii", !"/home/peter/crashdelta"}
!161 = !{!"BPLFunctionWriter.cpp", !"/home/peter/crashdelta"}
!162 = !{i32 1, !"Debug Info Version", i32 2}
