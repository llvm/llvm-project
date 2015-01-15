; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Check debug info output for merged global.
; DW_AT_location
; 0x03 DW_OP_addr
; 0x.. .long __MergedGlobals
; 0x10 DW_OP_constu
; 0x.. offset
; 0x22 DW_OP_plus

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x1"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x8> 03 [[ADDR:.. .. .. ..]] 10 00 22  )
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x2"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x8> 03 [[ADDR]] 10 04 22  )

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.7.0"

@x1 = internal unnamed_addr global i32 1, align 4
@x2 = internal unnamed_addr global i32 2, align 4
@x3 = internal unnamed_addr global i32 3, align 4
@x4 = internal unnamed_addr global i32 4, align 4
@x5 = global i32 0, align 4

define i32 @get1(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !30
  %1 = load i32* @x1, align 4, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !31
  store i32 %a, i32* @x1, align 4, !dbg !31
  ret i32 %1, !dbg !31
}

define i32 @get2(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !32
  %1 = load i32* @x2, align 4, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !33
  store i32 %a, i32* @x2, align 4, !dbg !33
  ret i32 %1, !dbg !33
}

define i32 @get3(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !34
  %1 = load i32* @x3, align 4, !dbg !35
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !17, metadata !{!"0x102"}), !dbg !35
  store i32 %a, i32* @x3, align 4, !dbg !35
  ret i32 %1, !dbg !35
}

define i32 @get4(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !19, metadata !{!"0x102"}), !dbg !36
  %1 = load i32* @x4, align 4, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !20, metadata !{!"0x102"}), !dbg !37
  store i32 %a, i32* @x4, align 4, !dbg !37
  ret i32 %1, !dbg !37
}

define i32 @get5(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !27, metadata !{!"0x102"}), !dbg !38
  %1 = load i32* @x5, align 4, !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !28, metadata !{!"0x102"}), !dbg !39
  store i32 %a, i32* @x5, align 4, !dbg !39
  ret i32 %1, !dbg !39
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!49}

!0 = !{!"0x11\0012\00clang\001\00\000\00\001", !47, !48, !48, !40, !41,  !48} ; [ DW_TAG_compile_unit ]
!1 = !{!"0x2e\00get1\00get1\00\005\000\001\000\006\00256\001\005", !47, !2, !3, null, i32 (i32)* @get1, null, null, !42} ; [ DW_TAG_subprogram ] [line 5] [def] [get1]
!2 = !{!"0x29", !47} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !47, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !0} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00get2\00get2\00\008\000\001\000\006\00256\001\008", !47, !2, !3, null, i32 (i32)* @get2, null, null, !43} ; [ DW_TAG_subprogram ] [line 8] [def] [get2]
!7 = !{!"0x2e\00get3\00get3\00\0011\000\001\000\006\00256\001\0011", !47, !2, !3, null, i32 (i32)* @get3, null, null, !44} ; [ DW_TAG_subprogram ] [line 11] [def] [get3]
!8 = !{!"0x2e\00get4\00get4\00\0014\000\001\000\006\00256\001\0014", !47, !2, !3, null, i32 (i32)* @get4, null, null, !45} ; [ DW_TAG_subprogram ] [line 14] [def] [get4]
!9 = !{!"0x2e\00get5\00get5\00\0017\000\001\000\006\00256\001\0017", !47, !2, !3, null, i32 (i32)* @get5, null, null, !46} ; [ DW_TAG_subprogram ] [line 17] [def] [get5]
!10 = !{!"0x101\00a\0016777221\000", !1, !2, !5} ; [ DW_TAG_arg_variable ]
!11 = !{!"0x100\00b\005\000", !12, !2, !5} ; [ DW_TAG_auto_variable ]
!12 = !{!"0xb\005\0019\000", !47, !1} ; [ DW_TAG_lexical_block ]
!13 = !{!"0x101\00a\0016777224\000", !6, !2, !5} ; [ DW_TAG_arg_variable ]
!14 = !{!"0x100\00b\008\000", !15, !2, !5} ; [ DW_TAG_auto_variable ]
!15 = !{!"0xb\008\0017\001", !47, !6} ; [ DW_TAG_lexical_block ]
!16 = !{!"0x101\00a\0016777227\000", !7, !2, !5} ; [ DW_TAG_arg_variable ]
!17 = !{!"0x100\00b\0011\000", !18, !2, !5} ; [ DW_TAG_auto_variable ]
!18 = !{!"0xb\0011\0019\002", !47, !7} ; [ DW_TAG_lexical_block ]
!19 = !{!"0x101\00a\0016777230\000", !8, !2, !5} ; [ DW_TAG_arg_variable ]
!20 = !{!"0x100\00b\0014\000", !21, !2, !5} ; [ DW_TAG_auto_variable ]
!21 = !{!"0xb\0014\0019\003", !47, !8} ; [ DW_TAG_lexical_block ]
!25 = !{!"0x34\00x1\00x1\00\004\001\001", !0, !2, !5, i32* @x1, null} ; [ DW_TAG_variable ]
!26 = !{!"0x34\00x2\00x2\00\007\001\001", !0, !2, !5, i32* @x2, null} ; [ DW_TAG_variable ]
!27 = !{!"0x101\00a\0016777233\000", !9, !2, !5} ; [ DW_TAG_arg_variable ]
!28 = !{!"0x100\00b\0017\000", !29, !2, !5} ; [ DW_TAG_auto_variable ]
!29 = !{!"0xb\0017\0019\004", !47, !9} ; [ DW_TAG_lexical_block ]
!30 = !MDLocation(line: 5, column: 16, scope: !1)
!31 = !MDLocation(line: 5, column: 32, scope: !12)
!32 = !MDLocation(line: 8, column: 14, scope: !6)
!33 = !MDLocation(line: 8, column: 29, scope: !15)
!34 = !MDLocation(line: 11, column: 16, scope: !7)
!35 = !MDLocation(line: 11, column: 32, scope: !18)
!36 = !MDLocation(line: 14, column: 16, scope: !8)
!37 = !MDLocation(line: 14, column: 32, scope: !21)
!38 = !MDLocation(line: 17, column: 16, scope: !9)
!39 = !MDLocation(line: 17, column: 32, scope: !29)
!40 = !{!1, !6, !7, !8, !9}
!41 = !{!25, !26}
!42 = !{!10, !11}
!43 = !{!13, !14}
!44 = !{!16, !17}
!45 = !{!19, !20}
!46 = !{!27, !28}
!47 = !{!"ss3.c", !"/private/tmp"}
!48 = !{}
!49 = !{i32 1, !"Debug Info Version", i32 2}
