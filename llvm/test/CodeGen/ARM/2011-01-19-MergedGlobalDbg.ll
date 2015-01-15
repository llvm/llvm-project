; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@x1 = internal global i8 1, align 1
@x2 = internal global i8 1, align 1
@x3 = internal global i8 1, align 1
@x4 = internal global i8 1, align 1
@x5 = global i8 1, align 1

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
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x8> 03 [[ADDR]] 10 01 22  )

define zeroext i8 @get1(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !30
  %0 = load i8* @x1, align 4, !dbg !30
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !30
  store i8 %a, i8* @x1, align 4, !dbg !30
  ret i8 %0, !dbg !31
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define zeroext i8 @get2(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !32
  %0 = load i8* @x2, align 4, !dbg !32
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !19, metadata !{!"0x102"}), !dbg !32
  store i8 %a, i8* @x2, align 4, !dbg !32
  ret i8 %0, !dbg !33
}

define zeroext i8 @get3(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !21, metadata !{!"0x102"}), !dbg !34
  %0 = load i8* @x3, align 4, !dbg !34
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !22, metadata !{!"0x102"}), !dbg !34
  store i8 %a, i8* @x3, align 4, !dbg !34
  ret i8 %0, !dbg !35
}

define zeroext i8 @get4(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !24, metadata !{!"0x102"}), !dbg !36
  %0 = load i8* @x4, align 4, !dbg !36
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !25, metadata !{!"0x102"}), !dbg !36
  store i8 %a, i8* @x4, align 4, !dbg !36
  ret i8 %0, !dbg !37
}

define zeroext i8 @get5(i8 zeroext %a) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i8 %a, i64 0, metadata !27, metadata !{!"0x102"}), !dbg !38
  %0 = load i8* @x5, align 4, !dbg !38
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !28, metadata !{!"0x102"}), !dbg !38
  store i8 %a, i8* @x5, align 4, !dbg !38
  ret i8 %0, !dbg !39
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!49}

!0 = !{!"0x2e\00get1\00get1\00get1\004\000\001\000\006\00256\001\004", !47, !1, !3, null, i8 (i8)* @get1, null, null, !42} ; [ DW_TAG_subprogram ]
!1 = !{!"0x29", !47} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build 2369.8)\001\00\000\00\000", !47, !48, !48, !40, !41,  !48} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !47, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5, !5}
!5 = !{!"0x24\00_Bool\000\008\008\000\000\002", !47, !1} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00get2\00get2\00get2\007\000\001\000\006\00256\001\007", !47, !1, !3, null, i8 (i8)* @get2, null, null, !43} ; [ DW_TAG_subprogram ]
!7 = !{!"0x2e\00get3\00get3\00get3\0010\000\001\000\006\00256\001\0010", !47, !1, !3, null, i8 (i8)* @get3, null, null, !44} ; [ DW_TAG_subprogram ]
!8 = !{!"0x2e\00get4\00get4\00get4\0013\000\001\000\006\00256\001\0013", !47, !1, !3, null, i8 (i8)* @get4, null, null, !45} ; [ DW_TAG_subprogram ]
!9 = !{!"0x2e\00get5\00get5\00get5\0016\000\001\000\006\00256\001\0016", !47, !1, !3, null, i8 (i8)* @get5, null, null, !46} ; [ DW_TAG_subprogram ]
!10 = !{!"0x101\00a\004\000", !0, !1, !5} ; [ DW_TAG_arg_variable ]
!11 = !{!"0x100\00b\004\000", !12, !1, !5} ; [ DW_TAG_auto_variable ]
!12 = !{!"0xb\004\000\000", !47, !0} ; [ DW_TAG_lexical_block ]
!13 = !{!"0x34\00x1\00x1\00\003\001\001", !1, !1, !5, i8* @x1, null} ; [ DW_TAG_variable ]
!14 = !{!"0x34\00x2\00x2\00\006\001\001", !1, !1, !5, i8* @x2, null} ; [ DW_TAG_variable ]
!15 = !{!"0x34\00x3\00x3\00\009\001\001", !1, !1, !5, i8* @x3, null} ; [ DW_TAG_variable ]
!16 = !{!"0x34\00x4\00x4\00\0012\001\001", !1, !1, !5, i8* @x4, null} ; [ DW_TAG_variable ]
!17 = !{!"0x34\00x5\00x5\00\0015\000\001", !1, !1, !5, i8* @x5, null} ; [ DW_TAG_variable ]
!18 = !{!"0x101\00a\007\000", !6, !1, !5} ; [ DW_TAG_arg_variable ]
!19 = !{!"0x100\00b\007\000", !20, !1, !5} ; [ DW_TAG_auto_variable ]
!20 = !{!"0xb\007\000\001", !47, !6} ; [ DW_TAG_lexical_block ]
!21 = !{!"0x101\00a\0010\000", !7, !1, !5} ; [ DW_TAG_arg_variable ]
!22 = !{!"0x100\00b\0010\000", !23, !1, !5} ; [ DW_TAG_auto_variable ]
!23 = !{!"0xb\0010\000\002", !47, !7} ; [ DW_TAG_lexical_block ]
!24 = !{!"0x101\00a\0013\000", !8, !1, !5} ; [ DW_TAG_arg_variable ]
!25 = !{!"0x100\00b\0013\000", !26, !1, !5} ; [ DW_TAG_auto_variable ]
!26 = !{!"0xb\0013\000\003", !47, !8} ; [ DW_TAG_lexical_block ]
!27 = !{!"0x101\00a\0016\000", !9, !1, !5} ; [ DW_TAG_arg_variable ]
!28 = !{!"0x100\00b\0016\000", !29, !1, !5} ; [ DW_TAG_auto_variable ]
!29 = !{!"0xb\0016\000\004", !47, !9} ; [ DW_TAG_lexical_block ]
!30 = !MDLocation(line: 4, scope: !0)
!31 = !MDLocation(line: 4, scope: !12)
!32 = !MDLocation(line: 7, scope: !6)
!33 = !MDLocation(line: 7, scope: !20)
!34 = !MDLocation(line: 10, scope: !7)
!35 = !MDLocation(line: 10, scope: !23)
!36 = !MDLocation(line: 13, scope: !8)
!37 = !MDLocation(line: 13, scope: !26)
!38 = !MDLocation(line: 16, scope: !9)
!39 = !MDLocation(line: 16, scope: !29)
!40 = !{!0, !6, !7, !8, !9}
!41 = !{!13, !14, !15, !16, !17}
!42 = !{!10, !11}
!43 = !{!18, !19}
!44 = !{!21, !22}
!45 = !{!24, !25}
!46 = !{!27, !28}
!47 = !{!"foo.c", !"/tmp/"}
!48 = !{}
!49 = !{i32 1, !"Debug Info Version", i32 2}
