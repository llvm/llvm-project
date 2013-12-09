; RUN: llc < %s | FileCheck %s
; Radar 9309221
; Test dwarf reg no for d16
;CHECK: DW_OP_regx
;CHECK-NEXT: 272

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00", align 4
@.str1 = private unnamed_addr constant [6 x i8] c"point\00", align 4

define i32 @inlineprinter(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !19), !dbg !26
  tail call void @llvm.dbg.value(metadata !{double %val}, i64 0, metadata !20), !dbg !26
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !21), !dbg !26
  %0 = zext i8 %c to i32, !dbg !27
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !27
  ret i32 0, !dbg !29
}

define i32 @printer(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize noinline {
entry:
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !16), !dbg !30
  tail call void @llvm.dbg.value(metadata !{double %val}, i64 0, metadata !17), !dbg !30
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !18), !dbg !30
  %0 = zext i8 %c to i32, !dbg !31
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !31
  ret i32 0, !dbg !33
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !22), !dbg !34
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !23), !dbg !34
  %0 = sitofp i32 %argc to double, !dbg !35
  %1 = fadd double %0, 5.555552e+05, !dbg !35
  tail call void @llvm.dbg.value(metadata !{double %1}, i64 0, metadata !24), !dbg !35
  %2 = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0)) nounwind, !dbg !36
  %3 = getelementptr inbounds i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !37
  %4 = trunc i32 %argc to i8, !dbg !37
  %5 = add i8 %4, 97, !dbg !37
  tail call void @llvm.dbg.value(metadata !{i8* %3}, i64 0, metadata !19) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata !{double %1}, i64 0, metadata !20) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i8 %5}, i64 0, metadata !21) nounwind, !dbg !38
  %6 = zext i8 %5 to i32, !dbg !39
  %7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %3, double %1, i32 %6) nounwind, !dbg !39
  %8 = tail call i32 @printer(i8* %3, double %1, i8 zeroext %5) nounwind, !dbg !40
  ret i32 0, !dbg !41
}

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!48}

!0 = metadata !{i32 786478, metadata !46, metadata !1, metadata !"printer", metadata !"printer", metadata !"printer", i32 12, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i8*, double, i8)* @printer, null, null, metadata !43, i32 12} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 786473, metadata !46} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 786449, metadata !46, i32 1, metadata !"(LLVM build 00)", i1 true, metadata !"", i32 0, metadata !47, metadata !47, metadata !42, null,  null, metadata !""} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 786453, metadata !46, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5, metadata !6, metadata !7, metadata !8}
!5 = metadata !{i32 786468, metadata !46, metadata !1, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 786447, metadata !46, metadata !1, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 786468, metadata !46, metadata !1, metadata !"double", i32 0, i64 64, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 786468, metadata !46, metadata !1, metadata !"unsigned char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 8} ; [ DW_TAG_base_type ]
!9 = metadata !{i32 786478, metadata !46, metadata !1, metadata !"inlineprinter", metadata !"inlineprinter", metadata !"inlineprinter", i32 5, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i8*, double, i8)* @inlineprinter, null, null, metadata !44, i32 5} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 786478, metadata !46, metadata !1, metadata !"main", metadata !"main", metadata !"main", i32 18, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i8**)* @main, null, null, metadata !45, i32 18} ; [ DW_TAG_subprogram ]
!11 = metadata !{i32 786453, metadata !46, metadata !1, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !5, metadata !5, metadata !13}
!13 = metadata !{i32 786447, metadata !46, metadata !1, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{i32 786447, metadata !46, metadata !1, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !15} ; [ DW_TAG_pointer_type ]
!15 = metadata !{i32 786468, metadata !46, metadata !1, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ]
!16 = metadata !{i32 786689, metadata !0, metadata !"ptr", metadata !1, i32 11, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 786689, metadata !0, metadata !"val", metadata !1, i32 11, metadata !7, i32 0, null} ; [ DW_TAG_arg_variable ]
!18 = metadata !{i32 786689, metadata !0, metadata !"c", metadata !1, i32 11, metadata !8, i32 0, null} ; [ DW_TAG_arg_variable ]
!19 = metadata !{i32 786689, metadata !9, metadata !"ptr", metadata !1, i32 4, metadata !6, i32 0, null} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 786689, metadata !9, metadata !"val", metadata !1, i32 4, metadata !7, i32 0, null} ; [ DW_TAG_arg_variable ]
!21 = metadata !{i32 786689, metadata !9, metadata !"c", metadata !1, i32 4, metadata !8, i32 0, null} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 786689, metadata !10, metadata !"argc", metadata !1, i32 17, metadata !5, i32 0, null} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 786689, metadata !10, metadata !"argv", metadata !1, i32 17, metadata !13, i32 0, null} ; [ DW_TAG_arg_variable ]
!24 = metadata !{i32 786688, metadata !25, metadata !"dval", metadata !1, i32 19, metadata !7, i32 0, null} ; [ DW_TAG_auto_variable ]
!25 = metadata !{i32 786443, metadata !46, metadata !10, i32 18, i32 0, i32 2} ; [ DW_TAG_lexical_block ]
!26 = metadata !{i32 4, i32 0, metadata !9, null}
!27 = metadata !{i32 6, i32 0, metadata !28, null}
!28 = metadata !{i32 786443, metadata !46, metadata !9, i32 5, i32 0, i32 1} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 7, i32 0, metadata !28, null}
!30 = metadata !{i32 11, i32 0, metadata !0, null}
!31 = metadata !{i32 13, i32 0, metadata !32, null}
!32 = metadata !{i32 786443, metadata !46, metadata !0, i32 12, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!33 = metadata !{i32 14, i32 0, metadata !32, null}
!34 = metadata !{i32 17, i32 0, metadata !10, null}
!35 = metadata !{i32 19, i32 0, metadata !25, null}
!36 = metadata !{i32 20, i32 0, metadata !25, null}
!37 = metadata !{i32 21, i32 0, metadata !25, null}
!38 = metadata !{i32 4, i32 0, metadata !9, metadata !37}
!39 = metadata !{i32 6, i32 0, metadata !28, metadata !37}
!40 = metadata !{i32 22, i32 0, metadata !25, null}
!41 = metadata !{i32 23, i32 0, metadata !25, null}
!42 = metadata !{metadata !0, metadata !9, metadata !10}
!43 = metadata !{metadata !16, metadata !17, metadata !18}
!44 = metadata !{metadata !19, metadata !20, metadata !21}
!45 = metadata !{metadata !22, metadata !23, metadata !24}
!46 = metadata !{metadata !"a.c", metadata !"/tmp/"}
!47 = metadata !{i32 0}
!48 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
