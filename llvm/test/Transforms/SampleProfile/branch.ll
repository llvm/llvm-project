; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/branch.prof | opt -analyze -branch-prob | FileCheck %s

; Original C++ code for this test case:
;
; #include <stdio.h>
; #include <stdlib.h>
;
; int main(int argc, char *argv[]) {
;   if (argc < 2)
;     return 1;
;   double result;
;   int limit = atoi(argv[1]);
;   if (limit > 100) {
;     double s = 23.041968;
;     for (int u = 0; u < limit; u++) {
;       double x = s;
;       s = x + 3.049 + (double)u;
;       s -= s + 3.94 / x * 0.32;
;     }
;     result = s;
;   } else {
;     result = 0;
;   }
;   printf("result is %lf\n", result);
;   return 0;
; }

@.str = private unnamed_addr constant [15 x i8] c"result is %lf\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) #0 {
; CHECK: Printing analysis 'Branch Probability Analysis' for function 'main':

entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !13, metadata !{}), !dbg !27
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !14, metadata !{}), !dbg !27
  %cmp = icmp slt i32 %argc, 2, !dbg !28
  br i1 %cmp, label %return, label %if.end, !dbg !28
; CHECK: edge entry -> return probability is 1 / 2 = 50%
; CHECK: edge entry -> if.end probability is 1 / 2 = 50%

if.end:                                           ; preds = %entry
  %arrayidx = getelementptr inbounds i8** %argv, i64 1, !dbg !30
  %0 = load i8** %arrayidx, align 8, !dbg !30, !tbaa !31
  %call = tail call i32 @atoi(i8* %0) #4, !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !17, metadata !{}), !dbg !30
  %cmp1 = icmp sgt i32 %call, 100, !dbg !35
  br i1 %cmp1, label %for.body, label %if.end6, !dbg !35
; CHECK: edge if.end -> for.body probability is 1 / 2 = 50%
; CHECK: edge if.end -> if.end6 probability is 1 / 2 = 50%

for.body:                                         ; preds = %if.end, %for.body
  %u.016 = phi i32 [ %inc, %for.body ], [ 0, %if.end ]
  %s.015 = phi double [ %sub, %for.body ], [ 0x40370ABE6A337A81, %if.end ]
  %add = fadd double %s.015, 3.049000e+00, !dbg !36
  %conv = sitofp i32 %u.016 to double, !dbg !36
  %add4 = fadd double %add, %conv, !dbg !36
  tail call void @llvm.dbg.value(metadata double %add4, i64 0, metadata !18, metadata !{}), !dbg !36
  %div = fdiv double 3.940000e+00, %s.015, !dbg !37
  %mul = fmul double %div, 3.200000e-01, !dbg !37
  %add5 = fadd double %add4, %mul, !dbg !37
  %sub = fsub double %add4, %add5, !dbg !37
  tail call void @llvm.dbg.value(metadata double %sub, i64 0, metadata !18, metadata !{}), !dbg !37
  %inc = add nsw i32 %u.016, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !21, metadata !{}), !dbg !38
  %exitcond = icmp eq i32 %inc, %call, !dbg !38
  br i1 %exitcond, label %if.end6, label %for.body, !dbg !38
; CHECK: edge for.body -> if.end6 probability is 1 / 10227 = 0.00977804
; CHECK: edge for.body -> for.body probability is 10226 / 10227 = 99.9902% [HOT edge]

if.end6:                                          ; preds = %for.body, %if.end
  %result.0 = phi double [ 0.000000e+00, %if.end ], [ %sub, %for.body ]
  %call7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @.str, i64 0, i64 0), double %result.0), !dbg !39
  br label %return, !dbg !40
; CHECK: edge if.end6 -> return probability is 16 / 16 = 100% [HOT edge]

return:                                           ; preds = %entry, %if.end6
  %retval.0 = phi i32 [ 0, %if.end6 ], [ 1, %entry ]
  ret i32 %retval.0, !dbg !41
}

; Function Attrs: nounwind readonly
declare i32 @atoi(i8* nocapture) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !42}
!llvm.ident = !{!26}

!0 = !{!"0x11\004\00clang version 3.4 (trunk 192896) (llvm/trunk 192895)\001\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [./branch.cc] [DW_LANG_C_plus_plus]
!1 = !{!"branch.cc", !"."}
!2 = !{i32 0}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\004\000\001\000\006\00256\001\004", !1, !5, !6, null, i32 (i32, i8**)* @main, null, null, !12} ; [ DW_TAG_subprogram ] [line 4] [def] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [./branch.cc]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8, !9}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0xf\00\000\0064\0064\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!11 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!12 = !{!13, !14, !15, !17, !18, !21, !23}
!13 = !{!"0x101\00argc\0016777220\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [argc] [line 4]
!14 = !{!"0x101\00argv\0033554436\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [argv] [line 4]
!15 = !{!"0x100\00result\007\000", !4, !5, !16} ; [ DW_TAG_auto_variable ] [result] [line 7]
!16 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!17 = !{!"0x100\00limit\008\000", !4, !5, !8} ; [ DW_TAG_auto_variable ] [limit] [line 8]
!18 = !{!"0x100\00s\0010\000", !19, !5, !16} ; [ DW_TAG_auto_variable ] [s] [line 10]
!19 = !{!"0xb\009\000\000", !1, !20} ; [ DW_TAG_lexical_block ] [./branch.cc]
!20 = !{!"0xb\009\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [./branch.cc]
!21 = !{!"0x100\00u\0011\000", !22, !5, !8} ; [ DW_TAG_auto_variable ] [u] [line 11]
!22 = !{!"0xb\0011\000\000", !1, !19} ; [ DW_TAG_lexical_block ] [./branch.cc]
!23 = !{!"0x100\00x\0012\000", !24, !5, !16} ; [ DW_TAG_auto_variable ] [x] [line 12]
!24 = !{!"0xb\0011\000\000", !1, !22} ; [ DW_TAG_lexical_block ] [./branch.cc]
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{!"clang version 3.4 (trunk 192896) (llvm/trunk 192895)"}
!27 = !MDLocation(line: 4, scope: !4)
!28 = !MDLocation(line: 5, scope: !29)
!29 = !{!"0xb\005\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [./branch.cc]
!30 = !MDLocation(line: 8, scope: !4)
!31 = !{!32, !32, i64 0}
!32 = !{!"any pointer", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !MDLocation(line: 9, scope: !20)
!36 = !MDLocation(line: 13, scope: !24)
!37 = !MDLocation(line: 14, scope: !24)
!38 = !MDLocation(line: 11, scope: !22)
!39 = !MDLocation(line: 20, scope: !4)
!40 = !MDLocation(line: 21, scope: !4)
!41 = !MDLocation(line: 22, scope: !4)
!42 = !{i32 1, !"Debug Info Version", i32 2}
