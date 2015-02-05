; RUN: opt %loadPolly -polly-detect -polly-report -disable-output < %s  2>&1 | FileCheck %s
target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo(float* %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body, !dbg !11

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %i.01 = trunc i64 %indvar to i32, !dbg !13
  %arrayidx = getelementptr float* %A, i64 %indvar, !dbg !13
  %conv = sitofp i32 %i.01 to float, !dbg !13
  store float %conv, float* %arrayidx, align 4, !dbg !13
  %indvar.next = add i64 %indvar, 1, !dbg !11
  %exitcond = icmp ne i64 %indvar.next, 100, !dbg !11
  br i1 %exitcond, label %for.body, label %for.end, !dbg !11

for.end:                                          ; preds = %for.body
  ret void, !dbg !14
}

; CHECK: note: Polly detected an optimizable loop region (scop) in function 'foo'
; CHECK: test.c:2: Start of scop
; CHECK: test.c:3: End of scop

; Function Attrs: nounwind uwtable
define void @bar(float* %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body, !dbg !15

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %i.01 = trunc i64 %indvar to i32, !dbg !17
  %arrayidx = getelementptr float* %A, i64 %indvar, !dbg !17
  %conv = sitofp i32 %i.01 to float, !dbg !17
  store float %conv, float* %arrayidx, align 4, !dbg !17
  %indvar.next = add i64 %indvar, 1, !dbg !15
  %exitcond = icmp ne i64 %indvar.next, 100, !dbg !15
  br i1 %exitcond, label %for.body, label %for.end, !dbg !15

for.end:                                          ; preds = %for.body
  ret void, !dbg !18
}

; CHECK: note: Polly detected an optimizable loop region (scop) in function 'bar'
; CHECK: test.c:9: Start of scop
; CHECK: test.c:13: End of scop

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !{!"0x11\0012\00clang version 3.5 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"/home/grosser/Projects/polly/git/tools/polly"}
!2 = !{i32 0}
!3 = !{!4, !7}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (float*)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!"0x2e\00bar\00bar\00\006\000\001\000\006\00256\000\006", !1, !5, !6, null, void (float*)* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [bar]
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 2}
!10 = !{!"clang version 3.5 "}
!11 = !MDLocation(line: 2, scope: !12)
!12 = !{!"0xb\002\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test.c]
!13 = !MDLocation(line: 3, scope: !12)
!14 = !MDLocation(line: 4, scope: !4)
!15 = !MDLocation(line: 9, scope: !16)
!16 = !{!"0xb\009\000\001", !1, !7} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test.c]
!17 = !MDLocation(line: 13, scope: !16)
!18 = !MDLocation(line: 14, scope: !7)

