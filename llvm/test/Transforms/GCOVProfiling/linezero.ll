; RUN: sed -e 's|PATTERN|%T|g' < %s > %t1
; RUN: opt -insert-gcov-profiling -disable-output < %t1
; RUN: rm %T/linezero.gcno %t1
; REQUIRES: shell

; This is a crash test.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.vector = type { i8 }

; Function Attrs: nounwind
define i32 @_Z4testv() #0 {
entry:
  %retval = alloca i32, align 4
  %__range = alloca %struct.vector*, align 8
  %ref.tmp = alloca %struct.vector, align 1
  %undef.agg.tmp = alloca %struct.vector, align 1
  %__begin = alloca i8*, align 8
  %__end = alloca i8*, align 8
  %spec = alloca i8, align 1
  call void @llvm.dbg.declare(metadata !{%struct.vector** %__range}, metadata !27), !dbg !30
  br label %0

; <label>:0                                       ; preds = %entry
  call void @_Z13TagFieldSpecsv(), !dbg !31
  store %struct.vector* %ref.tmp, %struct.vector** %__range, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata !{i8** %__begin}, metadata !32), !dbg !30
  %1 = load %struct.vector** %__range, align 8, !dbg !31
  %call = call i8* @_ZN6vector5beginEv(%struct.vector* %1), !dbg !31
  store i8* %call, i8** %__begin, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata !{i8** %__end}, metadata !33), !dbg !30
  %2 = load %struct.vector** %__range, align 8, !dbg !31
  %call1 = call i8* @_ZN6vector3endEv(%struct.vector* %2), !dbg !31
  store i8* %call1, i8** %__end, align 8, !dbg !31
  br label %for.cond, !dbg !31

for.cond:                                         ; preds = %for.inc, %0
  %3 = load i8** %__begin, align 8, !dbg !34
  %4 = load i8** %__end, align 8, !dbg !34
  %cmp = icmp ne i8* %3, %4, !dbg !34
  br i1 %cmp, label %for.body, label %for.end, !dbg !34

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.declare(metadata !{i8* %spec}, metadata !37), !dbg !31
  %5 = load i8** %__begin, align 8, !dbg !38
  %6 = load i8* %5, align 1, !dbg !38
  store i8 %6, i8* %spec, align 1, !dbg !38
  br label %for.inc, !dbg !38

for.inc:                                          ; preds = %for.body
  %7 = load i8** %__begin, align 8, !dbg !40
  %incdec.ptr = getelementptr inbounds i8* %7, i32 1, !dbg !40
  store i8* %incdec.ptr, i8** %__begin, align 8, !dbg !40
  br label %for.cond, !dbg !40

for.end:                                          ; preds = %for.cond
  call void @llvm.trap(), !dbg !42
  unreachable, !dbg !42

return:                                           ; No predecessors!
  %8 = load i32* %retval, !dbg !44
  ret i32 %8, !dbg !44
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @_Z13TagFieldSpecsv() #2

declare i8* @_ZN6vector5beginEv(%struct.vector*) #2

declare i8* @_ZN6vector3endEv(%struct.vector*) #2

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #3

; Function Attrs: nounwind
define void @_Z2f1v() #0 {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  ret void, !dbg !45
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.gcov = !{!25}
!llvm.ident = !{!26}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 209871)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !"PATTERN"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !5, null, metadata !"vector", i32 21, i64 8, i64 8, i32 0, i32 0, null, metadata !6, i32 0, null, null, metadata !"_ZTS6vector"} ; [ DW_TAG_structure_type ] [vector] [line 21, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !"linezero.cc", metadata !"PATTERN"}
!6 = metadata !{metadata !7, metadata !13}
!7 = metadata !{i32 786478, metadata !5, metadata !"_ZTS6vector", metadata !"begin", metadata !"begin", metadata !"_ZN6vector5beginEv", i32 25, metadata !8, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 25} ; [ DW_TAG_subprogram ] [line 25] [begin]
!8 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{metadata !10, metadata !12}
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!11 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS6vector"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS6vector]
!13 = metadata !{i32 786478, metadata !5, metadata !"_ZTS6vector", metadata !"end", metadata !"end", metadata !"_ZN6vector3endEv", i32 26, metadata !8, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 26} ; [ DW_TAG_subprogram ] [line 26] [end]
!14 = metadata !{metadata !15, metadata !20}
!15 = metadata !{i32 786478, metadata !5, metadata !16, metadata !"test", metadata !"test", metadata !"_Z4testv", i32 50, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z4testv, null, null, metadata !2, i32 50} ; [ DW_TAG_subprogram ] [line 50] [def] [test]
!16 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [./linezero.cc]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = metadata !{i32 786478, metadata !5, metadata !16, metadata !"f1", metadata !"f1", metadata !"_Z2f1v", i32 54, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z2f1v, null, null, metadata !2, i32 54} ; [ DW_TAG_subprogram ] [line 54] [def] [f1]
!21 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !22, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = metadata !{null}
!23 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!24 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!25 = metadata !{metadata !"PATTERN/linezero.o", metadata !0}
!26 = metadata !{metadata !"clang version 3.5.0 (trunk 209871)"}
!27 = metadata !{i32 786688, metadata !28, metadata !"__range", null, i32 0, metadata !29, i32 64, i32 0} ; [ DW_TAG_auto_variable ] [__range] [line 0]
!28 = metadata !{i32 786443, metadata !5, metadata !15, i32 51, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!29 = metadata !{i32 786498, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS6vector"} ; [ DW_TAG_rvalue_reference_type ] [line 0, size 0, align 0, offset 0] [from _ZTS6vector]
!30 = metadata !{i32 0, i32 0, metadata !28, null}
!31 = metadata !{i32 51, i32 0, metadata !28, null}
!32 = metadata !{i32 786688, metadata !28, metadata !"__begin", null, i32 0, metadata !10, i32 64, i32 0} ; [ DW_TAG_auto_variable ] [__begin] [line 0]
!33 = metadata !{i32 786688, metadata !28, metadata !"__end", null, i32 0, metadata !10, i32 64, i32 0} ; [ DW_TAG_auto_variable ] [__end] [line 0]
!34 = metadata !{i32 51, i32 0, metadata !35, null}
!35 = metadata !{i32 786443, metadata !5, metadata !36, i32 51, i32 0, i32 5, i32 5} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!36 = metadata !{i32 786443, metadata !5, metadata !28, i32 51, i32 0, i32 1, i32 1} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!37 = metadata !{i32 786688, metadata !28, metadata !"spec", metadata !16, i32 51, metadata !11, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [spec] [line 51]
!38 = metadata !{i32 51, i32 0, metadata !39, null}
!39 = metadata !{i32 786443, metadata !5, metadata !28, i32 51, i32 0, i32 2, i32 2} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!40 = metadata !{i32 51, i32 0, metadata !41, null}
!41 = metadata !{i32 786443, metadata !5, metadata !28, i32 51, i32 0, i32 4, i32 4} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!42 = metadata !{i32 51, i32 0, metadata !43, null}
!43 = metadata !{i32 786443, metadata !5, metadata !28, i32 51, i32 0, i32 3, i32 3} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!44 = metadata !{i32 52, i32 0, metadata !15, null}
!45 = metadata !{i32 54, i32 0, metadata !20, null}
