; RUN: sed -e 's|PATTERN|%/T|g' < %s > %t1
; RUN: opt -insert-gcov-profiling -disable-output < %t1
; RUN: rm %T/linezero.gcno %t1

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
  call void @llvm.dbg.declare(metadata %struct.vector** %__range, metadata !27, metadata !{}), !dbg !30
  br label %0

; <label>:0                                       ; preds = %entry
  call void @_Z13TagFieldSpecsv(), !dbg !31
  store %struct.vector* %ref.tmp, %struct.vector** %__range, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata i8** %__begin, metadata !32, metadata !{}), !dbg !30
  %1 = load %struct.vector** %__range, align 8, !dbg !31
  %call = call i8* @_ZN6vector5beginEv(%struct.vector* %1), !dbg !31
  store i8* %call, i8** %__begin, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata i8** %__end, metadata !33, metadata !{}), !dbg !30
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
  call void @llvm.dbg.declare(metadata i8* %spec, metadata !37, metadata !{}), !dbg !31
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
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

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

!0 = !{!"0x11\004\00clang version 3.5.0 (trunk 209871)\000\00\000\00\001", !1, !2, !3, !14, !2, !2} ; [ DW_TAG_compile_unit ] [<stdin>] [DW_LANG_C_plus_plus]
!1 = !{!"<stdin>", !"PATTERN"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00vector\0021\008\008\000\000\000", !5, null, null, !6, null, null, !"_ZTS6vector"} ; [ DW_TAG_structure_type ] [vector] [line 21, size 8, align 8, offset 0] [def] [from ]
!5 = !{!"linezero.cc", !"PATTERN"}
!6 = !{!7, !13}
!7 = !{!"0x2e\00begin\00begin\00_ZN6vector5beginEv\0025\000\000\000\006\00256\000\0025", !5, !"_ZTS6vector", !8, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 25] [begin]
!8 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = !{!10, !12}
!10 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!11 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!12 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS6vector"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS6vector]
!13 = !{!"0x2e\00end\00end\00_ZN6vector3endEv\0026\000\000\000\006\00256\000\0026", !5, !"_ZTS6vector", !8, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 26] [end]
!14 = !{!15, !20}
!15 = !{!"0x2e\00test\00test\00_Z4testv\0050\000\001\000\006\00256\000\0050", !5, !16, !17, null, i32 ()* @_Z4testv, null, null, !2} ; [ DW_TAG_subprogram ] [line 50] [def] [test]
!16 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [./linezero.cc]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{!19}
!19 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = !{!"0x2e\00f1\00f1\00_Z2f1v\0054\000\001\000\006\00256\000\0054", !5, !16, !21, null, void ()* @_Z2f1v, null, null, !2} ; [ DW_TAG_subprogram ] [line 54] [def] [f1]
!21 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !22, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 2}
!25 = !{!"PATTERN/linezero.o", !0}
!26 = !{!"clang version 3.5.0 (trunk 209871)"}
!27 = !{!"0x100\00__range\000\0064", !28, null, !29} ; [ DW_TAG_auto_variable ] [__range] [line 0]
!28 = !{!"0xb\0051\000\000", !5, !15} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!29 = !{!"0x42\00\000\000\000\000\000", null, null, !"_ZTS6vector"} ; [ DW_TAG_rvalue_reference_type ] [line 0, size 0, align 0, offset 0] [from _ZTS6vector]
!30 = !MDLocation(line: 0, scope: !28)
!31 = !MDLocation(line: 51, scope: !28)
!32 = !{!"0x100\00__begin\000\0064", !28, null, !10} ; [ DW_TAG_auto_variable ] [__begin] [line 0]
!33 = !{!"0x100\00__end\000\0064", !28, null, !10} ; [ DW_TAG_auto_variable ] [__end] [line 0]
!34 = !MDLocation(line: 51, scope: !35)
!35 = !{!"0xb\0051\000\005", !5, !36} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!36 = !{!"0xb\0051\000\001", !5, !28} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!37 = !{!"0x100\00spec\0051\000", !28, !16, !11} ; [ DW_TAG_auto_variable ] [spec] [line 51]
!38 = !MDLocation(line: 51, scope: !39)
!39 = !{!"0xb\0051\000\002", !5, !28} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!40 = !MDLocation(line: 51, scope: !41)
!41 = !{!"0xb\0051\000\004", !5, !28} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!42 = !MDLocation(line: 51, scope: !43)
!43 = !{!"0xb\0051\000\003", !5, !28} ; [ DW_TAG_lexical_block ] [./linezero.cc]
!44 = !MDLocation(line: 52, scope: !15)
!45 = !MDLocation(line: 54, scope: !20)
