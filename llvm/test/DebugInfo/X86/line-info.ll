; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: [[FILEID:[0-9]+]]]{{.*}}list0.h
; CHECK: [[FILEID]]      0      1   0  is_stmt{{$}}

; IR generated from clang -g -emit-llvm with the following source:
; list0.h:
; int foo (int x) {
;     return ++x;
; }
; list0.c:
; #include "list0.h"
; int main() {
; }

define i32 @foo(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !14), !dbg !15
  %0 = load i32* %x.addr, align 4, !dbg !16
  %inc = add nsw i32 %0, 1, !dbg !16
  store i32 %inc, i32* %x.addr, align 4, !dbg !16
  ret i32 %inc, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata) #1

define i32 @main() #0 {
entry:
  ret i32 0, !dbg !17
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2,  metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/list0.c] [DW_LANG_C99]
!1 = metadata !{metadata !"list0.c", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !10}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @foo, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"./list0.h", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/./list0.h]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786478, metadata !1, metadata !11, metadata !"main", metadata !"main", metadata !"", i32 2, metadata !12, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [main]
!11 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/list0.c]
!12 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{metadata !9}
!14 = metadata !{i32 786689, metadata !4, metadata !"x", metadata !6, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [x] [line 1]
!15 = metadata !{i32 1, i32 0, metadata !4, null}
!16 = metadata !{i32 2, i32 0, metadata !4, null}
!17 = metadata !{i32 3, i32 0, metadata !18, null}
!18 = metadata !{i32 786443, metadata !11, metadata !10} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/scratch/list0.c]
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
