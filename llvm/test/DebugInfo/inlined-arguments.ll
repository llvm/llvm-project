; REQUIRES: object-emission

; RUN: llc -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -O -g with the following source
;
; void f1(int x, int y);
; void f3(int line);
; void f2() {
;   f1(1, 2);
; }
; void f1(int x, int y) {
;   f3(y);
; }

; CHECK: DW_AT_name{{.*}}"f1"
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name{{.*}}"x"
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name{{.*}}"y"

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
  tail call void @llvm.dbg.value(metadata !15, i64 0, metadata !16), !dbg !18
  tail call void @llvm.dbg.value(metadata !19, i64 0, metadata !20), !dbg !18
  tail call void @_Z2f3i(i32 2), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: uwtable
define void @_Z2f1ii(i32 %x, i32 %y) #0 {
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !13), !dbg !23
  tail call void @llvm.dbg.value(metadata !{i32 %y}, i64 0, metadata !14), !dbg !23
  tail call void @_Z2f3i(i32 %y), !dbg !24
  ret void, !dbg !25
}

declare void @_Z2f3i(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"exp.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f2", metadata !"f2", metadata !"_Z2f2v", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @_Z2f2v, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [f2]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f1", metadata !"f1", metadata !"_Z2f1ii", i32 6, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32, i32)* @_Z2f1ii, null, null, metadata !12, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [f1]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11, metadata !11}
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !14}
!13 = metadata !{i32 786689, metadata !8, metadata !"x", metadata !5, i32 16777222, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [x] [line 6]
!14 = metadata !{i32 786689, metadata !8, metadata !"y", metadata !5, i32 33554438, metadata !11, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [y] [line 6]
!15 = metadata !{i32 undef}
!16 = metadata !{i32 786689, metadata !8, metadata !"x", metadata !5, i32 16777222, metadata !11, i32 0, metadata !17} ; [ DW_TAG_arg_variable ] [x] [line 6]
!17 = metadata !{i32 4, i32 0, metadata !4, null}
!18 = metadata !{i32 6, i32 0, metadata !8, metadata !17}
!19 = metadata !{i32 2}
!20 = metadata !{i32 786689, metadata !8, metadata !"y", metadata !5, i32 33554438, metadata !11, i32 0, metadata !17} ; [ DW_TAG_arg_variable ] [y] [line 6]
!21 = metadata !{i32 7, i32 0, metadata !8, metadata !17}
!22 = metadata !{i32 5, i32 0, metadata !4, null}
!23 = metadata !{i32 6, i32 0, metadata !8, null}
!24 = metadata !{i32 7, i32 0, metadata !8, null}
!25 = metadata !{i32 8, i32 0, metadata !8, null} ; [ DW_TAG_imported_declaration ]
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
