; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj < %s > %t
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
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"x"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"y"

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
  tail call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 2, i64 0, metadata !20, metadata !{!"0x102"}), !dbg !18
  tail call void @_Z2f3i(i32 2), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: uwtable
define void @_Z2f1ii(i32 %x, i32 %y) #0 {
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !13, metadata !{!"0x102"}), !dbg !23
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !23
  tail call void @_Z2f3i(i32 %y), !dbg !24
  ret void, !dbg !25
}

declare void @_Z2f3i(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = !{!"0x11\004\00clang version 3.4 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"exp.cpp", !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x2e\00f2\00f2\00_Z2f2v\003\000\001\000\006\00256\001\003", !1, !5, !6, null, void ()* @_Z2f2v, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f2]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x2e\00f1\00f1\00_Z2f1ii\006\000\001\000\006\00256\001\006", !1, !5, !9, null, void (i32, i32)* @_Z2f1ii, null, null, !12} ; [ DW_TAG_subprogram ] [line 6] [def] [f1]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11, !11}
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{!13, !14}
!13 = !{!"0x101\00x\0016777222\000", !8, !5, !11} ; [ DW_TAG_arg_variable ] [x] [line 6]
!14 = !{!"0x101\00y\0033554438\000", !8, !5, !11} ; [ DW_TAG_arg_variable ] [y] [line 6]
!15 = !{i32 undef}
!16 = !{!"0x101\00x\0016777222\000", !8, !5, !11, !17} ; [ DW_TAG_arg_variable ] [x] [line 6]
!17 = !MDLocation(line: 4, scope: !4)
!18 = !MDLocation(line: 6, scope: !8, inlinedAt: !17)
!19 = !{i32 2}
!20 = !{!"0x101\00y\0033554438\000", !8, !5, !11, !17} ; [ DW_TAG_arg_variable ] [y] [line 6]
!21 = !MDLocation(line: 7, scope: !8, inlinedAt: !17)
!22 = !MDLocation(line: 5, scope: !4)
!23 = !MDLocation(line: 6, scope: !8)
!24 = !MDLocation(line: 7, scope: !8)
!25 = !MDLocation(line: 8, scope: !8)
!26 = !{i32 1, !"Debug Info Version", i32 2}
