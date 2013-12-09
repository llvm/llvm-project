; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-unknown-unknown -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -g with the following source:
; struct foo {
;   foo(const foo&);
;   int i;
; };
;
; void func(foo f, foo g) {
;   f.i++;
; }

; CHECK: debug_info contents
; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name{{.*}}"_Z4func3fooS_"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name{{.*}}"f"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name{{.*}}"g"

%struct.foo = type { i32 }

; Function Attrs: nounwind uwtable
define void @_Z4func3fooS_(%struct.foo* %f, %struct.foo* %g) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.foo* %f}, metadata !19), !dbg !20
  call void @llvm.dbg.declare(metadata !{%struct.foo* %g}, metadata !21), !dbg !20
  %i = getelementptr inbounds %struct.foo* %f, i32 0, i32 0, !dbg !22
  %0 = load i32* %i, align 4, !dbg !22
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, i32* %i, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"scratch.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func", metadata !"func", metadata !"_Z4func3fooS_", i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.foo*, %struct.foo*)* @_Z4func3fooS_, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !8}
!8 = metadata !{i32 786451, metadata !1, null, metadata !"foo", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !12}
!10 = metadata !{i32 786445, metadata !1, metadata !8, metadata !"i", i32 3, i64 32, i64 32, i64 0, i32 0, metadata !11} ; [ DW_TAG_member ] [i] [line 3, size 32, align 32, offset 0] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"foo", metadata !"foo", metadata !"", i32 2, metadata !13, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !18, i32 2} ; [ DW_TAG_subprogram ] [line 2] [foo]
!13 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{null, metadata !15, metadata !16}
!15 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo]
!16 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo]
!18 = metadata !{i32 786468}
!19 = metadata !{i32 786689, metadata !4, metadata !"f", metadata !5, i32 16777222, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [f] [line 6]
!20 = metadata !{i32 6, i32 0, metadata !4, null}
!21 = metadata !{i32 786689, metadata !4, metadata !"g", metadata !5, i32 33554438, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [g] [line 6]
!22 = metadata !{i32 7, i32 0, metadata !4, null}
!23 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
