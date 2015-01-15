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
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name{{.*}}"_Z4func3fooS_"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"f"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"g"

%struct.foo = type { i32 }

; Function Attrs: nounwind uwtable
define void @_Z4func3fooS_(%struct.foo* %f, %struct.foo* %g) #0 {
entry:
  call void @llvm.dbg.declare(metadata %struct.foo* %f, metadata !19, metadata !{!"0x102"}), !dbg !20
  call void @llvm.dbg.declare(metadata %struct.foo* %g, metadata !21, metadata !{!"0x102"}), !dbg !20
  %i = getelementptr inbounds %struct.foo* %f, i32 0, i32 0, !dbg !22
  %0 = load i32* %i, align 4, !dbg !22
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, i32* %i, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24}

!0 = !{!"0x11\004\00clang version 3.4 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"scratch.cpp", !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00func\00func\00_Z4func3fooS_\006\000\001\000\006\00256\000\006", !1, !5, !6, null, void (%struct.foo*, %struct.foo*)* @_Z4func3fooS_, null, null, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8, !8}
!8 = !{!"0x13\00foo\001\0032\0032\000\000\000", !1, null, null, !9, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = !{!10, !12}
!10 = !{!"0xd\00i\003\0032\0032\000\000", !1, !8, !11} ; [ DW_TAG_member ] [i] [line 3, size 32, align 32, offset 0] [from int]
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{!"0x2e\00foo\00foo\00\002\000\000\000\006\00256\000\002", !1, !8, !13, null, null, null, i32 0, !18} ; [ DW_TAG_subprogram ] [line 2] [foo]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{null, !15, !16}
!15 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo]
!16 = !{!"0x10\00\000\000\000\000\000", null, null, !17} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = !{!"0x26\00\000\000\000\000\000", null, null, !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo]
!18 = !{i32 786468}
!19 = !{!"0x101\00f\0016777222\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [f] [line 6]
!20 = !MDLocation(line: 6, scope: !4)
!21 = !{!"0x101\00g\0033554438\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [g] [line 6]
!22 = !MDLocation(line: 7, scope: !4)
!23 = !MDLocation(line: 8, scope: !4)
!24 = !{i32 1, !"Debug Info Version", i32 2}
