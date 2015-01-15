; REQUIRES: object-emission
;
; RUN: llvm-link %s %p/type-unique-odr-b.ll -S -o - | %llc_dwarf -filetype=obj -O0 | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; Test ODR-based type uniquing for C++ class members.
; rdar://problem/15851313.
;
; $ cat -n type-unique-odr-a.cpp
;     1	class A {
;     2	  int data;
;     3	protected:
;     4	  void getFoo();
;     5	};
;     6
;     7	static void bar() {
;     8	  A a;
;     9	}
;    10
;    11	void baz() { bar(); }
;; #include "ab.h"
; foo_t bar() {
;     return A().getFoo();
; }
;
; CHECK:      DW_TAG_class_type
; CHECK-NEXT:   DW_AT_name {{.*}} "A"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_member
; CHECK-NEXT:   DW_AT_name {{.*}} "data"
; CHECK-NOT:  DW_TAG
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN1A6getFooEv"
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "getFoo"
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_Z3bazv"
; CHECK:      DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZL3barv"

; getFoo and A may only appear once.
; CHECK-NOT:  AT_name{{.*(getFoo)|("A")}}


; ModuleID = 'type-unique-odr-a.cpp'

%class.A = type { i32 }

; Function Attrs: nounwind
define void @_Z3bazv() #0 {
entry:
  call void @_ZL3barv(), !dbg !23
  ret void, !dbg !23
}

; Function Attrs: nounwind
define internal void @_ZL3barv() #0 {
entry:
  %a = alloca %class.A, align 4
  call void @llvm.dbg.declare(metadata %class.A* %a, metadata !24, metadata !{!"0x102"}), !dbg !25
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !3, !14, !2, !2} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = !{!"<unknown>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2\00A\001\0032\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"type-unique-odr-a.cpp", !""}
!6 = !{!7, !9}
!7 = !{!"0xd\00data\002\0032\0032\000\001", !5, !"_ZTS1A", !8} ; [ DW_TAG_member ] [data] [line 2, size 32, align 32, offset 0] [private] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\004\000\000\000\006\00258\000\004", !5, !"_ZTS1A", !10, null, null, null, i32 0, !13} ; [ DW_TAG_subprogram ] [line 4] [protected] [getFoo]
!10 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null, !12}
!12 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!13 = !{i32 786468}
!14 = !{!15, !19}
!15 = !{!"0x2e\00baz\00baz\00_Z3bazv\0011\000\001\000\006\00256\000\0011", !5, !16, !17, null, void ()* @_Z3bazv, null, null, !2} ; [ DW_TAG_subprogram ] [line 11] [def] [baz]
!16 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [type-unique-odr-a.cpp]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null}
!19 = !{!"0x2e\00bar\00bar\00_ZL3barv\007\001\001\000\006\00256\000\007", !5, !16, !17, null, void ()* @_ZL3barv, null, null, !2} ; [ DW_TAG_subprogram ] [line 7] [local] [def] [bar]
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 1, !"Debug Info Version", i32 2}
!22 = !{!"clang version 3.5.0 "}
!23 = !MDLocation(line: 11, scope: !15)
!24 = !{!"0x100\00a\008\000", !19, !16, !"_ZTS1A"} ; [ DW_TAG_auto_variable ] [a] [line 8]
!25 = !MDLocation(line: 8, scope: !19)
!26 = !MDLocation(line: 9, scope: !19)
