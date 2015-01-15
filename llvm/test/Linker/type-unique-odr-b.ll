; RUN: true
; This file belongs to type-unique-odr-a.ll.
;
; Test ODR-based type uniquing for C++ class members.
; rdar://problem/15851313.
;
; $ cat -n type-unique-odr-b.cpp
;     1	// Make this declaration start on a different line.
;     2	class A {
;     3	  int data;
;     4	protected:
;     5	  void getFoo();
;     6	};
;     7
;     8	void A::getFoo() {}
;     9
;    10	static void bar() {}
;    11	void f() { bar(); };

; ModuleID = 'type-unique-odr-b.cpp'

%class.A = type { i32 }

; Function Attrs: nounwind
define void @_ZN1A6getFooEv(%class.A* %this) #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !24, metadata !{!"0x102"}), !dbg !26
  %this1 = load %class.A** %this.addr
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define void @_Z1fv() #0 {
entry:
  call void @_ZL3barv(), !dbg !28
  ret void, !dbg !28
}

; Function Attrs: nounwind
define internal void @_ZL3barv() #0 {
entry:
  ret void, !dbg !29
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !3, !14, !2, !2} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = !{!"<unknown>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2\00A\002\0032\0032\000\000\000", !5, null, null, !6, null, null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"type-unique-odr-b.cpp", !""}
!6 = !{!7, !9}
!7 = !{!"0xd\00data\003\0032\0032\000\001", !5, !"_ZTS1A", !8} ; [ DW_TAG_member ] [data] [line 3, size 32, align 32, offset 0] [private] [from int]
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\005\000\000\000\006\00258\000\005", !5, !"_ZTS1A", !10, null, null, null, i32 0, !13} ; [ DW_TAG_subprogram ] [line 5] [protected] [getFoo]
!10 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{null, !12}
!12 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!13 = !{i32 786468}
!14 = !{!15, !16, !20}
!15 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\008\000\001\000\006\00256\000\008", !5, !"_ZTS1A", !10, null, void (%class.A*)* @_ZN1A6getFooEv, null, !9, !2} ; [ DW_TAG_subprogram ] [line 8] [def] [getFoo]
!16 = !{!"0x2e\00f\00f\00_Z1fv\0011\000\001\000\006\00256\000\0011", !5, !17, !18, null, void ()* @_Z1fv, null, null, !2} ; [ DW_TAG_subprogram ] [line 11] [def] [f]
!17 = !{!"0x29", !5}         ; [ DW_TAG_file_type ] [type-unique-odr-b.cpp]
!18 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = !{null}
!20 = !{!"0x2e\00bar\00bar\00_ZL3barv\0010\001\001\000\006\00256\000\0010", !5, !17, !18, null, void ()* @_ZL3barv, null, null, !2} ; [ DW_TAG_subprogram ] [line 10] [local] [def] [bar]
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 1, !"Debug Info Version", i32 2}
!23 = !{!"clang version 3.5.0 "}
!24 = !{!"0x101\00this\0016777216\001088", !15, null, !25} ; [ DW_TAG_arg_variable ] [this] [line 0]
!25 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!26 = !MDLocation(line: 0, scope: !15)
!27 = !MDLocation(line: 8, scope: !15)
!28 = !MDLocation(line: 11, scope: !16)
!29 = !MDLocation(line: 10, scope: !20)
