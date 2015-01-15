; REQUIRES: object-emission
;
; RUN: llvm-link %s %p/type-unique-simple2-b.ll -S -o - | %llc_dwarf -filetype=obj -O0 | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; Tests for a merge error where attributes are inserted twice into the same DIE.
;
; $ cat ab.h
; typedef int foo_t;
; class A {
; public:
;   virtual void setFoo();
;   virtual const foo_t getFoo();
; };
;
; $ cat a.cpp
; #include "ab.h"
; foo_t bar() {
;     return A().getFoo();
; }
;
; CHECK: _ZN1A6setFooEv
; CHECK: DW_AT_accessibility [DW_FORM_data1]   (DW_ACCESS_public)
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG

; ModuleID = 'a.cpp'

%class.A = type { i32 (...)** }

@_ZTV1A = external unnamed_addr constant [4 x i8*]

; Function Attrs: nounwind
define i32 @_Z3barv() #0 {
entry:
  %tmp = alloca %class.A, align 8
  %0 = bitcast %class.A* %tmp to i8*, !dbg !38
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 8, i32 8, i1 false), !dbg !38
  call void @_ZN1AC1Ev(%class.A* %tmp) #1, !dbg !38
  %call = call i32 @_ZN1A6getFooEv(%class.A* %tmp), !dbg !38
  ret i32 %call, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !39, metadata !{!"0x102"}), !dbg !41
  %this1 = load %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1) #1, !dbg !42
  ret void, !dbg !42
}

declare i32 @_ZN1A6getFooEv(%class.A*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #4

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !44, metadata !{!"0x102"}), !dbg !45
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !46
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !46
  ret void, !dbg !46
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { inlinehint nounwind }
attributes #4 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36}
!llvm.ident = !{!37}

!0 = !{!"0x11\004\00clang version 3.5 \000\00\000\00\000", !1, !2, !3, !26, !2, !2} ; [ DW_TAG_compile_unit ] [/<unknown>] [DW_LANG_C_plus_plus]
!1 = !{!"<unknown>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2\00A\002\0064\0064\000\000\000", !5, null, null, !6, !"_ZTS1A", null, !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 64, align 64, offset 0] [def] [from ]
!5 = !{!"./ab.h", !""}
!6 = !{!7, !14, !19}
!7 = !{!"0xd\00_vptr$A\000\0064\000\000\0064", !5, !8, !9} ; [ DW_TAG_member ] [_vptr$A] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!8 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [/./ab.h]
!9 = !{!"0xf\00\000\0064\000\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!10 = !{!"0xf\00__vtbl_ptr_type\000\0064\000\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!11 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!13}
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !{!"0x2e\00setFoo\00setFoo\00_ZN1A6setFooEv\004\000\000\001\006\00259\000\004", !5, !"_ZTS1A", !15, !"_ZTS1A", null, null, i32 0, !18} ; [ DW_TAG_subprogram ] [line 4] [setFoo]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !17}
!17 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!18 = !{i32 786468}
!19 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\005\000\000\001\006\00259\000\005", !5, !"_ZTS1A", !20, !"_ZTS1A", null, null, i32 0, !25} ; [ DW_TAG_subprogram ] [line 5] [getFoo]
!20 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{!22, !17}
!22 = !{!"0x26\00\000\000\000\000\000", null, null, !23} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo_t]
!23 = !{!"0x16\00foo_t\001\000\000\000\000", !24, null, !13} ; [ DW_TAG_typedef ] [foo_t] [line 1, size 0, align 0, offset 0] [from int]
!24 = !{!"a.cpp", !""}
!25 = !{i32 786468}
!26 = !{!27, !31, !34}
!27 = !{!"0x2e\00bar\00bar\00_Z3barv\002\000\001\000\006\00256\000\002", !24, !28, !29, null, i32 ()* @_Z3barv, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [bar]
!28 = !{!"0x29", !24}        ; [ DW_TAG_file_type ] [/a.cpp]
!29 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !30, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!23}
!31 = !{!"0x2e\00A\00A\00_ZN1AC1Ev\002\000\001\000\006\00320\000\002", !5, !"_ZTS1A", !15, null, void (%class.A*)* @_ZN1AC1Ev, null, !32, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [A]
!32 = !{!"0x2e\00A\00A\00\000\000\000\000\006\00320\000\000", null, !"_ZTS1A", !15, null, null, null, i32 0, !33} ; [ DW_TAG_subprogram ] [line 0] [A]
!33 = !{i32 786468}
!34 = !{!"0x2e\00A\00A\00_ZN1AC2Ev\002\000\001\000\006\00320\000\002", !5, !"_ZTS1A", !15, null, void (%class.A*)* @_ZN1AC2Ev, null, !32, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [A]
!35 = !{i32 2, !"Dwarf Version", i32 2}
!36 = !{i32 1, !"Debug Info Version", i32 2}
!37 = !{!"clang version 3.5 "}
!38 = !MDLocation(line: 3, scope: !27)
!39 = !{!"0x101\00this\0016777216\001088", !31, null, !40} ; [ DW_TAG_arg_variable ] [this] [line 0]
!40 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!41 = !MDLocation(line: 0, scope: !31)
!42 = !MDLocation(line: 2, scope: !43)
!43 = !{!"0xb\000", !5, !31} ; [ DW_TAG_lexical_block ] [/./ab.h]
!44 = !{!"0x101\00this\0016777216\001088", !34, null, !40} ; [ DW_TAG_arg_variable ] [this] [line 0]
!45 = !MDLocation(line: 0, scope: !34)
!46 = !MDLocation(line: 2, scope: !34)
