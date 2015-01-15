; RUN: true
; This file belongs to type-unique-simple2-a.ll.
;
; $ cat b.cpp
; #include "ab.h"
; void A::setFoo() {}
; const
; foo_t A::getFoo() { return 1; }
; ModuleID = 'b.cpp'
; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
; target triple = "x86_64-apple-macosx10.9.0"

%class.A = type { i32 (...)** }

@_ZTV1A = unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1A6setFooEv to i8*), i8* bitcast (i32 (%class.A*)* @_ZN1A6getFooEv to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = constant [3 x i8] c"1A\00"
@_ZTI1A = unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1A, i32 0, i32 0) }

; Function Attrs: nounwind
define void @_ZN1A6setFooEv(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !32, metadata !{!"0x102"}), !dbg !34
  %this1 = load %class.A** %this.addr
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define i32 @_ZN1A6getFooEv(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !36, metadata !{!"0x102"}), !dbg !37
  %this1 = load %class.A** %this.addr
  ret i32 1, !dbg !38
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = !{!"0x11\004\00clang version 3.5 \000\00\000\00\000", !1, !2, !3, !25, !2, !2} ; [ DW_TAG_compile_unit ] [/<unknown>] [DW_LANG_C_plus_plus]
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
!19 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\005\000\000\001\006\00259\000\005", !5, !"_ZTS1A", !20, !"_ZTS1A", null, null, i32 0, !24} ; [ DW_TAG_subprogram ] [line 5] [getFoo]
!20 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = !{!22, !17}
!22 = !{!"0x26\00\000\000\000\000\000", null, null, !23} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo_t]
!23 = !{!"0x16\00foo_t\001\000\000\000\000", !5, null, !13} ; [ DW_TAG_typedef ] [foo_t] [line 1, size 0, align 0, offset 0] [from int]
!24 = !{i32 786468}
!25 = !{!26, !28}
!26 = !{!"0x2e\00setFoo\00setFoo\00_ZN1A6setFooEv\002\000\001\000\006\00259\000\002", !27, !"_ZTS1A", !15, null, void (%class.A*)* @_ZN1A6setFooEv, null, !14, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [setFoo]
!27 = !{!"b.cpp", !""}
!28 = !{!"0x2e\00getFoo\00getFoo\00_ZN1A6getFooEv\004\000\001\000\006\00259\000\004", !27, !"_ZTS1A", !20, null, i32 (%class.A*)* @_ZN1A6getFooEv, null, !19, !2} ; [ DW_TAG_subprogram ] [line 4] [def] [getFoo]
!29 = !{i32 2, !"Dwarf Version", i32 2}
!30 = !{i32 1, !"Debug Info Version", i32 2}
!31 = !{!"clang version 3.5 "}
!32 = !{!"0x101\00this\0016777216\001088", !26, null, !33} ; [ DW_TAG_arg_variable ] [this] [line 0]
!33 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!34 = !MDLocation(line: 0, scope: !26)
!35 = !MDLocation(line: 2, scope: !26)
!36 = !{!"0x101\00this\0016777216\001088", !28, null, !33} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = !MDLocation(line: 0, scope: !28)
!38 = !MDLocation(line: 4, scope: !28)
