; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; Test the DW_AT_accessibility DWARF attribute.
;
;
; Regenerate me:
; clang++ -g tools/clang/test/CodeGenCXX/debug-info-access.cpp -S -emit-llvm -o -
;
;   struct A {
;     void pub_default();
;     static int pub_default_static;
;   };
;
;   class B : public A {
;   public:
;     void pub();
;     static int public_static;
;   protected:
;     void prot();
;   private:
;     void priv_default();
;   };
;
;   union U {
;     void union_pub_default();
;   private:
;     int union_priv;
;   };
;
;   void free() {}
;
;   A a;
;   B b;
;   U u;

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"pub_default_static")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_inheritance
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"public_static")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"prot")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_protected)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"priv_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"union_priv")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_private)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"union_pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"free")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; ModuleID = '/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i8 }
%class.B = type { i8 }
%union.U = type { i32 }

@a = global %struct.A zeroinitializer, align 1
@b = global %class.B zeroinitializer, align 1
@u = global %union.U zeroinitializer, align 4

; Function Attrs: nounwind ssp uwtable
define void @_Z4freev() #0 {
  ret void, !dbg !41
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38, !39}
!llvm.ident = !{!40}

!0 = !{!"0x11\004\00clang version 3.6.0 \000\00\000\00\001", !1, !2, !3, !29, !34, !2} ; [ DW_TAG_compile_unit ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", !""}
!2 = !{}
!3 = !{!4, !12, !22}
!4 = !{!"0x13\00A\003\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6, !8}
!6 = !{!"0xd\00pub_default_static\007\000\000\000\004096", !1, !"_ZTS1A", !7, null} ; [ DW_TAG_member ] [pub_default_static] [line 7, size 0, align 0, offset 0] [static] [from int]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!"0x2e\00pub_default\00pub_default\00_ZN1A11pub_defaultEv\005\000\000\000\006\00256\000\005", !1, !"_ZTS1A", !9, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 5] [pub_default]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11}
!11 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = !{!"0x2\00B\0011\008\008\000\000\000", !1, null, null, !13, null, null, !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 11, size 8, align 8, offset 0] [def] [from ]
!13 = !{!14, !15, !16, !20, !21}
!14 = !{!"0x1c\00\000\000\000\000\003", null, !"_ZTS1B", !"_ZTS1A"} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [public] [from _ZTS1A]
!15 = !{!"0xd\00public_static\0016\000\000\000\004099", !1, !"_ZTS1B", !7, null} ; [ DW_TAG_member ] [public_static] [line 16, size 0, align 0, offset 0] [public] [static] [from int]
!16 = !{!"0x2e\00pub\00pub\00_ZN1B3pubEv\0014\000\000\000\006\00259\000\0014", !1, !"_ZTS1B", !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 14] [public] [pub]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19}
!19 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!20 = !{!"0x2e\00prot\00prot\00_ZN1B4protEv\0019\000\000\000\006\00258\000\0019", !1, !"_ZTS1B", !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 19] [protected] [prot]
!21 = !{!"0x2e\00priv_default\00priv_default\00_ZN1B12priv_defaultEv\0022\000\000\000\006\00256\000\0022", !1, !"_ZTS1B", !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 22] [priv_default]
!22 = !{!"0x17\00U\0025\0032\0032\000\000\000", !1, null, null, !23, null, null, !"_ZTS1U"} ; [ DW_TAG_union_type ] [U] [line 25, size 32, align 32, offset 0] [def] [from ]
!23 = !{!24, !25}
!24 = !{!"0xd\00union_priv\0030\0032\0032\000\001", !1, !"_ZTS1U", !7} ; [ DW_TAG_member ] [union_priv] [line 30, size 32, align 32, offset 0] [private] [from int]
!25 = !{!"0x2e\00union_pub_default\00union_pub_default\00_ZN1U17union_pub_defaultEv\0027\000\000\000\006\00256\000\0027", !1, !"_ZTS1U", !26, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 27] [union_pub_default]
!26 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = !{null, !28}
!28 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1U"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1U]
!29 = !{!30}
!30 = !{!"0x2e\00free\00free\00_Z4freev\0035\000\001\000\006\00256\000\0035", !1, !31, !32, null, void ()* @_Z4freev, null, null, !2} ; [ DW_TAG_subprogram ] [line 35] [def] [free]
!31 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp]
!32 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = !{null}
!34 = !{!35, !36, !37}
!35 = !{!"0x34\00a\00a\00\0037\000\001", null, !31, !"_ZTS1A", %struct.A* @a, null} ; [ DW_TAG_variable ] [a] [line 37] [def]
!36 = !{!"0x34\00b\00b\00\0038\000\001", null, !31, !"_ZTS1B", %class.B* @b, null} ; [ DW_TAG_variable ] [b] [line 38] [def]
!37 = !{!"0x34\00u\00u\00\0039\000\001", null, !31, !"_ZTS1U", %union.U* @u, null} ; [ DW_TAG_variable ] [u] [line 39] [def]
!38 = !{i32 2, !"Dwarf Version", i32 2}
!39 = !{i32 2, !"Debug Info Version", i32 2}
!40 = !{!"clang version 3.6.0 "}
!41 = !MDLocation(line: 35, column: 14, scope: !30)
