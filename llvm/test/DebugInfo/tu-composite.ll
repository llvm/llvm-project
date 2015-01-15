; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: [[TYPE:.*]]: DW_TAG_structure_type
; Make sure we correctly handle containing type of a struct being a type identifier.
; CHECK-NEXT: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]})
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}= "C")

; Make sure we correctly handle context of a subprogram being a type identifier.
; CHECK: [[SP:.*]]: DW_TAG_subprogram
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "foo")
; Make sure we correctly handle containing type of a subprogram being a type identifier.
; CHECK: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]})
; CHECK: DW_TAG_formal_parameter
; CHECK: NULL
; CHECK: NULL

; CHECK: [[TYPE2:.*]]: DW_TAG_structure_type
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "bar")
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "D")
; CHECK: DW_TAG_member
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "a") 
; Make sure we correctly handle context of a struct being a type identifier.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}= "Nested")
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}= "Nested2")
; CHECK-NEXT: DW_AT_declaration [DW_FORM_flag]      (0x01)
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}= "virt<bar>")
; Make sure we correctly handle type of a template_type being a type identifier.
; CHECK: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]})
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*}}= "T")
; Make sure we correctly handle derived-from of a typedef being a type identifier.
; CHECK: DW_TAG_typedef
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]})
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "baz2")
; Make sure we correctly handle derived-from of a pointer type being a type identifier.
; CHECK: DW_TAG_pointer_type
; CHECK: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE]]})
; CHECK: DW_TAG_typedef
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]})
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "baz")
; Make sure we correctly handle derived-from of an array type being a type identifier.
; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE2]]})
; IR generated from clang -g with the following source:
; struct C {
;   virtual void foo();
; };
; void C::foo() {
; }
;
; struct bar { };
; typedef bar baz;
; struct D {
;   typedef bar baz2;
;   static int a;
;   struct Nested { };
;   struct Nested2 { };
;   template <typename T>
;   struct virt {
;     T* values;
;   };
; };
; void test() {
;   baz B;
;   bar A[3];
;   D::baz2 B2;
;   D::Nested e;
;   D::Nested2 *p;
;   D::virt<bar> t;
; }

%struct.C = type { i32 (...)** }
%struct.bar = type { i8 }
%"struct.D::Nested" = type { i8 }
%"struct.D::Nested2" = type { i8 }
%"struct.D::virt" = type { %struct.bar* }

@_ZTV1C = unnamed_addr constant [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1C to i8*), i8* bitcast (void (%struct.C*)* @_ZN1C3fooEv to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1C = constant [3 x i8] c"1C\00"
@_ZTI1C = unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1C, i32 0, i32 0) }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1C3fooEv(%struct.C* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !36, metadata !{!"0x102"}), !dbg !38
  %this1 = load %struct.C** %this.addr
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_Z4testv() #0 {
entry:
  %B = alloca %struct.bar, align 1
  %A = alloca [3 x %struct.bar], align 1
  %B2 = alloca %struct.bar, align 1
  %e = alloca %"struct.D::Nested", align 1
  %p = alloca %"struct.D::Nested2"*, align 8
  %t = alloca %"struct.D::virt", align 8
  call void @llvm.dbg.declare(metadata %struct.bar* %B, metadata !40, metadata !{!"0x102"}), !dbg !42
  call void @llvm.dbg.declare(metadata [3 x %struct.bar]* %A, metadata !43, metadata !{!"0x102"}), !dbg !47
  call void @llvm.dbg.declare(metadata %struct.bar* %B2, metadata !48, metadata !{!"0x102"}), !dbg !50
  call void @llvm.dbg.declare(metadata %"struct.D::Nested"* %e, metadata !51, metadata !{!"0x102"}), !dbg !52
  call void @llvm.dbg.declare(metadata %"struct.D::Nested2"** %p, metadata !53, metadata !{!"0x102"}), !dbg !55
  call void @llvm.dbg.declare(metadata %"struct.D::virt"* %t, metadata !56, metadata !{!"0x102"}), !dbg !57
  ret void, !dbg !58
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !59}

!0 = !{!"0x11\004\00clang version 3.4\000\00\000\00\000", !1, !2, !3, !30, !2, !2} ; [ DW_TAG_compile_unit ] [tmp.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"tmp.cpp", !"."}
!2 = !{}
!3 = !{!4, !18, !19, !22, !23, !24}
!4 = !{!"0x13\00C\001\0064\0064\000\000\000", !1, null, null, !5, !"_ZTS1C", null, !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 64, align 64, offset 0] [def] [from ]
!5 = !{!6, !13}
!6 = !{!"0xd\00_vptr$C\000\0064\000\000\0064", !1, !7, !8} ; [ DW_TAG_member ] [_vptr$C] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!7 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [tmp.cpp]
!8 = !{!"0xf\00\000\0064\000\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!9 = !{!"0xf\00__vtbl_ptr_type\000\0064\000\000\000", null, null, !10} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!10 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = !{!12}
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{!"0x2e\00foo\00foo\00_ZN1C3fooEv\002\000\000\001\006\00256\000\002", !1, !"_ZTS1C", !14, !"_ZTS1C", null, null, i32 0, !17} ; [ DW_TAG_subprogram ] [line 2] [foo]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{null, !16}
!16 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!17 = !{i32 786468}
!18 = !{!"0x13\00bar\007\008\008\000\000\000", !1, null, null, !2, null, null, !"_ZTS3bar"} ; [ DW_TAG_structure_type ] [bar] [line 7, size 8, align 8, offset 0] [def] [from ]
!19 = !{!"0x13\00D\009\008\008\000\000\000", !1, null, null, !20, null, null, !"_ZTS1D"} ; [ DW_TAG_structure_type ] [D] [line 9, size 8, align 8, offset 0] [def] [from ]
!20 = !{!21}
!21 = !{!"0xd\00a\0011\000\000\000\004096", !1, !"_ZTS1D", !12, null} ; [ DW_TAG_member ] [a] [line 11, size 0, align 0, offset 0] [static] [from int]
!22 = !{!"0x13\00Nested\0012\008\008\000\000\000", !1, !"_ZTS1D", null, !2, null, null, !"_ZTSN1D6NestedE"} ; [ DW_TAG_structure_type ] [Nested] [line 12, size 8, align 8, offset 0] [def] [from ]
!23 = !{!"0x13\00Nested2\0013\000\000\000\004\000", !1, !"_ZTS1D", null, null, null, null, !"_ZTSN1D7Nested2E"} ; [ DW_TAG_structure_type ] [Nested2] [line 13, size 0, align 0, offset 0] [decl] [from ]
!24 = !{!"0x13\00virt<bar>\0015\0064\0064\000\000\000", !1, !"_ZTS1D", null, !25, null, !28, !"_ZTSN1D4virtI3barEE"} ; [ DW_TAG_structure_type ] [virt<bar>] [line 15, size 64, align 64, offset 0] [def] [from ]
!25 = !{!26}
!26 = !{!"0xd\00values\0016\0064\0064\000\000", !1, !"_ZTSN1D4virtI3barEE", !27} ; [ DW_TAG_member ] [values] [line 16, size 64, align 64, offset 0] [from ]
!27 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS3bar"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3bar]
!28 = !{!29}
!29 = !{!"0x2f\00T\000\000", null, !"_ZTS3bar", null} ; [ DW_TAG_template_type_parameter ]
!30 = !{!31, !32}
!31 = !{!"0x2e\00foo\00foo\00_ZN1C3fooEv\004\000\001\000\006\00256\000\004", !1, null, !14, null, void (%struct.C*)* @_ZN1C3fooEv, null, !13, !2} ; [ DW_TAG_subprogram ] [line 4] [def] [foo]
!32 = !{!"0x2e\00test\00test\00_Z4testv\0020\000\001\000\006\00256\000\0020", !1, !7, !33, null, void ()* @_Z4testv, null, null, !2} ; [ DW_TAG_subprogram ] [line 20] [def] [test]
!33 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !34, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = !{null}
!35 = !{i32 2, !"Dwarf Version", i32 2}
!36 = !{!"0x101\00this\0016777216\001088", !31, null, !37} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!38 = !MDLocation(line: 0, scope: !31)
!39 = !MDLocation(line: 5, scope: !31)
!40 = !{!"0x100\00B\0021\000", !32, !7, !41} ; [ DW_TAG_auto_variable ] [B] [line 21]
!41 = !{!"0x16\00baz\008\000\000\000\000", !1, null, !"_ZTS3bar"} ; [ DW_TAG_typedef ] [baz] [line 8, size 0, align 0, offset 0] [from _ZTS3bar]
!42 = !MDLocation(line: 21, scope: !32)
!43 = !{!"0x100\00A\0022\000", !32, !7, !44} ; [ DW_TAG_auto_variable ] [A] [line 22]
!44 = !{!"0x1\00\000\0024\008\000\000", null, null, !"_ZTS3bar", !45, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 24, align 8, offset 0] [from _ZTS3bar]
!45 = !{!46}
!46 = !{!"0x21\000\003"}        ; [ DW_TAG_subrange_type ] [0, 2]
!47 = !MDLocation(line: 22, scope: !32)
!48 = !{!"0x100\00B2\0023\000", !32, !7, !49} ; [ DW_TAG_auto_variable ] [B2] [line 23]
!49 = !{!"0x16\00baz2\0010\000\000\000\000", !1, !"_ZTS1D", !"_ZTS3bar"} ; [ DW_TAG_typedef ] [baz2] [line 10, size 0, align 0, offset 0] [from _ZTS3bar]
!50 = !MDLocation(line: 23, scope: !32)
!51 = !{!"0x100\00e\0024\000", !32, !7, !22} ; [ DW_TAG_auto_variable ] [e] [line 24]
!52 = !MDLocation(line: 24, scope: !32)
!53 = !{!"0x100\00p\0025\000", !32, !7, !54} ; [ DW_TAG_auto_variable ] [p] [line 25]
!54 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTSN1D7Nested2E"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTSN1D7Nested2E]
!55 = !MDLocation(line: 25, scope: !32)
!56 = !{!"0x100\00t\0026\000", !32, !7, !24} ; [ DW_TAG_auto_variable ] [t] [line 26]
!57 = !MDLocation(line: 26, scope: !32)
!58 = !MDLocation(line: 27, scope: !32)
!59 = !{i32 1, !"Debug Info Version", i32 2}
