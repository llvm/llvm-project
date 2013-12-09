; REQUIRES: object-emission

; RUN: llc -filetype=obj -O0 < %s > %t
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
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !36), !dbg !38
  %this1 = load %struct.C** %this.addr
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_Z4testv() #0 {
entry:
  %B = alloca %struct.bar, align 1
  %A = alloca [3 x %struct.bar], align 1
  %B2 = alloca %struct.bar, align 1
  %e = alloca %"struct.D::Nested", align 1
  %p = alloca %"struct.D::Nested2"*, align 8
  %t = alloca %"struct.D::virt", align 8
  call void @llvm.dbg.declare(metadata !{%struct.bar* %B}, metadata !40), !dbg !42
  call void @llvm.dbg.declare(metadata !{[3 x %struct.bar]* %A}, metadata !43), !dbg !47
  call void @llvm.dbg.declare(metadata !{%struct.bar* %B2}, metadata !48), !dbg !50
  call void @llvm.dbg.declare(metadata !{%"struct.D::Nested"* %e}, metadata !51), !dbg !52
  call void @llvm.dbg.declare(metadata !{%"struct.D::Nested2"** %p}, metadata !53), !dbg !55
  call void @llvm.dbg.declare(metadata !{%"struct.D::virt"* %t}, metadata !56), !dbg !57
  ret void, !dbg !58
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !59}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !30, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [tmp.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tmp.cpp", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !18, metadata !19, metadata !22, metadata !23, metadata !24}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !5, i32 0, metadata !"_ZTS1C", null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !13}
!6 = metadata !{i32 786445, metadata !1, metadata !7, metadata !"_vptr$C", i32 0, i64 64, i64 0, i64 0, i32 64, metadata !8} ; [ DW_TAG_member ] [_vptr$C] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!7 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [tmp.cpp]
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!9 = metadata !{i32 786447, null, null, metadata !"__vtbl_ptr_type", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1C", metadata !"foo", metadata !"foo", metadata !"_ZN1C3fooEv", i32 2, metadata !14, i1 false, i1 false, i32 1, i32 0, metadata !"_ZTS1C", i32 256, i1 false, null, null, i32 0, metadata !17, i32 2} ; [ DW_TAG_subprogram ] [line 2] [foo]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null, metadata !16}
!16 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!17 = metadata !{i32 786468}
!18 = metadata !{i32 786451, metadata !1, null, metadata !"bar", i32 7, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTS3bar"} ; [ DW_TAG_structure_type ] [bar] [line 7, size 8, align 8, offset 0] [def] [from ]
!19 = metadata !{i32 786451, metadata !1, null, metadata !"D", i32 9, i64 8, i64 8, i32 0, i32 0, null, metadata !20, i32 0, null, null, metadata !"_ZTS1D"} ; [ DW_TAG_structure_type ] [D] [line 9, size 8, align 8, offset 0] [def] [from ]
!20 = metadata !{metadata !21}
!21 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1D", metadata !"a", i32 11, i64 0, i64 0, i64 0, i32 4096, metadata !12, null} ; [ DW_TAG_member ] [a] [line 11, size 0, align 0, offset 0] [static] [from int]
!22 = metadata !{i32 786451, metadata !1, metadata !"_ZTS1D", metadata !"Nested", i32 12, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null, metadata !"_ZTSN1D6NestedE"} ; [ DW_TAG_structure_type ] [Nested] [line 12, size 8, align 8, offset 0] [def] [from ]
!23 = metadata !{i32 786451, metadata !1, metadata !"_ZTS1D", metadata !"Nested2", i32 13, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, metadata !"_ZTSN1D7Nested2E"} ; [ DW_TAG_structure_type ] [Nested2] [line 13, size 0, align 0, offset 0] [decl] [from ]
!24 = metadata !{i32 786451, metadata !1, metadata !"_ZTS1D", metadata !"virt<bar>", i32 15, i64 64, i64 64, i32 0, i32 0, null, metadata !25, i32 0, null, metadata !28, metadata !"_ZTSN1D4virtI3barEE"} ; [ DW_TAG_structure_type ] [virt<bar>] [line 15, size 64, align 64, offset 0] [def] [from ]
!25 = metadata !{metadata !26}
!26 = metadata !{i32 786445, metadata !1, metadata !"_ZTSN1D4virtI3barEE", metadata !"values", i32 16, i64 64, i64 64, i64 0, i32 0, metadata !27} ; [ DW_TAG_member ] [values] [line 16, size 64, align 64, offset 0] [from ]
!27 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS3bar"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3bar]
!28 = metadata !{metadata !29}
!29 = metadata !{i32 786479, null, metadata !"T", metadata !"_ZTS3bar", null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!30 = metadata !{metadata !31, metadata !32}
!31 = metadata !{i32 786478, metadata !1, null, metadata !"foo", metadata !"foo", metadata !"_ZN1C3fooEv", i32 4, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C3fooEv, null, metadata !13, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [foo]
!32 = metadata !{i32 786478, metadata !1, metadata !7, metadata !"test", metadata !"test", metadata !"_Z4testv", i32 20, metadata !33, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z4testv, null, null, metadata !2, i32 20} ; [ DW_TAG_subprogram ] [line 20] [def] [test]
!33 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !34, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{null}
!35 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!36 = metadata !{i32 786689, metadata !31, metadata !"this", null, i32 16777216, metadata !37, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!38 = metadata !{i32 0, i32 0, metadata !31, null}
!39 = metadata !{i32 5, i32 0, metadata !31, null}
!40 = metadata !{i32 786688, metadata !32, metadata !"B", metadata !7, i32 21, metadata !41, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [B] [line 21]
!41 = metadata !{i32 786454, metadata !1, null, metadata !"baz", i32 8, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS3bar"} ; [ DW_TAG_typedef ] [baz] [line 8, size 0, align 0, offset 0] [from _ZTS3bar]
!42 = metadata !{i32 21, i32 0, metadata !32, null}
!43 = metadata !{i32 786688, metadata !32, metadata !"A", metadata !7, i32 22, metadata !44, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [A] [line 22]
!44 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 24, i64 8, i32 0, i32 0, metadata !"_ZTS3bar", metadata !45, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 24, align 8, offset 0] [from _ZTS3bar]
!45 = metadata !{metadata !46}
!46 = metadata !{i32 786465, i64 0, i64 3}        ; [ DW_TAG_subrange_type ] [0, 2]
!47 = metadata !{i32 22, i32 0, metadata !32, null}
!48 = metadata !{i32 786688, metadata !32, metadata !"B2", metadata !7, i32 23, metadata !49, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [B2] [line 23]
!49 = metadata !{i32 786454, metadata !1, metadata !"_ZTS1D", metadata !"baz2", i32 10, i64 0, i64 0, i64 0, i32 0, metadata !"_ZTS3bar"} ; [ DW_TAG_typedef ] [baz2] [line 10, size 0, align 0, offset 0] [from _ZTS3bar]
!50 = metadata !{i32 23, i32 0, metadata !32, null}
!51 = metadata !{i32 786688, metadata !32, metadata !"e", metadata !7, i32 24, metadata !22, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 24]
!52 = metadata !{i32 24, i32 0, metadata !32, null}
!53 = metadata !{i32 786688, metadata !32, metadata !"p", metadata !7, i32 25, metadata !54, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [p] [line 25]
!54 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTSN1D7Nested2E"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTSN1D7Nested2E]
!55 = metadata !{i32 25, i32 0, metadata !32, null}
!56 = metadata !{i32 786688, metadata !32, metadata !"t", metadata !7, i32 26, metadata !24, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [t] [line 26]
!57 = metadata !{i32 26, i32 0, metadata !32, null}
!58 = metadata !{i32 27, i32 0, metadata !32, null}
!59 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
