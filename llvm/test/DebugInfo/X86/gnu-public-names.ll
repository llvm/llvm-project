; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections < %s | FileCheck -check-prefix=ASM %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; ModuleID = 'dwarf-public-names.cpp'
;
; Generated from:
;
; struct C {
;   void member_function();
;   static int static_member_function();
;   static int static_member_variable;
; };
;
; int C::static_member_variable = 0;
;
; void C::member_function() {
;   static_member_variable = 0;
; }
;
; int C::static_member_function() {
;   return static_member_variable;
; }
;
; C global_variable;
;
; int global_function() {
;   return -1;
; }
;
; namespace ns {
;   void global_namespace_function() {
;     global_variable.member_function();
;   }
;   int global_namespace_variable = 1;
;   struct D {
;     int A;
;   } d;
; }

; ASM: .section        .debug_gnu_pubnames
; ASM: .byte   32                      # Kind: VARIABLE, EXTERNAL
; ASM-NEXT: .asciz  "global_variable"       # External Name

; ASM: .section        .debug_gnu_pubtypes
; ASM: .byte   16                      # Kind: TYPE, EXTERNAL
; ASM-NEXT: .asciz  "C"                     # External Name

; CHECK: .debug_info contents:
; CHECK: Compile Unit:
; CHECK: DW_AT_GNU_pubnames [DW_FORM_flag_present]   (true)
; CHECK-NOT: DW_AT_GNU_pubtypes [

; CHECK: [[STATIC_MEM_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "static_member_variable"

; CHECK: [[C:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "C"

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_variable"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "member_function"

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name {{.*}} "static_member_function"

; CHECK: [[INT:0x[0-9a-f]+]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} "int"

; CHECK: [[GLOB_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_variable"

; CHECK: [[NS:0x[0-9a-f]+]]: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"

; CHECK: [[GLOB_NS_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "global_namespace_variable"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D_VAR:0x[0-9a-f]+]]: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "d"
; CHECK-NOT: DW_AT_specification
; CHECK: DW_AT_location
; CHECK-NOT: DW_AT_specification

; CHECK: [[D:0x[0-9a-f]+]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "D"

; CHECK: [[GLOB_NS_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_namespace_function"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "f3"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[F3_Z:.*]]:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "z"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[OUTER:.*]]: DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "outer"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[OUTER_ANON:.*]]:  DW_TAG_namespace
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK-NOT:     DW_AT_name
; CHECK: [[OUTER_ANON_C:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "c"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: We probably shouldn't bother describing the implicit
; import of the preceding anonymous namespace. This should be fixed
; in clang.
; CHECK:     DW_TAG_imported_module
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[ANON:.*]]: DW_TAG_namespace
; CHECK-NOT:   DW_AT_name
; CHECK: [[ANON_INNER:.*]]:  DW_TAG_namespace
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "inner"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_INNER_B:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     NULL
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: [[ANON_I:.*]]: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   NULL
; CHECK-NOT: {{DW_TAG|NULL}}

; CHECK: [[MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C15member_functionEv"

; CHECK: [[STATIC_MEM_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "_ZN1C22static_member_functionEv"

; CHECK: [[GLOBAL_FUNC:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "global_function"

; CHECK-LABEL: .debug_gnu_pubnames contents:
; CHECK-NEXT: length = {{.*}} version = 0x0002 unit_offset = 0x00000000 unit_size = {{.*}}
; CHECK-NEXT: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[GLOBAL_FUNC]] EXTERNAL FUNCTION "global_function"
; CHECK-DAG:  [[NS]] EXTERNAL TYPE     "ns"
; CHECK-DAG:  [[MEM_FUNC]] EXTERNAL FUNCTION "C::member_function"
; CHECK-DAG:  [[GLOB_VAR]] EXTERNAL VARIABLE "global_variable"
; CHECK-DAG:  [[GLOB_NS_VAR]] EXTERNAL VARIABLE "ns::global_namespace_variable"
; CHECK-DAG:  [[GLOB_NS_FUNC]] EXTERNAL FUNCTION "ns::global_namespace_function"
; CHECK-DAG:  [[D_VAR]] EXTERNAL VARIABLE "ns::d"
; CHECK-DAG:  [[STATIC_MEM_VAR]] EXTERNAL VARIABLE "C::static_member_variable"
; CHECK-DAG:  [[STATIC_MEM_FUNC]] EXTERNAL FUNCTION "C::static_member_function"
; CHECK-DAG:  [[ANON]] EXTERNAL TYPE "(anonymous namespace)"
; CHECK-DAG:  [[ANON_INNER]] EXTERNAL TYPE "(anonymous namespace)::inner"
; CHECK-DAG:  [[OUTER]] EXTERNAL TYPE "outer"
; CHECK-DAG:  [[OUTER_ANON]] EXTERNAL TYPE "outer::(anonymous namespace)"
; CHECK-DAG:  [[ANON_I]] STATIC VARIABLE "(anonymous namespace)::i"
; CHECK-DAG:  [[ANON_INNER_B]] STATIC VARIABLE "(anonymous namespace)::inner::b"
; CHECK-DAG:  [[OUTER_ANON_C]] STATIC VARIABLE "outer::(anonymous namespace)::c"

; GCC Doesn't put local statics in pubnames, but it seems not unreasonable and
; comes out naturally from LLVM's implementation, so I'm OK with it for now. If
; it's demonstrated that this is a major size concern or degrades debug info
; consumer behavior, feel free to change it.

; CHECK-DAG:  [[F3_Z]] STATIC VARIABLE "f3::z"


; CHECK-LABEL: debug_gnu_pubtypes contents:
; CHECK: Offset     Linkage  Kind     Name
; CHECK-DAG:  [[C]] EXTERNAL TYPE     "C"
; CHECK-DAG:  [[D]] EXTERNAL TYPE     "ns::D"
; CHECK-DAG:  [[INT]] STATIC   TYPE     "int"

%struct.C = type { i8 }
%"struct.ns::D" = type { i32 }

@_ZN1C22static_member_variableE = global i32 0, align 4
@global_variable = global %struct.C zeroinitializer, align 1
@_ZN2ns25global_namespace_variableE = global i32 1, align 4
@_ZN2ns1dE = global %"struct.ns::D" zeroinitializer, align 4
@_ZZ2f3vE1z = internal global i32 0, align 4
@_ZN12_GLOBAL__N_11iE = internal global i32 0, align 4
@_ZN12_GLOBAL__N_15inner1bE = internal global i32 0, align 4
@_ZN5outer12_GLOBAL__N_11cE = internal global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1C15member_functionEv(%struct.C* %this) #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !50, metadata !{!"0x102"}), !dbg !52
  %this1 = load %struct.C** %this.addr
  store i32 0, i32* @_ZN1C22static_member_variableE, align 4, !dbg !53
  ret void, !dbg !54
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_ZN1C22static_member_functionEv() #0 align 2 {
entry:
  %0 = load i32* @_ZN1C22static_member_variableE, align 4, !dbg !55
  ret i32 %0, !dbg !55
}

; Function Attrs: nounwind uwtable
define i32 @_Z15global_functionv() #0 {
entry:
  ret i32 -1, !dbg !56
}

; Function Attrs: nounwind uwtable
define void @_ZN2ns25global_namespace_functionEv() #0 {
entry:
  call void @_ZN1C15member_functionEv(%struct.C* @global_variable), !dbg !57
  ret void, !dbg !58
}

; Function Attrs: nounwind uwtable
define i32* @_Z2f3v() #0 {
entry:
  ret i32* @_ZZ2f3vE1z, !dbg !59
}

; Function Attrs: nounwind uwtable
define i32 @_Z2f7v() #0 {
entry:
  %0 = load i32* @_ZN12_GLOBAL__N_11iE, align 4, !dbg !60
  %call = call i32* @_Z2f3v(), !dbg !60
  %1 = load i32* %call, align 4, !dbg !60
  %add = add nsw i32 %0, %1, !dbg !60
  %2 = load i32* @_ZN12_GLOBAL__N_15inner1bE, align 4, !dbg !60
  %add1 = add nsw i32 %add, %2, !dbg !60
  %3 = load i32* @_ZN5outer12_GLOBAL__N_11cE, align 4, !dbg !60
  %add2 = add nsw i32 %add1, %3, !dbg !60
  ret i32 %add2, !dbg !60
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !48}
!llvm.ident = !{!49}

!0 = !{!"0x11\004\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !3, !19, !32, !45} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/pubnames.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"pubnames.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4, !15}
!4 = !{!"0x13\00C\001\008\008\000\000\000", !1, null, null, !5, null, null, !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6, !8, !12}
!6 = !{!"0xd\00static_member_variable\004\000\000\000\004096", !1, !"_ZTS1C", !7, null} ; [ DW_TAG_member ] [static_member_variable] [line 4, size 0, align 0, offset 0] [static] [from int]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!"0x2e\00member_function\00member_function\00_ZN1C15member_functionEv\002\000\000\000\006\00256\000\002", !1, !"_ZTS1C", !9, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 2] [member_function]
!9 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11}
!11 = !{!"0xf\00\000\0064\0064\000\001088", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!12 = !{!"0x2e\00static_member_function\00static_member_function\00_ZN1C22static_member_functionEv\003\000\000\000\006\00256\000\003", !1, !"_ZTS1C", !13, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 3] [static_member_function]
!13 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{!7}
!15 = !{!"0x13\00D\0028\0032\0032\000\000\000", !1, !16, null, !17, null, null, !"_ZTSN2ns1DE"} ; [ DW_TAG_structure_type ] [D] [line 28, size 32, align 32, offset 0] [def] [from ]
!16 = !{!"0x39\00ns\0023", !1, null} ; [ DW_TAG_namespace ] [ns] [line 23]
!17 = !{!18}
!18 = !{!"0xd\00A\0029\0032\0032\000\000", !1, !"_ZTSN2ns1DE", !7} ; [ DW_TAG_member ] [A] [line 29, size 32, align 32, offset 0] [from int]
!19 = !{!20, !21, !22, !24, !27, !31}
!20 = !{!"0x2e\00member_function\00member_function\00_ZN1C15member_functionEv\009\000\001\000\006\00256\000\009", !1, !"_ZTS1C", !9, null, void (%struct.C*)* @_ZN1C15member_functionEv, null, !8, !2} ; [ DW_TAG_subprogram ] [line 9] [def] [member_function]
!21 = !{!"0x2e\00static_member_function\00static_member_function\00_ZN1C22static_member_functionEv\0013\000\001\000\006\00256\000\0013", !1, !"_ZTS1C", !13, null, i32 ()* @_ZN1C22static_member_functionEv, null, !12, !2} ; [ DW_TAG_subprogram ] [line 13] [def] [static_member_function]
!22 = !{!"0x2e\00global_function\00global_function\00_Z15global_functionv\0019\000\001\000\006\00256\000\0019", !1, !23, !13, null, i32 ()* @_Z15global_functionv, null, null, !2} ; [ DW_TAG_subprogram ] [line 19] [def] [global_function]
!23 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/pubnames.cpp]
!24 = !{!"0x2e\00global_namespace_function\00global_namespace_function\00_ZN2ns25global_namespace_functionEv\0024\000\001\000\006\00256\000\0024", !1, !16, !25, null, void ()* @_ZN2ns25global_namespace_functionEv, null, null, !2} ; [ DW_TAG_subprogram ] [line 24] [def] [global_namespace_function]
!25 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !26, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = !{null}
!27 = !{!"0x2e\00f3\00f3\00_Z2f3v\0037\000\001\000\006\00256\000\0037", !1, !23, !28, null, i32* ()* @_Z2f3v, null, null, !2} ; [ DW_TAG_subprogram ] [line 37] [def] [f3]
!28 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = !{!30}
!30 = !{!"0xf\00\000\0064\0064\000\000", null, null, !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!31 = !{!"0x2e\00f7\00f7\00_Z2f7v\0054\000\001\000\006\00256\000\0054", !1, !23, !13, null, i32 ()* @_Z2f7v, null, null, !2} ; [ DW_TAG_subprogram ] [line 54] [def] [f7]
!32 = !{!33, !34, !35, !36, !37, !38, !41, !44}
!33 = !{!"0x34\00static_member_variable\00static_member_variable\00_ZN1C22static_member_variableE\007\000\001", null, !23, !7, i32* @_ZN1C22static_member_variableE, !6} ; [ DW_TAG_variable ] [static_member_variable] [line 7] [def]
!34 = !{!"0x34\00global_variable\00global_variable\00\0017\000\001", null, !23, !"_ZTS1C", %struct.C* @global_variable, null} ; [ DW_TAG_variable ] [global_variable] [line 17] [def]
!35 = !{!"0x34\00global_namespace_variable\00global_namespace_variable\00_ZN2ns25global_namespace_variableE\0027\000\001", !16, !23, !7, i32* @_ZN2ns25global_namespace_variableE, null} ; [ DW_TAG_variable ] [global_namespace_variable] [line 27] [def]
!36 = !{!"0x34\00d\00d\00_ZN2ns1dE\0030\000\001", !16, !23, !"_ZTSN2ns1DE", %"struct.ns::D"* @_ZN2ns1dE, null} ; [ DW_TAG_variable ] [d] [line 30] [def]
!37 = !{!"0x34\00z\00z\00\0038\001\001", !27, !23, !7, i32* @_ZZ2f3vE1z, null} ; [ DW_TAG_variable ] [z] [line 38] [local] [def]
!38 = !{!"0x34\00c\00c\00_ZN5outer12_GLOBAL__N_11cE\0050\001\001", !39, !23, !7, i32* @_ZN5outer12_GLOBAL__N_11cE, null} ; [ DW_TAG_variable ] [c] [line 50] [local] [def]
!39 = !{!"0x39\00\0049", !1, !40} ; [ DW_TAG_namespace ] [line 49]
!40 = !{!"0x39\00outer\0048", !1, null} ; [ DW_TAG_namespace ] [outer] [line 48]
!41 = !{!"0x34\00b\00b\00_ZN12_GLOBAL__N_15inner1bE\0044\001\001", !42, !23, !7, i32* @_ZN12_GLOBAL__N_15inner1bE, null} ; [ DW_TAG_variable ] [b] [line 44] [local] [def]
!42 = !{!"0x39\00inner\0043", !1, !43} ; [ DW_TAG_namespace ] [inner] [line 43]
!43 = !{!"0x39\00\0033", !1, null} ; [ DW_TAG_namespace ] [line 33]
!44 = !{!"0x34\00i\00i\00_ZN12_GLOBAL__N_11iE\0034\001\001", !43, !23, !7, i32* @_ZN12_GLOBAL__N_11iE, null} ; [ DW_TAG_variable ] [i] [line 34] [local] [def]
!45 = !{!46}
!46 = !{!"0x3a\0040\00", !40, !39} ; [ DW_TAG_imported_module ]
!47 = !{i32 2, !"Dwarf Version", i32 4}
!48 = !{i32 2, !"Debug Info Version", i32 2}
!49 = !{!"clang version 3.5.0 "}
!50 = !{!"0x101\00this\0016777216\001088", !20, null, !51} ; [ DW_TAG_arg_variable ] [this] [line 0]
!51 = !{!"0xf\00\000\0064\0064\000\000", null, null, !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!52 = !MDLocation(line: 0, scope: !20)
!53 = !MDLocation(line: 10, scope: !20)
!54 = !MDLocation(line: 11, scope: !20)
!55 = !MDLocation(line: 14, scope: !21)
!56 = !MDLocation(line: 20, scope: !22)
!57 = !MDLocation(line: 25, scope: !24)
!58 = !MDLocation(line: 26, scope: !24)
!59 = !MDLocation(line: 39, scope: !27)
!60 = !MDLocation(line: 55, scope: !31)
