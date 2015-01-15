; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1:".*debug-info-namespace.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(5)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:".*foo.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(1)
; CHECK-NOT: NULL
; CHECK: [[I:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK: [[VAR_FWD:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "var_fwd"
; CHECK-NOT: NULL
; CHECK: [[FOO:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "foo"
; CHECK-NEXT: DW_AT_declaration
; CHECK-NOT: NULL
; CHECK: [[BAR:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "bar"
; CHECK: [[FUNC1:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
; CHECK: [[BAZ:0x[0-9a-f]*]]:{{.*}}DW_TAG_typedef
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "baz"
; CHECK: [[VAR_DECL:0x[0-9a-f]*]]:{{.*}}DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "var_decl"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_declaration
; CHECK: [[FUNC_DECL:0x[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func_decl"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_declaration
; CHECK: [[FUNC_FWD:0x[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func_fwd"
; CHECK-NOT: DW_AT_declaration
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
; CHECK: NULL

; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; This is a bug, it should be in F2 but it inherits the file from its
; enclosing scope
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(15)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_imported_module
; Same bug as above, this should be F2, not F1
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(18)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(26)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(27)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FOO]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(28)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAR]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(29)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(30)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[I]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(31)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAZ]]})
; CHECK-NOT: NULL
; CHECK: [[X:0x[0-9a-f]*]]:{{ *}}DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(32)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NEXT: DW_AT_name{{.*}}"X"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(33)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[X]]})
; CHECK-NEXT: DW_AT_name{{.*}}"Y"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(34)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[VAR_DECL]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(35)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC_DECL]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(36)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[VAR_FWD]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(37)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC_FWD]]})

; CHECK: DW_TAG_lexical_block
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(23)
; CHECK-NEXT: DW_AT_import{{.*}}=>
; CHECK: NULL
; CHECK: NULL
; CHECK: NULL

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is:
; // RUN...
; // RUN...
; // RUN...
;
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; extern int i;
; int f1() { return 0; }
; void f1(int) { }
; struct foo;
; struct bar { };
; typedef bar baz;
; extern int var_decl;
; void func_decl(void);
; extern int var_fwd;
; void func_fwd(void);
; }
; }
; namespace A {
; using namespace B;
; }
;
; using namespace A;
; namespace E = A;
; int B::i = f1();
; int func(bool b) {
;   if (b) {
;     using namespace A::B;
;     return i;
;   }
;   using namespace A;
;   using B::foo;
;   using B::bar;
;   using B::f1;
;   using B::i;
;   using B::baz;
;   namespace X = A;
;   namespace Y = X;
;   using B::var_decl;
;   using B::func_decl;
;   using B::var_fwd;
;   using B::func_fwd;
;   return i + X::B::i + Y::B::i;
; }
;
; namespace A {
; using B::i;
; namespace B {
; int var_fwd = i;
; }
; }
; void B::func_fwd() {}

@_ZN1A1B1iE = global i32 0, align 4
@_ZN1A1B7var_fwdE = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_debug_info_namespace.cpp, i8* null }]

; Function Attrs: nounwind ssp uwtable
define i32 @_ZN1A1B2f1Ev() #0 {
entry:
  ret i32 0, !dbg !60
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B2f1Ei(i32) #0 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !61, metadata !62), !dbg !63
  ret void, !dbg !64
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  %call = call i32 @_ZN1A1B2f1Ev(), !dbg !65
  store i32 %call, i32* @_ZN1A1B1iE, align 4, !dbg !65
  ret void, !dbg !65
}

; Function Attrs: nounwind ssp uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %b.addr, metadata !66, metadata !62), !dbg !67
  %0 = load i8* %b.addr, align 1, !dbg !68
  %tobool = trunc i8 %0 to i1, !dbg !68
  br i1 %tobool, label %if.then, label %if.end, !dbg !68

if.then:                                          ; preds = %entry
  %1 = load i32* @_ZN1A1B1iE, align 4, !dbg !69
  store i32 %1, i32* %retval, !dbg !69
  br label %return, !dbg !69

if.end:                                           ; preds = %entry
  %2 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %3 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add = add nsw i32 %2, %3, !dbg !70
  %4 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add1 = add nsw i32 %add, %4, !dbg !70
  store i32 %add1, i32* %retval, !dbg !70
  br label %return, !dbg !70

return:                                           ; preds = %if.end, %if.then
  %5 = load i32* %retval, !dbg !71
  ret i32 %5, !dbg !71
}

define internal void @__cxx_global_var_init1() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  %0 = load i32* @_ZN1A1B1iE, align 4, !dbg !72
  store i32 %0, i32* @_ZN1A1B7var_fwdE, align 4, !dbg !72
  ret void, !dbg !72
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B8func_fwdEv() #0 {
entry:
  ret void, !dbg !73
}

define internal void @_GLOBAL__sub_I_debug_info_namespace.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @__cxx_global_var_init(), !dbg !74
  call void @__cxx_global_var_init1(), !dbg !74
  ret void, !dbg !74
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!57, !58}
!llvm.ident = !{!59}

!0 = !{!"0x11\004\00clang version 3.6.0 \000\00\000\00\001", !1, !2, !3, !9, !30, !33} ; [ DW_TAG_compile_unit ] [/tmp/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"debug-info-namespace.cpp", !"/tmp"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x13\00foo\005\000\000\000\004\000", !5, !6, null, null, null, null, !"_ZTSN1A1B3fooE"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
!5 = !{!"foo.cpp", !"/tmp"}
!6 = !{!"0x39\00B\001", !5, !7} ; [ DW_TAG_namespace ] [B] [line 1]
!7 = !{!"0x39\00A\005", !1, null} ; [ DW_TAG_namespace ] [A] [line 5]
!8 = !{!"0x13\00bar\006\008\008\000\000\000", !5, !6, null, !2, null, null, !"_ZTSN1A1B3barE"} ; [ DW_TAG_structure_type ] [bar] [line 6, size 8, align 8, offset 0] [def] [from ]
!9 = !{!10, !14, !17, !21, !25, !26, !27}
!10 = !{!"0x2e\00f1\00f1\00_ZN1A1B2f1Ev\003\000\001\000\000\00256\000\003", !5, !6, !11, null, i32 ()* @_ZN1A1B2f1Ev, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f1]
!11 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = !{!13}
!13 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = !{!"0x2e\00f1\00f1\00_ZN1A1B2f1Ei\004\000\001\000\000\00256\000\004", !5, !6, !15, null, void (i32)* @_ZN1A1B2f1Ei, null, null, !2} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
!15 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null, !13}
!17 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\0020\001\001\000\000\00256\000\0020", !5, !18, !19, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 20] [local] [def] [__cxx_global_var_init]
!18 = !{!"0x29", !5}   ; [ DW_TAG_file_type ] [/tmp/foo.cpp]
!19 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = !{null}
!21 = !{!"0x2e\00func\00func\00_Z4funcb\0021\000\001\000\000\00256\000\0021", !5, !18, !22, null, i32 (i1)* @_Z4funcb, null, null, !2} ; [ DW_TAG_subprogram ] [line 21] [def] [func]
!22 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{!13, !24}
!24 = !{!"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!25 = !{!"0x2e\00__cxx_global_var_init1\00__cxx_global_var_init1\00\0044\001\001\000\000\00256\000\0044", !5, !18, !19, null, void ()* @__cxx_global_var_init1, null, null, !2} ; [ DW_TAG_subprogram ] [line 44] [local] [def] [__cxx_global_var_init1]
!26 = !{!"0x2e\00func_fwd\00func_fwd\00_ZN1A1B8func_fwdEv\0047\000\001\000\000\00256\000\0047", !5, !6, !19, null, void ()* @_ZN1A1B8func_fwdEv, null, null, !2} ; [ DW_TAG_subprogram ] [line 47] [def] [func_fwd]
!27 = !{!"0x2e\00\00\00_GLOBAL__sub_I_debug_info_namespace.cpp\000\001\001\000\000\0064\000\000", !1, !28, !29, null, void ()* @_GLOBAL__sub_I_debug_info_namespace.cpp, null, null, !2} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!28 = !{!"0x29", !1}   ; [ DW_TAG_file_type ] [/tmp/debug-info-namespace.cpp]
!29 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = !{!31, !32}
!31 = !{!"0x34\00i\00i\00_ZN1A1B1iE\0020\000\001", !6, !18, !13, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 20] [def]
!32 = !{!"0x34\00var_fwd\00var_fwd\00_ZN1A1B7var_fwdE\0044\000\001", !6, !18, !13, i32* @_ZN1A1B7var_fwdE, null} ; [ DW_TAG_variable ] [var_fwd] [line 44] [def]
!33 = !{!34, !35, !36, !37, !40, !41, !42, !43, !44, !45, !47, !48, !49, !51, !54, !55, !56}
!34 = !{!"0x3a\0015\00", !7, !6} ; [ DW_TAG_imported_module ]
!35 = !{!"0x3a\0018\00", !0, !7} ; [ DW_TAG_imported_module ]
!36 = !{!"0x8\0019\00E", !0, !7} ; [ DW_TAG_imported_declaration ]
!37 = !{!"0x3a\0023\00", !38, !6} ; [ DW_TAG_imported_module ]
!38 = !{!"0xb\0022\0010\001", !5, !39} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
!39 = !{!"0xb\0022\007\000", !5, !21} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
!40 = !{!"0x3a\0026\00", !21, !7} ; [ DW_TAG_imported_module ]
!41 = !{!"0x8\0027\00", !21, !"_ZTSN1A1B3fooE"} ; [ DW_TAG_imported_declaration ]
!42 = !{!"0x8\0028\00", !21, !"_ZTSN1A1B3barE"} ; [ DW_TAG_imported_declaration ]
!43 = !{!"0x8\0029\00", !21, !14} ; [ DW_TAG_imported_declaration ]
!44 = !{!"0x8\0030\00", !21, !31} ; [ DW_TAG_imported_declaration ]
!45 = !{!"0x8\0031\00", !21, !46} ; [ DW_TAG_imported_declaration ]
!46 = !{!"0x16\00baz\007\000\000\000\000", !5, !6, !"_ZTSN1A1B3barE"} ; [ DW_TAG_typedef ] [baz] [line 7, size 0, align 0, offset 0] [from _ZTSN1A1B3barE]
!47 = !{!"0x8\0032\00X", !21, !7} ; [ DW_TAG_imported_declaration ]
!48 = !{!"0x8\0033\00Y", !21, !47} ; [ DW_TAG_imported_declaration ]
!49 = !{!"0x8\0034\00", !21, !50} ; [ DW_TAG_imported_declaration ]
!50 = !{!"0x34\00var_decl\00var_decl\00_ZN1A1B8var_declE\008\000\000", !6, !18, !13, null, null} ; [ DW_TAG_variable ] [var_decl] [line 8]
!51 = !{!"0x8\0035\00", !21, !52} ; [ DW_TAG_imported_declaration ]
!52 = !{!"0x2e\00func_decl\00func_decl\00_ZN1A1B9func_declEv\009\000\000\000\000\00256\000\000", !5, !6, !19, null, null, null, null, !53} ; [ DW_TAG_subprogram ] [line 9] [scope 0] [func_decl]
!53 = !{!"0x24"}
!54 = !{!"0x8\0036\00", !21, !32} ; [ DW_TAG_imported_declaration ]
!55 = !{!"0x8\0037\00", !21, !26} ; [ DW_TAG_imported_declaration ]
!56 = !{!"0x8\0042\00", !7, !31} ; [ DW_TAG_imported_declaration ]
!57 = !{i32 2, !"Dwarf Version", i32 2}
!58 = !{i32 2, !"Debug Info Version", i32 2}
!59 = !{!"clang version 3.6.0 "}
!60 = !MDLocation(line: 3, column: 12, scope: !10)
!61 = !{!"0x101\00\0016777220\000", !14, !18, !13} ; [ DW_TAG_arg_variable ] [line 4]
!62 = !{!"0x102"}               ; [ DW_TAG_expression ]
!63 = !MDLocation(line: 4, column: 12, scope: !14)
!64 = !MDLocation(line: 4, column: 16, scope: !14)
!65 = !MDLocation(line: 20, column: 12, scope: !17)
!66 = !{!"0x101\00b\0016777237\000", !21, !18, !24} ; [ DW_TAG_arg_variable ] [b] [line 21]
!67 = !MDLocation(line: 21, column: 15, scope: !21)
!68 = !MDLocation(line: 22, column: 7, scope: !21)
!69 = !MDLocation(line: 24, column: 5, scope: !38)
!70 = !MDLocation(line: 38, column: 3, scope: !21)
!71 = !MDLocation(line: 39, column: 1, scope: !21)
!72 = !MDLocation(line: 44, column: 15, scope: !25)
!73 = !MDLocation(line: 47, column: 21, scope: !26)
!74 = !MDLocation(line: 0, scope: !75)
!75 = !{!"0xb\000", !5, !27} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
