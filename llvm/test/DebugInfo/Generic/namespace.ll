; RUN: %llc_dwarf -O0 -filetype=obj -dwarf-linkage-names=All < %s | llvm-dwarfdump -v - | FileCheck %s
; CHECK: debug_info contents
; CHECK: DW_AT_name{{.*}}= [[F1:.*]])
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NOT: DW_AT_decl_file
; CHECK-NOT: DW_AT_decl_line
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NOT: DW_AT_decl_file
; CHECK-NOT: DW_AT_decl_line
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
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
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
; CHECK: NULL

; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:.*]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(15)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:.*]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(18)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:.*]])
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

@_ZN1A1B1iE = global i32 0, align 4, !dbg !131
@_ZN1A1B7var_fwdE = global i32 0, align 4, !dbg !132
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_debug_info_namespace.cpp, i8* null }]

; Function Attrs: nounwind ssp uwtable
define i32 @_ZN1A1B2f1Ev() #0 !dbg !10 {
entry:
  ret i32 0, !dbg !60
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B2f1Ei(i32) #0 !dbg !14 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !61, metadata !62), !dbg !63
  ret void, !dbg !64
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" !dbg !17 {
entry:
  %call = call i32 @_ZN1A1B2f1Ev(), !dbg !65
  store i32 %call, i32* @_ZN1A1B1iE, align 4, !dbg !65
  ret void, !dbg !65
}

; Function Attrs: nounwind ssp uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 !dbg !21 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %b.addr, metadata !66, metadata !62), !dbg !67
  %0 = load i8, i8* %b.addr, align 1, !dbg !68
  %tobool = trunc i8 %0 to i1, !dbg !68
  br i1 %tobool, label %if.then, label %if.end, !dbg !68

if.then:                                          ; preds = %entry
  %1 = load i32, i32* @_ZN1A1B1iE, align 4, !dbg !69
  store i32 %1, i32* %retval, !dbg !69
  br label %return, !dbg !69

if.end:                                           ; preds = %entry
  %2 = load i32, i32* @_ZN1A1B1iE, align 4, !dbg !70
  %3 = load i32, i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add = add nsw i32 %2, %3, !dbg !70
  %4 = load i32, i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add1 = add nsw i32 %add, %4, !dbg !70
  store i32 %add1, i32* %retval, !dbg !70
  br label %return, !dbg !70

return:                                           ; preds = %if.end, %if.then
  %5 = load i32, i32* %retval, !dbg !71
  ret i32 %5, !dbg !71
}

define internal void @__cxx_global_var_init1() section "__TEXT,__StaticInit,regular,pure_instructions" !dbg !25 {
entry:
  %0 = load i32, i32* @_ZN1A1B1iE, align 4, !dbg !72
  store i32 %0, i32* @_ZN1A1B7var_fwdE, align 4, !dbg !72
  ret void, !dbg !72
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B8func_fwdEv() #0 !dbg !26 {
entry:
  ret void, !dbg !73
}

define internal void @_GLOBAL__sub_I_debug_info_namespace.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @__cxx_global_var_init(), !dbg !74
  call void @__cxx_global_var_init1(), !dbg !74
  ret void, !dbg !74
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!57, !58}
!llvm.ident = !{!59}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !30, imports: !33)
!1 = !DIFile(filename: "debug-info-namespace.cpp", directory: "/tmp")
!2 = !{}
!3 = !{!4, !8}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 5, flags: DIFlagFwdDecl, file: !5, scope: !6, identifier: "_ZTSN1A1B3fooE")
!5 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!6 = !DINamespace(name: "B", scope: !7)
!7 = !DINamespace(name: "A", scope: null)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "bar", line: 6, size: 8, align: 8, file: !5, scope: !6, elements: !2, identifier: "_ZTSN1A1B3barE")
!10 = distinct !DISubprogram(name: "f1", linkageName: "_ZN1A1B2f1Ev", line: 3, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !11, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = distinct !DISubprogram(name: "f1", linkageName: "_ZN1A1B2f1Ei", line: 4, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !5, scope: !6, type: !15, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !13}
!17 = distinct !DISubprogram(name: "__cxx_global_var_init", line: 20, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 20, file: !5, scope: !18, type: !19, retainedNodes: !2)
!18 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcb", line: 21, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 21, file: !5, scope: !18, type: !22, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{!13, !24}
!24 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!25 = distinct !DISubprogram(name: "__cxx_global_var_init1", line: 44, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 44, file: !5, scope: !18, type: !19, retainedNodes: !2)
!26 = distinct !DISubprogram(name: "func_fwd", linkageName: "_ZN1A1B8func_fwdEv", line: 47, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 47, file: !5, scope: !6, type: !19, retainedNodes: !2)
!27 = distinct !DISubprogram(name: "", linkageName: "_GLOBAL__sub_I_debug_info_namespace.cpp", isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, file: !1, scope: !28, type: !29, retainedNodes: !2)
!28 = !DIFile(filename: "debug-info-namespace.cpp", directory: "/tmp")
!29 = !DISubroutineType(types: !2)
!30 = !{!131, !132}
!31 = !DIGlobalVariable(name: "i", linkageName: "_ZN1A1B1iE", line: 20, isLocal: false, isDefinition: true, scope: !6, file: !18, type: !13)
!32 = !DIGlobalVariable(name: "var_fwd", linkageName: "_ZN1A1B7var_fwdE", line: 44, isLocal: false, isDefinition: true, scope: !6, file: !18, type: !13)
!33 = !{!34, !35, !36, !37, !40, !41, !42, !43, !44, !45, !47, !48, !49, !51, !54, !55, !56}
!34 = !DIImportedEntity(tag: DW_TAG_imported_module, file: !5, line: 15, scope: !7, entity: !6)
!35 = !DIImportedEntity(tag: DW_TAG_imported_module, file: !5, line: 18, scope: !0, entity: !7)
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 19, name: "E", scope: !0, entity: !7)
!37 = !DIImportedEntity(tag: DW_TAG_imported_module, file: !5, line: 23, scope: !38, entity: !6)
!38 = distinct !DILexicalBlock(line: 22, column: 10, file: !5, scope: !39)
!39 = distinct !DILexicalBlock(line: 22, column: 7, file: !5, scope: !21)
!40 = !DIImportedEntity(tag: DW_TAG_imported_module, file: !5, line: 26, scope: !21, entity: !7)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 27, scope: !21, entity: !4)
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 28, scope: !21, entity: !8)
!43 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 29, scope: !21, entity: !14)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 30, scope: !21, entity: !31)
!45 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 31, scope: !21, entity: !46)
!46 = !DIDerivedType(tag: DW_TAG_typedef, name: "baz", line: 7, file: !5, scope: !6, baseType: !8)
!47 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 32, name: "X", scope: !21, entity: !7)
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 33, name: "Y", scope: !21, entity: !47)
!49 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 34, scope: !21, entity: !50)
!50 = !DIGlobalVariable(name: "var_decl", linkageName: "_ZN1A1B8var_declE", line: 8, isLocal: false, isDefinition: false, scope: !6, file: !18, type: !13)
!51 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 35, scope: !21, entity: !52)
!52 = !DISubprogram(name: "func_decl", linkageName: "_ZN1A1B9func_declEv", line: 9, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false, file: !5, scope: !6, type: !19, retainedNodes: !53)
!53 = !{} ; previously: invalid DW_TAG_base_type
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 36, scope: !21, entity: !32)
!55 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 37, scope: !21, entity: !26)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, file: !5, line: 42, scope: !7, entity: !31)
!57 = !{i32 2, !"Dwarf Version", i32 2}
!58 = !{i32 2, !"Debug Info Version", i32 3}
!59 = !{!"clang version 3.6.0 "}
!60 = !DILocation(line: 3, column: 12, scope: !10)
!61 = !DILocalVariable(name: "", line: 4, arg: 1, scope: !14, file: !18, type: !13)
!62 = !DIExpression()
!63 = !DILocation(line: 4, column: 12, scope: !14)
!64 = !DILocation(line: 4, column: 16, scope: !14)
!65 = !DILocation(line: 20, column: 12, scope: !17)
!66 = !DILocalVariable(name: "b", line: 21, arg: 1, scope: !21, file: !18, type: !24)
!67 = !DILocation(line: 21, column: 15, scope: !21)
!68 = !DILocation(line: 22, column: 7, scope: !21)
!69 = !DILocation(line: 24, column: 5, scope: !38)
!70 = !DILocation(line: 38, column: 3, scope: !21)
!71 = !DILocation(line: 39, column: 1, scope: !21)
!72 = !DILocation(line: 44, column: 15, scope: !25)
!73 = !DILocation(line: 47, column: 21, scope: !26)
!74 = !DILocation(line: 0, scope: !75)
!75 = !DILexicalBlockFile(discriminator: 0, file: !5, scope: !27)
!131 = !DIGlobalVariableExpression(var: !31, expr: !DIExpression())
!132 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
