; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Generated from the source file pr19307.cc:
; #include <string>
; void parse_range(unsigned long long &offset, unsigned long long &limit,
;                  std::string range) {
;   if (range.compare(0, 6, "items=") != 0 || range[6] == '-')
;     offset = 1;
;   range.erase(0, 6);
;   limit = 2;
; }
; with "clang++ -S -emit-llvm -O0 -g pr19307.cc"

; Location of "range" string is spilled from %rdx to stack and is
; addressed via %rbp.
; CHECK: movq %rdx, {{[-0-9]+}}(%rbp)
; CHECK-NEXT: [[START_LABEL:.Ltmp[0-9]+]]:
; This location should be valid until the end of the function.

; Verify that we have proper range in debug_loc section:
; CHECK: .Ldebug_loc{{[0-9]+}}:
; CHECK: DW_OP_breg1
; CHECK:      .quad [[START_LABEL]]-.Lfunc_begin0
; CHECK-NEXT: .quad .Lfunc_end0-.Lfunc_begin0
; CHECK: DW_OP_breg6
; CHECK: DW_OP_deref

; ModuleID = 'pr19307.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }

@.str = private unnamed_addr constant [7 x i8] c"items=\00", align 1

; Function Attrs: uwtable
define void @_Z11parse_rangeRyS_Ss(i64* %offset, i64* %limit, %"class.std::basic_string"* %range) #0 !dbg !13 {
entry:
  %offset.addr = alloca i64*, align 8
  %limit.addr = alloca i64*, align 8
  store i64* %offset, i64** %offset.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %offset.addr, metadata !45, metadata !DIExpression()), !dbg !46
  store i64* %limit, i64** %limit.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %limit.addr, metadata !47, metadata !DIExpression()), !dbg !46
  call void @llvm.dbg.declare(metadata %"class.std::basic_string"* %range, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !49
  %call = call i32 @_ZNKSs7compareEmmPKc(%"class.std::basic_string"* %range, i64 0, i64 6, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0)), !dbg !50
  %cmp = icmp ne i32 %call, 0, !dbg !50
  br i1 %cmp, label %if.then, label %lor.lhs.false, !dbg !50

lor.lhs.false:                                    ; preds = %entry
  %call1 = call i8* @_ZNSsixEm(%"class.std::basic_string"* %range, i64 6), !dbg !52
  %0 = load i8, i8* %call1, !dbg !52
  %conv = sext i8 %0 to i32, !dbg !52
  %cmp2 = icmp eq i32 %conv, 45, !dbg !52
  br i1 %cmp2, label %if.then, label %if.end, !dbg !52

if.then:                                          ; preds = %lor.lhs.false, %entry
  %1 = load i64*, i64** %offset.addr, align 8, !dbg !54
  store i64 1, i64* %1, align 8, !dbg !54
  br label %if.end, !dbg !54

if.end:                                           ; preds = %if.then, %lor.lhs.false
  %call3 = call %"class.std::basic_string"* @_ZNSs5eraseEmm(%"class.std::basic_string"* %range, i64 0, i64 6), !dbg !55
  %2 = load i64*, i64** %limit.addr, align 8, !dbg !56
  store i64 2, i64* %2, align 8, !dbg !56
  ret void, !dbg !57
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @_ZNKSs7compareEmmPKc(%"class.std::basic_string"*, i64, i64, i8*) #2

declare i8* @_ZNSsixEm(%"class.std::basic_string"*, i64) #2

declare %"class.std::basic_string"* @_ZNSs5eraseEmm(%"class.std::basic_string"*, i64, i64) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!42, !43}
!llvm.ident = !{!44}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (209308)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !21)
!1 = !DIFile(filename: "pr19307.cc", directory: "/llvm_cmake_gcc")
!2 = !{}
!3 = !{!4, !6, !8}
!4 = !DICompositeType(tag: DW_TAG_structure_type, line: 83, flags: DIFlagFwdDecl, file: !5, identifier: "_ZTS11__mbstate_t")
!5 = !DIFile(filename: "/usr/include/wchar.h", directory: "/llvm_cmake_gcc")
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "lconv", line: 54, flags: DIFlagFwdDecl, file: !7, identifier: "_ZTS5lconv")
!7 = !DIFile(filename: "/usr/include/locale.h", directory: "/llvm_cmake_gcc")
!8 = !DICompositeType(tag: DW_TAG_class_type, name: "basic_string<char, std::char_traits<char>, std::allocator<char> >", line: 1134, flags: DIFlagFwdDecl, file: !9, scope: !10, identifier: "_ZTSSs")
!9 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/basic_string.tcc", directory: "/llvm_cmake_gcc")
!10 = !DINamespace(name: "std", line: 153, file: !11, scope: null)
!11 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/x86_64-linux-gnu/bits/c++config.h", directory: "/llvm_cmake_gcc")
!13 = distinct !DISubprogram(name: "parse_range", linkageName: "_Z11parse_rangeRyS_Ss", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !14, type: !15, variables: !2)
!14 = !DIFile(filename: "pr19307.cc", directory: "/llvm_cmake_gcc")
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !17, !19}
!17 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !18)
!18 = !DIBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "string", line: 65, file: !20, scope: !10, baseType: !8)
!20 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/stringfwd.h", directory: "/llvm_cmake_gcc")
!21 = !{!22, !26, !29, !33, !38, !41}
!22 = !DIImportedEntity(tag: DW_TAG_imported_module, line: 57, scope: !23, entity: !25)
!23 = !DINamespace(name: "__gnu_debug", line: 55, file: !24, scope: null)
!24 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/debug/debug.h", directory: "/llvm_cmake_gcc")
!25 = !DINamespace(name: "__debug", line: 49, file: !24, scope: !10)
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, line: 66, scope: !10, entity: !27)
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "mbstate_t", line: 106, file: !5, baseType: !28)
!28 = !DIDerivedType(tag: DW_TAG_typedef, name: "__mbstate_t", line: 95, file: !5, baseType: !4)
!29 = !DIImportedEntity(tag: DW_TAG_imported_declaration, line: 141, scope: !10, entity: !30)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "wint_t", line: 141, file: !31, baseType: !32)
!31 = !DIFile(filename: "/llvm_cmake_gcc/bin/../lib/clang/3.5.0/include/stddef.h", directory: "/llvm_cmake_gcc")
!32 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!33 = !DIImportedEntity(tag: DW_TAG_imported_declaration, line: 42, scope: !34, entity: !36)
!34 = !DINamespace(name: "__gnu_cxx", line: 69, file: !35, scope: null)
!35 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/cpp_type_traits.h", directory: "/llvm_cmake_gcc")
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", line: 155, file: !11, scope: !10, baseType: !37)
!37 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, line: 43, scope: !34, entity: !39)
!39 = !DIDerivedType(tag: DW_TAG_typedef, name: "ptrdiff_t", line: 156, file: !11, scope: !10, baseType: !40)
!40 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, line: 55, scope: !10, entity: !6)
!42 = !{i32 2, !"Dwarf Version", i32 4}
!43 = !{i32 2, !"Debug Info Version", i32 3}
!44 = !{!"clang version 3.5.0 (209308)"}
!45 = !DILocalVariable(name: "offset", line: 3, arg: 1, scope: !13, file: !14, type: !17)
!46 = !DILocation(line: 3, scope: !13)
!47 = !DILocalVariable(name: "limit", line: 3, arg: 2, scope: !13, file: !14, type: !17)
!48 = !DILocalVariable(name: "range", line: 4, arg: 3, scope: !13, file: !14, type: !19)
!49 = !DILocation(line: 4, scope: !13)
!50 = !DILocation(line: 5, scope: !51)
!51 = distinct !DILexicalBlock(line: 5, column: 0, file: !1, scope: !13)
!52 = !DILocation(line: 5, scope: !53)
!53 = distinct !DILexicalBlock(line: 5, column: 0, file: !1, scope: !51)
!54 = !DILocation(line: 6, scope: !51)
!55 = !DILocation(line: 7, scope: !13)
!56 = !DILocation(line: 8, scope: !13)
!57 = !DILocation(line: 9, scope: !13)

