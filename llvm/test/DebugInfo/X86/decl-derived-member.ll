; REQUIRES: object-emission

; RUN: llc -mtriple x86_64-pc-linux -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Testcase from:
; struct base {
;  virtual ~base();
; };
; typedef base base_type;
; struct foo {
;  base_type b;
; };
; foo f;

; Where member b should be seen as a field at an offset and not a bitfield.

; CHECK: DW_TAG_member
; CHECK: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_AT_bit_offset

source_filename = "test/DebugInfo/X86/decl-derived-member.ll"

%struct.foo = type { %struct.base }
%struct.base = type { i32 (...)** }

$_ZN3fooC2Ev = comdat any

$_ZN3fooD2Ev = comdat any

$_ZN4baseC2Ev = comdat any

@f = global %struct.foo zeroinitializer, align 8, !dbg !0
@__dso_handle = external global i8
@_ZTV4base = external unnamed_addr constant [4 x i8*]
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_decl_derived_member.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !15 {
entry:
  call void @_ZN3fooC2Ev(%struct.foo* @f) #2, !dbg !18
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.foo*)* @_ZN3fooD2Ev to void (i8*)*), i8* bitcast (%struct.foo* @f to i8*), i8* @__dso_handle) #2, !dbg !18
  ret void, !dbg !18
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3fooC2Ev(%struct.foo* %this) unnamed_addr #0 comdat align 2 !dbg !19 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !24, metadata !26), !dbg !27
  %this1 = load %struct.foo*, %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo, %struct.foo* %this1, i32 0, i32 0, !dbg !28
  call void @_ZN4baseC2Ev(%struct.base* %b) #2, !dbg !28
  ret void, !dbg !28
}

; Function Attrs: inlinehint uwtable
define linkonce_odr void @_ZN3fooD2Ev(%struct.foo* %this) unnamed_addr #1 comdat align 2 !dbg !29 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !31, metadata !26), !dbg !32
  %this1 = load %struct.foo*, %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo, %struct.foo* %this1, i32 0, i32 0, !dbg !33
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !33
  ret void, !dbg !35
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 comdat align 2 !dbg !36 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.base** %this.addr, metadata !41, metadata !26), !dbg !43
  %this1 = load %struct.base*, %struct.base** %this.addr
  %0 = bitcast %struct.base* %this1 to i32 (...)***, !dbg !44
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV4base, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, !dbg !44
  ret void, !dbg !44
}

declare void @_ZN4baseD1Ev(%struct.base*) #4

define internal void @_GLOBAL__sub_I_decl_derived_member.cpp() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !45
  ret void
}

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 8, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "decl-derived-member.cpp", directory: "/tmp/dbginfo")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 5, size: 64, align: 64, elements: !4, identifier: "_ZTS3foo")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !2, line: 6, baseType: !6, size: 64, align: 64)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "base_type", file: !2, line: 4, baseType: !7)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "base", file: !2, line: 1, flags: DIFlagFwdDecl, identifier: "_ZTS4base")
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !10, globals: !11, imports: !9)
!9 = !{}
!10 = !{!3, !7}
!11 = !{!0}
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)"}
!15 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !2, file: !2, line: 8, type: !16, isLocal: true, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !8, variables: !9)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DILocation(line: 8, column: 5, scope: !15)
!19 = distinct !DISubprogram(name: "foo", linkageName: "_ZN3fooC2Ev", scope: !3, file: !2, line: 5, type: !20, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !8, declaration: !23, variables: !9)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!23 = !DISubprogram(name: "foo", scope: !3, type: !20, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!24 = !DILocalVariable(name: "this", arg: 1, scope: !19, type: !25, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!26 = !DIExpression()
!27 = !DILocation(line: 0, scope: !19)
!28 = !DILocation(line: 5, column: 8, scope: !19)
!29 = distinct !DISubprogram(name: "~foo", linkageName: "_ZN3fooD2Ev", scope: !3, file: !2, line: 5, type: !20, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !8, declaration: !30, variables: !9)
!30 = !DISubprogram(name: "~foo", scope: !3, type: !20, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!31 = !DILocalVariable(name: "this", arg: 1, scope: !29, type: !25, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = !DILocation(line: 0, scope: !29)
!33 = !DILocation(line: 5, column: 8, scope: !34)
!34 = distinct !DILexicalBlock(scope: !29, file: !2, line: 5, column: 8)
!35 = !DILocation(line: 5, column: 8, scope: !29)
!36 = distinct !DISubprogram(name: "base", linkageName: "_ZN4baseC2Ev", scope: !7, file: !2, line: 1, type: !37, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !8, declaration: !40, variables: !9)
!37 = !DISubroutineType(types: !38)
!38 = !{null, !39}
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!40 = !DISubprogram(name: "base", scope: !7, type: !37, isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!41 = !DILocalVariable(name: "this", arg: 1, scope: !36, type: !42, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!43 = !DILocation(line: 0, scope: !36)
!44 = !DILocation(line: 1, column: 8, scope: !36)
!45 = !DILocation(line: 0, scope: !46)
!46 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_decl_derived_member.cpp", scope: !2, file: !2, type: !47, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !8, variables: !9)
!47 = !DISubroutineType(types: !9)

