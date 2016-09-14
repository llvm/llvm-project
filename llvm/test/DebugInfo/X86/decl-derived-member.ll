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

%struct.foo = type { %struct.base }
%struct.base = type { i32 (...)** }

$_ZN3fooC2Ev = comdat any

$_ZN3fooD2Ev = comdat any

$_ZN4baseC2Ev = comdat any

@f = global %struct.foo zeroinitializer, align 8, !dbg !29
@__dso_handle = external global i8
@_ZTV4base = external unnamed_addr constant [4 x i8*]
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_decl_derived_member.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" !dbg !10 {
entry:
  call void @_ZN3fooC2Ev(%struct.foo* @f) #2, !dbg !33
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.foo*)* @_ZN3fooD2Ev to void (i8*)*), i8* bitcast (%struct.foo* @f to i8*), i8* @__dso_handle) #2, !dbg !33
  ret void, !dbg !33
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3fooC2Ev(%struct.foo* %this) unnamed_addr #0 comdat align 2 !dbg !14 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !34, metadata !36), !dbg !37
  %this1 = load %struct.foo*, %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo, %struct.foo* %this1, i32 0, i32 0, !dbg !38
  call void @_ZN4baseC2Ev(%struct.base* %b) #2, !dbg !38
  ret void, !dbg !38
}

; Function Attrs: inlinehint uwtable
define linkonce_odr void @_ZN3fooD2Ev(%struct.foo* %this) unnamed_addr #1 comdat align 2 !dbg !24 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.foo** %this.addr, metadata !39, metadata !36), !dbg !40
  %this1 = load %struct.foo*, %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo, %struct.foo* %this1, i32 0, i32 0, !dbg !41
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !41
  ret void, !dbg !43
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 comdat align 2 !dbg !19 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.base** %this.addr, metadata !44, metadata !36), !dbg !46
  %this1 = load %struct.base*, %struct.base** %this.addr
  %0 = bitcast %struct.base* %this1 to i32 (...)***, !dbg !47
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTV4base, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, !dbg !47
  ret void, !dbg !47
}

declare void @_ZN4baseD1Ev(%struct.base*) #4

define internal void @_GLOBAL__sub_I_decl_derived_member.cpp() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !48
  ret void
}

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30, !31}
!llvm.ident = !{!32}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !28, imports: !2)
!1 = !DIFile(filename: "decl-derived-member.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !8}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 5, size: 64, align: 64, file: !1, elements: !5, identifier: "_ZTS3foo")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 6, size: 64, align: 64, file: !1, scope: !4, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "base_type", line: 4, file: !1, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "base", line: 1, flags: DIFlagFwdDecl, file: !1, identifier: "_ZTS4base")
!10 = distinct !DISubprogram(name: "__cxx_global_var_init", line: 8, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 8, file: !1, scope: !11, type: !12, variables: !2)
!11 = !DIFile(filename: "decl-derived-member.cpp", directory: "/tmp/dbginfo")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = distinct !DISubprogram(name: "foo", linkageName: "_ZN3fooC2Ev", line: 5, isLocal: false, isDefinition: true, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !4, type: !15, declaration: !18, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!18 = !DISubprogram(name: "foo", isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scope: !4, type: !15)
!19 = distinct !DISubprogram(name: "base", linkageName: "_ZN4baseC2Ev", line: 1, isLocal: false, isDefinition: true, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !8, type: !20, declaration: !23, variables: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22}
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !8)
!23 = !DISubprogram(name: "base", isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scope: !8, type: !20)
!24 = distinct !DISubprogram(name: "~foo", linkageName: "_ZN3fooD2Ev", line: 5, isLocal: false, isDefinition: true, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !4, type: !15, declaration: !25, variables: !2)
!25 = !DISubprogram(name: "~foo", isLocal: false, isDefinition: false, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scope: !4, type: !15)
!26 = distinct !DISubprogram(name: "", linkageName: "_GLOBAL__sub_I_decl_derived_member.cpp", isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !0, file: !1, scope: !11, type: !27, variables: !2)
!27 = !DISubroutineType(types: !2)
!28 = !{!29}
!29 = !DIGlobalVariable(name: "f", line: 8, isLocal: false, isDefinition: true, scope: null, file: !11, type: !4)
!30 = !{i32 2, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{!"clang version 3.7.0 (trunk 227104) (llvm/trunk 227103)"}
!33 = !DILocation(line: 8, column: 5, scope: !10)
!34 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !14, type: !35)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!36 = !DIExpression()
!37 = !DILocation(line: 0, scope: !14)
!38 = !DILocation(line: 5, column: 8, scope: !14)
!39 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !24, type: !35)
!40 = !DILocation(line: 0, scope: !24)
!41 = !DILocation(line: 5, column: 8, scope: !42)
!42 = distinct !DILexicalBlock(line: 5, column: 8, file: !1, scope: !24)
!43 = !DILocation(line: 5, column: 8, scope: !24)
!44 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !19, type: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!46 = !DILocation(line: 0, scope: !19)
!47 = !DILocation(line: 1, column: 8, scope: !19)
!48 = !DILocation(line: 0, scope: !26)
