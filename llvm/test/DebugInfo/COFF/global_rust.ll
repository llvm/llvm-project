; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
;
; This test verifies global variables with no scope, as rust generates them for vtables, are handled correctly.
;
; -- global_rust.rs ----------------------------------------------------
; #![crate_type = "lib"]
; pub trait Foo {
;     fn foo(&self) {}
; }
;
; impl Foo for u32 {}
;
; pub fn foo(_: &dyn Foo) {}
;
; pub fn bar() {
;     foo(&42);
; }
; -----------------------------------------------------------------------------
;
; $ rustc -emit=llvm-ir --target=x86_64-pc-windows-msvc -C debuginfo=2 global_rust.rs
;

; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (6)

; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     SubSectionSize: 0x34
; CHECK:     DataSym {
; CHECK:       Kind: S_LDATA32 (0x110C)
; CHECK:       DataOffset: .rdata+0x8
; CHECK:       Type: impl$<u32, global_rust::Foo>::vtable_type$ (0x101D)
; CHECK:       DisplayName: impl$<u32, global_rust::Foo>::vtable$
; CHECK:       LinkageName: .rdata
; CHECK:     }
; CHECK:   ]

; ModuleID = 'global_rust.5984dd87-cgu.0'
source_filename = "global_rust.5984dd87-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@alloc7 = private unnamed_addr constant <{ [4 x i8] }> <{ [4 x i8] c"*\00\00\00" }>, align 4
@vtable.0 = private unnamed_addr constant <{ i8*, [16 x i8], i8* }> <{ i8* bitcast (void (i32*)* @"_ZN4core3ptr24drop_in_place$LT$u32$GT$17h5aa897c8344c34eaE" to i8*), [16 x i8] c"\04\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00", i8* bitcast (void (i32*)* @_ZN11global_rust3Foo3foo17h17696ada84e467feE to i8*) }>, align 8, !dbg !0

; core::ptr::drop_in_place<u32>
; Function Attrs: inlinehint uwtable
define internal void @"_ZN4core3ptr24drop_in_place$LT$u32$GT$17h5aa897c8344c34eaE"(i32* %_1) unnamed_addr #0 !dbg !22 {
start:
  %_1.dbg.spill = alloca i32*, align 8
  store i32* %_1, i32** %_1.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata i32** %_1.dbg.spill, metadata !30, metadata !DIExpression()), !dbg !33
  ret void, !dbg !33
}

; global_rust::Foo::foo
; Function Attrs: uwtable
define void @_ZN11global_rust3Foo3foo17h17696ada84e467feE(i32* align 4 %self) unnamed_addr #1 !dbg !34 {
start:
  %self.dbg.spill = alloca i32*, align 8
  store i32* %self, i32** %self.dbg.spill, align 8
  call void @llvm.dbg.declare(metadata i32** %self.dbg.spill, metadata !42, metadata !DIExpression()), !dbg !45
  ret void, !dbg !45
}

; global_rust::foo
; Function Attrs: uwtable
define void @_ZN11global_rust3foo17hbb4da99a7bb855e3E({}* align 1 %_1.0, [3 x i64]* align 8 %_1.1) unnamed_addr #1 !dbg !46 {
start:
  %_1.dbg.spill = alloca { {}*, [3 x i64]* }, align 8
  %0 = getelementptr inbounds { {}*, [3 x i64]* }, { {}*, [3 x i64]* }* %_1.dbg.spill, i32 0, i32 0
  store {}* %_1.0, {}** %0, align 8
  %1 = getelementptr inbounds { {}*, [3 x i64]* }, { {}*, [3 x i64]* }* %_1.dbg.spill, i32 0, i32 1
  store [3 x i64]* %_1.1, [3 x i64]** %1, align 8
  call void @llvm.dbg.declare(metadata { {}*, [3 x i64]* }* %_1.dbg.spill, metadata !60, metadata !DIExpression()), !dbg !61
  ret void, !dbg !61
}

; global_rust::bar
; Function Attrs: uwtable
define void @_ZN11global_rust3bar17h9ea8de32fc3f1360E() unnamed_addr #1 !dbg !62 {
start:
; call global_rust::foo
  call void @_ZN11global_rust3foo17hbb4da99a7bb855e3E({}* align 1 bitcast (<{ [4 x i8] }>* @alloc7 to {}*), [3 x i64]* align 8 bitcast (<{ i8*, [16 x i8], i8* }>* @vtable.0 to [3 x i64]*)), !dbg !65
  br label %bb1, !dbg !65

bb1:                                              ; preds = %start
  ret void, !dbg !66
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { inlinehint uwtable "target-cpu"="x86-64" }
attributes #1 = { uwtable "target-cpu"="x86-64" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!16, !17, !18}
!llvm.dbg.cu = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "impl$<u32, global_rust::Foo>::vtable$", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "impl$<u32, global_rust::Foo>::vtable_type$", file: !2, size: 256, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !14, templateParams: !8, identifier: "4a384a40e448d9d82ef8cb395527d231")
!4 = !{!5, !9, !12, !13}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: !3, file: !2, baseType: !6, size: 64, align: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ptr_const$<tuple$<> >", baseType: !7, size: 64, align: 64, dwarfAddressSpace: 0)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "tuple$<>", file: !2, align: 8, elements: !8, identifier: "65e33f3994015bcf158992dbe0323c0")
!8 = !{}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !3, file: !2, baseType: !10, size: 64, align: 64, offset: 64)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "usize", file: !2, baseType: !11)
!11 = !DIBasicType(name: "size_t", size: 64, encoding: DW_ATE_unsigned)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "align", scope: !3, file: !2, baseType: !10, size: 64, align: 64, offset: 128)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "__method3", scope: !3, file: !2, baseType: !6, size: 64, align: 64, offset: 192)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "u32", file: !2, baseType: !15)
!15 = !DIBasicType(name: "unsigned __int32", size: 32, encoding: DW_ATE_unsigned)
!16 = !{i32 7, !"PIC Level", i32 2}
!17 = !{i32 2, !"CodeView", i32 1}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !20, producer: "clang LLVM (rustc version 1.64.0 (a55dd71d5 2022-09-19))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !21)
!20 = !DIFile(filename: "global_rust.rs/@/global_rust.5984dd87-cgu.0", directory: "/tmp/llvm2")
!21 = !{!0}
!22 = distinct !DISubprogram(name: "drop_in_place<u32>", linkageName: "_ZN4core3ptr24drop_in_place$LT$u32$GT$17h5aa897c8344c34eaE", scope: !24, file: !23, line: 487, type: !26, scopeLine: 487, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !19, templateParams: !31, retainedNodes: !29)
!23 = !DIFile(filename: "/rustc/a55dd71d5fb0ec5a6a3a9e8c27b2127ba491ce52\\library\\core\\src\\ptr\\mod.rs", directory: "", checksumkind: CSK_SHA1, checksum: "09826e1d72e15b98151545882632f685c97dc29f")
!24 = !DINamespace(name: "ptr", scope: !25)
!25 = !DINamespace(name: "core", scope: null)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ptr_mut$<u32>", baseType: !14, size: 64, align: 64, dwarfAddressSpace: 0)
!29 = !{!30}
!30 = !DILocalVariable(arg: 1, scope: !22, file: !23, line: 487, type: !28)
!31 = !{!32}
!32 = !DITemplateTypeParameter(name: "T", type: !14)
!33 = !DILocation(line: 487, scope: !22)
!34 = distinct !DISubprogram(name: "foo<u32>", linkageName: "_ZN11global_rust3Foo3foo17h17696ada84e467feE", scope: !36, file: !35, line: 3, type: !38, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !19, templateParams: !43, retainedNodes: !41)
!35 = !DIFile(filename: "global_rust.rs", directory: "/tmp/llvm2", checksumkind: CSK_SHA1, checksum: "d28353083c5fcd457fffa5827b0372c4a843c302")
!36 = !DINamespace(name: "Foo", scope: !37)
!37 = !DINamespace(name: "global_rust", scope: null)
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ref$<u32>", baseType: !14, size: 64, align: 64, dwarfAddressSpace: 0)
!41 = !{!42}
!42 = !DILocalVariable(name: "self", arg: 1, scope: !34, file: !35, line: 3, type: !40)
!43 = !{!44}
!44 = !DITemplateTypeParameter(name: "Self", type: !14)
!45 = !DILocation(line: 3, scope: !34)
!46 = distinct !DISubprogram(name: "foo", linkageName: "_ZN11global_rust3foo17hbb4da99a7bb855e3E", scope: !37, file: !35, line: 8, type: !47, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !19, templateParams: !8, retainedNodes: !59)
!47 = !DISubroutineType(types: !48)
!48 = !{null, !49}
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "ref$<dyn$<global_rust::Foo> >", file: !2, size: 128, align: 64, elements: !50, templateParams: !8, identifier: "2c39c7f196ba93e4e4fbfefe6e460dfb")
!50 = !{!51, !54}
!51 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !49, file: !2, baseType: !52, size: 64, align: 64)
!52 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !53, size: 64, align: 64, dwarfAddressSpace: 0)
!53 = !DICompositeType(tag: DW_TAG_structure_type, name: "dyn$<global_rust::Foo>", file: !2, align: 8, elements: !8, identifier: "dc5af67081d01f4b3cf3420f9b3ec7fa")
!54 = !DIDerivedType(tag: DW_TAG_member, name: "vtable", scope: !49, file: !2, baseType: !55, size: 64, align: 64, offset: 64)
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ref$<array$<usize,3> >", baseType: !56, size: 64, align: 64, dwarfAddressSpace: 0)
!56 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 192, align: 64, elements: !57)
!57 = !{!58}
!58 = !DISubrange(count: 3, lowerBound: 0)
!59 = !{!60}
!60 = !DILocalVariable(arg: 1, scope: !46, file: !35, line: 8, type: !49)
!61 = !DILocation(line: 8, scope: !46)
!62 = distinct !DISubprogram(name: "bar", linkageName: "_ZN11global_rust3bar17h9ea8de32fc3f1360E", scope: !37, file: !35, line: 10, type: !63, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !19, templateParams: !8, retainedNodes: !8)
!63 = !DISubroutineType(types: !64)
!64 = !{null}
!65 = !DILocation(line: 11, scope: !62)
!66 = !DILocation(line: 12, scope: !62)
