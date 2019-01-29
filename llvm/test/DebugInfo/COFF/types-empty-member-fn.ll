; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; ModuleID = 'foo.3a1fbbbh-cgu.0'
source_filename = "foo.3a1fbbbh-cgu.0"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Rust source to regenerate:
; $ cat foo.rs
; pub struct Foo;
; impl Foo {
;     pub fn foo() {}
; }
; $ rustc foo.rs --crate-type lib -Cdebuginfo=1 --emit=llvm-ir

; CHECK:      CodeViewTypes [
; CHECK:        MemberFunction (0x1006) {
; CHECK-NEXT:     TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:     ReturnType: void (0x3)
; CHECK-NEXT:     ClassType: foo::Foo (0x1000)
; CHECK-NEXT:     ThisType: 0x0
; CHECK-NEXT:     CallingConvention: NearC (0x0)
; CHECK-NEXT:     FunctionOptions [ (0x0)
; CHECK-NEXT:     ]
; CHECK-NEXT:     NumParameters: 0
; CHECK-NEXT:     ArgListType: () (0x1005)
; CHECK-NEXT:     ThisAdjustment: 0
; CHECK-NEXT:   }
; CHECK-NEXT:   MemberFuncId (0x1007) {
; CHECK-NEXT:     TypeLeafKind: LF_MFUNC_ID (0x1602)
; CHECK-NEXT:     ClassType: foo::Foo (0x1000)
; CHECK-NEXT:     FunctionType: void foo::Foo::() (0x1006)
; CHECK-NEXT:     Name: foo
; CHECK-NEXT:   }
; CHECK:      CodeViewDebugInfo [
; CHECK:        FunctionLineTable [
; CHECK-NEXT:     LinkageName: _ZN3foo3Foo3foo17hc557c2121772885bE
; CHECK-NEXT:     Flags: 0x0
; CHECK-NEXT:     CodeSize: 0x1
; CHECK-NEXT:     FilenameSegment [
; CHECK-NEXT:       Filename: D:\rust\foo.rs (0x0)
; CHECK-NEXT:       +0x0 [
; CHECK-NEXT:         LineNumberStart: 3
; CHECK-NEXT:         LineNumberEndDelta: 0
; CHECK-NEXT:         IsStatement: No
; CHECK-NEXT:       ]
; CHECK-NEXT:     ]
; CHECK-NEXT:   ]

; foo::Foo::foo
; Function Attrs: uwtable
define void @_ZN3foo3Foo3foo17hc557c2121772885bE() unnamed_addr #0 !dbg !5 {
start:
  ret void, !dbg !10
}

attributes #0 = { uwtable "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "clang LLVM (rustc version 1.33.0-nightly (8b0f0156e 2019-01-22))", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "foo.rs", directory: "D:\5Crust")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", linkageName: "_ZN3foo3Foo3foo17hc557c2121772885bE", scope: !6, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, templateParams: !2, retainedNodes: !2)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", scope: !8, file: !7, align: 8, elements: !2, templateParams: !2, identifier: "5105d9fe1a2a3c68518268151b672274")
!7 = !DIFile(filename: "<unknown>", directory: "")
!8 = !DINamespace(name: "foo", scope: null)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 3, scope: !5)
