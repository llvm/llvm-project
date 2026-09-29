; RUN: opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -passes=verify -S --disable-debug-info-type-map < %s | FileCheck --check-prefix=CHECK-NOT-ODR %s

; The Rust source:
; pub struct Foo;
; impl Foo {
;   pub fn bar() {}
; }

; ModuleID = 'foo.7e668e4a-cgu.0'
source_filename = "foo.7e668e4a-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; foo::Foo::bar
; Function Attrs: nonlazybind uwtable
define void @_ZN3foo3Foo3bar17h32d65c44145019c8E() unnamed_addr #0 !dbg !6 {
start:
  ret void, !dbg !14
}

attributes #0 = { nonlazybind uwtable "probe-stack"="__rust_probestack" "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 7, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.69.0 (84c898d65 2023-04-16))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!5 = !DIFile(filename: "foo.rs/@/foo.7e668e4a-cgu.0", directory: "/tmp")
!6 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3Foo3bar17h32d65c44145019c8E", scope: !8, file: !7, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, templateParams: !11, retainedNodes: !11)
!7 = !DIFile(filename: "foo.rs", directory: "/tmp", checksumkind: CSK_MD5, checksum: "427d9b572596c2f310b6185a1781f222")
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", scope: !10, file: !9, align: 8, elements: !11, identifier: "a8fd67db4e906c9c4aceea39cb8b0f61")
!9 = !DIFile(filename: "<unknown>", directory: "")
!10 = !DINamespace(name: "foo", scope: null)
!11 = !{}
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 6, column: 6, scope: !6)

; CHECK: definition subprograms cannot be nested within DICompositeType when enabling ODR
; CHECK: warning: ignoring invalid debug info

; CHECK-NOT-ODR-NOT: definition subprograms cannot be nested within DICompositeType when enabling ODR
; CHECK-NOT-ODR-NOT: warning: ignoring invalid debug info
