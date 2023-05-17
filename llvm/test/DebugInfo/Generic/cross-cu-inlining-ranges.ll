; RUN: %llc_dwarf %s -O0 -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o --debug-info --verify

; We want to check that when the CU attaches ranges use the correct ContextCU.
; In the following example, after accessing `@foo`, `bar` an `inlined_baz` are created in `foo.swift` CU.
; Setting ranges in `@bar` will then use `bar.swift` CU.
; An incorrect address is eventually calculated based on Base.

; The origin code is:
; foo.swift
; import AppKit.NSLayoutConstraint
; public class Foo {
;     public var c: Int {
;         get {
;             Int(NSLayoutConstraint().constant)
;         }
;         set {
;         }
;     }
; }
; main.swift
; // no mapping for range
; let f: Foo! = nil

; After LTO, `main.swift` will create a global variable, then `Foo` (and relative DIE) created in `main.swift` CU.

define void @foo() !dbg !6 {
  ret void, !dbg !9
}

define void @bar(ptr %0) !dbg !15 {
  store i32 1, ptr %0, align 4, !dbg !16
  store i32 1, ptr %0, align 4, !dbg !21
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.swift", directory: "")
!2 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug)
!3 = !DIFile(filename: "bar.swift", directory: "")
!4 = !{i32 7, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", scope: !0, type: !7, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !DILocation(line: 0, scope: !10, inlinedAt: !13)
!10 = distinct !DISubprogram(name: "init", scope: !12, file: !11, type: !7, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DIFile(filename: "<compiler-generated>", directory: "")
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Type", file: !3, runtimeLang: DW_LANG_Swift, identifier: "Type")
!13 = !DILocation(line: 0, scope: !14)
!14 = distinct !DILexicalBlock(scope: !6, file: !1)
!15 = distinct !DISubprogram(name: "bar", scope: !12, type: !7, spFlags: DISPFlagDefinition, unit: !2)
!16 = !DILocation(line: 0, scope: !17, inlinedAt: !19)
!17 = distinct !DILexicalBlock(scope: !18, file: !3)
!18 = distinct !DISubprogram(name: "inlined_baz", scope: !12, file: !3, type: !7, spFlags: DISPFlagDefinition, unit: !2)
!19 = !DILocation(line: 0, scope: !20)
!20 = distinct !DILexicalBlock(scope: !15, file: !3)
!21 = !DILocation(line: 0, scope: !15)
