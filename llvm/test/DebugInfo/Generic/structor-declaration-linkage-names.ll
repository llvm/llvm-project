; REQUIRES: aarch64-registered-target
; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-info - | FileCheck %s

; Make sure we attach DW_AT_linkage_name on function declarations but only
; attach it on definitions if the value is different than on the declaration.

target triple = "arm64-apple-macosx"

define void @_Z11SameLinkagev() !dbg !4 {
entry:
  ret void
}

; CHECK:     DW_AT_linkage_name ("_Z11SameLinkagev")
; CHECK:     DW_AT_declaration (true)
; CHECK-NOT: DW_AT_linkage_name ("_Z11SameLinkagev")

define void @_Z11DiffLinkagev() !dbg !8 {
entry:
  ret void
}

; CHECK: DW_AT_linkage_name ("SomeName")
; CHECK: DW_AT_declaration (true)
; CHECK: DW_AT_linkage_name ("_Z11DiffLinkagev")

define void @_Z15EmptyDefLinkagev() !dbg !10 {
entry:
  ret void
}

; CHECK:     DW_AT_linkage_name ("_Z15EmptyDefLinkagev")
; CHECK:     DW_AT_declaration (true)
; CHECK-NOT: DW_AT_linkage_name

define void @_Z16EmptyDeclLinkagev() !dbg !12 {
entry:
  ret void
}

; CHECK: DW_AT_declaration (true)
; CHECK: DW_AT_linkage_name ("_Z16EmptyDeclLinkagev")

define void @_Z13EmptyLinkagesv() !dbg !14 {
entry:
  ret void
}

; CHECK-NOT: DW_AT_linkage_name

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "SameLinkage", linkageName: "_Z11SameLinkagev", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DISubprogram(name: "SameLinkage", linkageName: "_Z11SameLinkagev", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!8 = distinct !DISubprogram(name: "DiffLinkage", linkageName: "_Z11DiffLinkagev", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !9)
!9 = !DISubprogram(name: "DiffLinkage", linkageName: "SomeName", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!10 = distinct !DISubprogram(name: "EmptyDefLinkage", linkageName: "", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !11)
!11 = !DISubprogram(name: "EmptyDefLinkage", linkageName: "_Z15EmptyDefLinkagev", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!12 = distinct !DISubprogram(name: "EmptyDeclLinkage", linkageName: "_Z16EmptyDeclLinkagev", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !13)
!13 = !DISubprogram(name: "EmptyDeclLinkage", linkageName: "", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!14 = distinct !DISubprogram(name: "EmptyLinkages", linkageName: "", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !15)
!15 = !DISubprogram(name: "EmptyLinkages", linkageName: "", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
