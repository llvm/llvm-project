; RUN: opt -S -passes='default<O0>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O0 --check-prefix=UNIQUE
; RUN: opt -S -passes='default<O1>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='default<O2>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='thinlto-pre-link<O1>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes='thinlto-pre-link<O2>' -new-pm-pseudo-probe-for-profiling -new-pm-unique-internal-linkage-names -debug-pass-manager < %s 2>&1 | FileCheck %s --check-prefix=O2 --check-prefix=UNIQUE
; RUN: opt -S -passes=unique-internal-linkage-names < %s -o - | FileCheck %s --check-prefix=DBG

define internal i32 @foo() !dbg !15 {
entry:
  ret i32 0
}

define dso_local i32 (...)* @bar() {
entry:
  ret i32 (...)* bitcast (i32 ()* @foo to i32 (...)*)
}

define internal i32 @go() !dbg !19 {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 5, type: !16, scopeLine: 5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !18, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!13}
!18 = !DISubprogram(name: "foo", linkageName: "foo", scope: !1, isDefinition: false, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!19 = distinct !DISubprogram(name: "go", scope: !1, file: !1, line: 5, type: !16, scopeLine: 5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !18, retainedNodes: !2)

; O0: Running pass: UniqueInternalLinkageNamesPass

;; Check UniqueInternalLinkageNamesPass is scheduled before SampleProfileProbePass.
; O2: Running pass: UniqueInternalLinkageNamesPass
; O2: Running pass: SampleProfileProbePass

; UNIQUE: define internal i32 @foo.__uniq.{{[0-9]+}}() [[ATTR:#[0-9]+]]
; UNIQUE: ret {{.*}} @foo.__uniq.{{[0-9]+}} {{.*}}
; UNIQUE: attributes [[ATTR]] = {{{.*}} "sample-profile-suffix-elision-policy"="selected" {{.*}}}

; DBG: distinct !DISubprogram(name: "foo", linkageName: "foo.__uniq.{{[0-9]+}}", scope: ![[#]]
; DBG: !DISubprogram(name: "foo", linkageName: "foo.__uniq.{{[0-9]+}}", scope: ![[#]]
; DBG: distinct !DISubprogram(name: "go", scope: ![[#]]
