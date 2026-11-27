; RUN: opt %s -passes=inline -S -o - | FileCheck %s

;; $ cat test.cpp
;; int g;
;; [[clang::always_inline, gnu::nodebug]]  void a() { g = 1; }
;; void b() { a(); }
;;
;; Check the inlined instructions don't inherit the call's atom info.
;; FIXME: Perhaps we want to do actually do that, to preserve existing
;; behaviour? Unclear what's best.

; CHECK: _Z1bv()
; CHECK: store i32 1, ptr @g, align 4, !dbg [[DBG:!.*]]

; CHECK: distinct !DISubprogram(name: "b", {{.*}}keyInstructions: true)
; CHECK: [[DBG]] = !DILocation(line: 3, scope: ![[#]])

@g = hidden global i32 0, align 4

define hidden void @_Z1av() {
entry:
  store i32 1, ptr @g, align 4
  ret void
}

define hidden void @_Z1bv() !dbg !15 {
entry:
  call void @_Z1av(), !dbg !18
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 19.0.0"}
!15 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!16 = !DISubroutineType(types: !17)
!17 = !{}
!18 = !DILocation(line: 3, scope: !15, atomGroup: 1, atomRank: 1)
!19 = !DILocation(line: 3, scope: !15, atomGroup: 2, atomRank: 1)
