; RUN: split-file %s %t
; RUN: llvm-as %t/key-instr-enabled.ll -o %t/key-instr-enabled.bc
; RUN: llvm-as %t/key-instr-disabled.ll -o %t/key-instr-disabled.bc
; RUN: llvm-link %t/key-instr-enabled.bc %t/key-instr-disabled.bc -o - | llvm-dis | FileCheck %s

;; Check the Key Instructions metadata is preserved correctly when linking a
;; modules with Key Instructions enabled/disabled.

;; Key Instructions enabled.
; CHECK: void @f() !dbg [[f:!.*]] {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void, !dbg [[enabled:!.*]]
; CHECK-NEXT: }

;; Key Instructions disabled.
; CHECK: void @g() !dbg [[g:!.*]] {
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void, !dbg [[disabled:!.*]]
; CHECK-NEXT: }

; CHECK: !llvm.dbg.cu = !{!0, !2}
; CHECK-NEXT: !llvm.module.flags = !{!4}

; CHECK: [[file1:!.*]] = !DIFile(filename: "key-instr-enabled.cpp", directory: "/")
; CHECK: [[file2:!.*]] = !DIFile(filename: "key-instr-disabled.cpp", directory: "/")
; CHECK: [[f]] = distinct !DISubprogram(name: "f", scope: [[file1]]{{.*}},  keyInstructions: true)
; CHECK: [[enabled]] = !DILocation(line: 1, column: 11, scope: [[f]], atomGroup: 1, atomRank: 1)
; CHECK: [[g]] = distinct !DISubprogram(name: "g", scope: [[file2]]
; CHECK-NOT:                            keyInstructions
; CHECK-SAME:                           )
; CHECK: [[disabled]] = !DILocation(line: 1, column: 11, scope: [[g]])

;--- key-instr-enabled.ll
define dso_local void @f() !dbg !10 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "key-instr-enabled.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 21.0.0git"}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 1, column: 11, scope: !10, atomGroup: 1, atomRank: 1)

;--- key-instr-disabled.ll
define dso_local void @g() !dbg !10 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "key-instr-disabled.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 21.0.0git"}
!10 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: false)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 1, column: 11, scope: !10)
