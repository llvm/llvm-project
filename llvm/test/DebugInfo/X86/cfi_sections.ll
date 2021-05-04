; RUN: split-file %s %t
; RUN: llc < %t/1.ll -mtriple=x86_64 | FileCheck %s --check-prefix=DEBUG_FRAME
; RUN: llc < %t/1.ll -mtriple=x86_64 -force-dwarf-frame-section | FileCheck %s --check-prefix=DEBUG_FRAME
; RUN: llc < %t/2.ll -mtriple=x86_64 | FileCheck %s --check-prefix=EH_FRAME
; RUN: llc < %t/2.ll -mtriple=x86_64 -force-dwarf-frame-section | FileCheck %s --check-prefix=BOTH
; RUN: llc < %t/3.ll -mtriple=x86_64 | FileCheck %s --check-prefix=EH_FRAME
; RUN: llc < %t/3.ll -mtriple=x86_64 -force-dwarf-frame-section | FileCheck %s --check-prefix=BOTH

; EH_FRAME-NOT: .cfi_sections
; DEBUG_FRAME: .cfi_sections .debug_frame{{$}}
; BOTH: .cfi_sections .eh_frame, .debug_frame{{$}}

;--- 1.ll
;; No function has uwtable/personality or throws && f needs debug info -> emit .debug_frame

define void @f() nounwind !dbg !6 {
entry:
  ret void
}

define void @g() nounwind {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 1, column: 15, scope: !6)

;--- 2.ll
;; g has uwtable and thus needs .eh_frame

define void @f() nounwind !dbg !6 {
entry:
  ret void
}

define void @g() nounwind uwtable !dbg !10 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 1, column: 15, scope: !6)
!10 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 2, column: 15, scope: !10)

;--- 3.ll
;; g has no nounwind, so it is treated as throwable. Emit .eh_frame.

define void @f() nounwind !dbg !6 {
entry:
  ret void
}

define void @g() !dbg !10 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 1, column: 15, scope: !6)
!10 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 2, column: 15, scope: !10)
