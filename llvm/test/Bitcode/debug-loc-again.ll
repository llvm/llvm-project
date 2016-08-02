; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s -check-prefix=BC
; PR23436: Actually emit DEBUG_LOC_AGAIN records.

; BC: <DEBUG_LOC op
; BC: <DEBUG_LOC_AGAIN/>
; BC: <DEBUG_LOC op
; BC: <DEBUG_LOC_AGAIN/>

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
; Check that this round-trips correctly.

define void @foo() {
entry:
  %a = add i32 0, 0, !dbg !3
  %b = add i32 0, 1, !dbg !3
  %c = add i32 0, 2, !dbg !4
  ret void, !dbg !4
}

; CHECK-LABEL: entry:
; CHECK-NEXT: %a = add i32 0, 0, !dbg ![[LINE1:[0-9]+]]
; CHECK-NEXT: %b = add i32 0, 1, !dbg ![[LINE1]]
; CHECK-NEXT: %c = add i32 0, 2, !dbg ![[LINE2:[0-9]+]]
; CHECK-NEXT: ret void, !dbg ![[LINE2]]
; CHECK: ![[LINE1]] = !DILocation(line: 1,
; CHECK: ![[LINE2]] = !DILocation(line: 2,

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !DIFile(filename: "f", directory: "/d"),
                             subprograms: !{!2})
!2 = distinct !DISubprogram(name: "foo")
!3 = !DILocation(line: 1, scope: !2)
!4 = !DILocation(line: 2, scope: !2)
