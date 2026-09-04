; RUN: llc -mtriple=thumbv7-windows-msvc -O1 < %s | FileCheck %s

; We used to crash on ARM with optimizations on.

; C source:
; void foo(void) {
;   __annotation(L"annotation");
; }

define void @foo() !dbg !8 {
entry:
; CHECK-LABEL: foo:
; CHECK: {{.*}}annotation0:
; CHECK: bx lr
  call void @llvm.codeview.annotation(metadata !12), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.codeview.annotation(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "pr190497.c", directory: ".")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!12 = !{!"annotation"}
!13 = !DILocation(line: 2, scope: !8)
!14 = !DILocation(line: 3, scope: !8)

; CHECK: .short	4121                            @ Record kind: S_ANNOTATION
; CHECK-NEXT: .secrel32	{{.*}}annotation0
; CHECK-NEXT: .secidx	{{.*}}annotation0
; CHECK-NEXT: .short	1
; CHECK-NEXT: .asciz	"annotation"
