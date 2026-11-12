; RUN: llc < %s -mtriple=s390x-ibm-zos --generate-arange-section | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos --generate-arange-section \
; RUN:   -filetype=obj -o /dev/null
;
; The stack restore stub follows the last function. With debug information,
; it must be emitted before the code section is ended for the address ranges.

; CHECK-LABEL: f DS 0H
; CHECK:       larl 3,[[STUB:L#stackrestore[0-9]+]]
; CHECK:      * XPLINK Stub Entry Marker
; CHECK:      [[STUB]] DS 0H
; CHECK:       b 2(3)

declare void @use(ptr)

define void @f(i64 %n) !dbg !4 {
  %sp = call ptr @llvm.stacksave.p0(), !dbg !7
  %p = alloca i8, i64 %n, align 8, !dbg !7
  call void @use(ptr %p), !dbg !7
  call void @llvm.stackrestore.p0(ptr %sp), !dbg !7
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 1, scope: !4)
