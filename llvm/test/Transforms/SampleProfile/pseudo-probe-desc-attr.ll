
; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=pseudo-probe -S -o %t
; RUN: FileCheck %s --input-file %t --check-prefix=CHECK-IL
; RUN: llc %t -mtriple=x86_64-unknown-linux-gnu -filetype=asm -o - | FileCheck %s --check-prefix=CHECK-ASM

;; The descriptors record always_inline (1) and noinline (2).
; CHECK-IL: ![[#]] = !{i64 {{-?[0-9]+}}, i64 4294967295, !"baz", i8 0}
; CHECK-IL: ![[#]] = !{i64 {{-?[0-9]+}}, i64 4294967295, !"bar", i8 1}
; CHECK-IL: ![[#]] = !{i64 {{-?[0-9]+}}, i64 4294967295, !"foo", i8 2}

;; In .pseudo_probe_desc the attributes take bits [60:63] of the hash field.
; CHECK-ASM:      .section .pseudo_probe_desc,"",@progbits
; CHECK-ASM-NEXT: .quad {{-?[0-9]+}}
; CHECK-ASM-NEXT: .quad 4294967295
; CHECK-ASM-NEXT: .byte 3
; CHECK-ASM-NEXT: .ascii "baz"
; CHECK-ASM-NEXT: .quad {{-?[0-9]+}}
;; (1 << 60) | 4294967295
; CHECK-ASM-NEXT: .quad 1152921508901814271
; CHECK-ASM-NEXT: .byte 3
; CHECK-ASM-NEXT: .ascii "bar"
; CHECK-ASM-NEXT: .quad {{-?[0-9]+}}
;; (2 << 60) | 4294967295
; CHECK-ASM-NEXT: .quad 2305843013508661247
; CHECK-ASM-NEXT: .byte 3
; CHECK-ASM-NEXT: .ascii "foo"

; Function Attrs: nounwind uwtable
define dso_local void @baz() #0 !dbg !11 {
entry:
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define dso_local void @bar() #1 !dbg !12 {
entry:
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() #2 !dbg !13 {
entry:
  ret void
}

attributes #0 = { nounwind uwtable}
attributes #1 = { alwaysinline nounwind uwtable}
attributes #2 = { noinline nounwind uwtable}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 17.0.0 "}
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!12 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 10, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!13 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 15, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
