; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; Generated from:
;     void stack_offsets() {
;       asm("" ::: "d8", "d9", "d11", "d13");
;     }
; Compiled with: "clang -target armv7-linux-gnueabihf -O3"

; The important point we're checking here is that the .cfi directives describe
; the layout of the VFP registers correctly. The fact that the numbers are
; monotonic in memory is also a nice property to have.

define void @stack_offsets() !dbg !4 {
; CHECK-LABEL: stack_offsets:
; CHECK: vpush {d13}
; CHECK: vpush {d11}
; CHECK: vpush {d8, d9}

; CHECK: .cfi_offset {{269|d13}}, -8
; CHECK: .cfi_offset {{267|d11}}, -16
; CHECK: .cfi_offset {{265|d9}}, -24
; CHECK: .cfi_offset {{264|d8}}, -32

; CHECK: vpop {d8, d9}
; CHECK: vpop {d11}
; CHECK: vpop {d13}
  call void asm sideeffect "", "~{d8},~{d9},~{d11},~{d13}"() #1
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "tmp.c", directory: "/Users/tim/llvm/build")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "bar", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "tmp.c", directory: "/Users/tim/llvm/build")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}

