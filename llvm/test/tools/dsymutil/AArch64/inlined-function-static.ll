; A static variable of a function that is also inlined is described once, under
; the abstract subprogram. That subprogram has no address, so it is only kept
; for being referenced, and the variable has to be kept along with it.
;
; int f(void) {
;   static int seen;
;   return seen;
; }
; int main(void) { return f(); }

; RUN: rm -rf %t && split-file %s %t
; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input.ll -o %t/input.o

; RUN: dsymutil -f --linker=classic -oso-prepend-path=%t -y %t/input.map -o %t/classic.dwarf
; RUN: llvm-dwarfdump --debug-info %t/classic.dwarf | FileCheck %s

; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/parallel.dwarf
; RUN: llvm-dwarfdump --debug-info %t/parallel.dwarf | FileCheck %s

; With ODR uniquing the abstract subprogram would move to the type unit. The
; variable has an address and cannot follow, so the subprogram stays with it.
; RUN: sed 's/DW_LANG_C11/DW_LANG_C_plus_plus_14/' %t/input.ll > %t/input-cxx.ll
; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input-cxx.ll -o %t/input.o
; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/parallel-cxx.dwarf
; RUN: llvm-dwarfdump --debug-info %t/parallel-cxx.dwarf | FileCheck %s
; RUN: llvm-dwarfdump --verify %t/parallel-cxx.dwarf

; The instances refer to the subprogram that holds the variable.
; CHECK: DW_AT_name ("a.c")
; CHECK: DW_AT_abstract_origin (0x[[F:[0-9a-f]+]] "f")
; CHECK: 0x[[F]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_name ("f")
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name ("seen")
; CHECK: DW_AT_location (DW_OP_addr 0x2000)
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin (0x[[F]] "f")

;--- input.ll
target triple = "arm64-apple-darwin"

@f.seen = internal global i32 1, align 4, !dbg !0

define i32 @f() !dbg !2 {
  %v = load i32, ptr @f.seen, align 4, !dbg !11
  ret i32 %v, !dbg !11
}

define i32 @main() !dbg !12 {
  %v = load i32, ptr @f.seen, align 4, !dbg !13
  ret i32 %v, !dbg !15
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!9, !10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "seen", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "a.c", directory: "/tmp")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!8 = !{!0}
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 3, column: 3, scope: !2)
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !7)
!13 = !DILocation(line: 3, column: 3, scope: !2, inlinedAt: !14)
!14 = distinct !DILocation(line: 5, column: 25, scope: !12)
!15 = !DILocation(line: 5, column: 18, scope: !12)

;--- input.map
---
triple: 'arm64-apple-darwin'
objects:
  - filename: input.o
    symbols:
      - { sym: _f, objAddr: 0x0, binAddr: 0x1000, size: 0xc }
      - { sym: _main, objAddr: 0xc, binAddr: 0x100c, size: 0xc }
      - { sym: _f.seen, objAddr: 0x18, binAddr: 0x2000, size: 0x4 }
...
