; With LTO a function can be inlined into another compile unit and then have no
; copy of its own left, so its abstract subprogram is only kept for a reference
; from that other unit. The units are marked concurrently, and which static
; variables of the function are kept must not depend on how far the unit holding
; them has come. Here "live" has a symbol in the linked binary and "dead" does
; not.
;
; a.c:
;   int f(void) {
;     static int live, dead;
;     return live;
;   }
; main.c:
;   int main(void) { return f(); }

; RUN: rm -rf %t && split-file %s %t
; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input.ll -o %t/input.o

; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/threads.dwarf
; RUN: llvm-dwarfdump --show-parents --name=live --name=dead %t/threads.dwarf | FileCheck %s --implicit-check-not=DW_TAG_
; RUN: dsymutil -f --linker=parallel --num-threads 1 -oso-prepend-path=%t -y %t/input.map -o %t/serial.dwarf
; RUN: cmp %t/threads.dwarf %t/serial.dwarf

; The same with ODR uniquing, which marks the abstract subprogram for the type
; unit instead.
; RUN: sed 's/DW_LANG_C11/DW_LANG_C_plus_plus_14/' %t/input.ll > %t/input-cxx.ll
; RUN: %llc_dwarf -mtriple=arm64-apple-darwin -filetype=obj %t/input-cxx.ll -o %t/input.o
; RUN: dsymutil -f --linker=parallel -oso-prepend-path=%t -y %t/input.map -o %t/threads-cxx.dwarf
; RUN: llvm-dwarfdump --show-parents --name=live --name=dead %t/threads-cxx.dwarf | FileCheck %s --implicit-check-not=DW_TAG_
; RUN: dsymutil -f --linker=parallel --num-threads 1 -oso-prepend-path=%t -y %t/input.map -o %t/serial-cxx.dwarf
; RUN: cmp %t/threads-cxx.dwarf %t/serial-cxx.dwarf

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name ("a.c")
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("f")
; CHECK: DW_TAG_variable
; CHECK:   DW_AT_name ("live")
; CHECK:   DW_AT_location (DW_OP_addr 0x2000)

;--- input.ll
target triple = "arm64-apple-darwin"

@f.live = internal global i32 1, align 4, !dbg !0
@f.dead = internal global i32 1, align 4, !dbg !9

define i32 @main() !dbg !15 {
  %v = load i32, ptr @f.live, align 4, !dbg !16
  ret i32 %v, !dbg !18
}

!llvm.dbg.cu = !{!13, !7}
!llvm.module.flags = !{!11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "live", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "a.c", directory: "/tmp")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!8 = !{!0, !9}
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "dead", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true)
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DICompileUnit(language: DW_LANG_C11, file: !14, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!14 = !DIFile(filename: "main.c", directory: "/tmp")
!15 = distinct !DISubprogram(name: "main", scope: !14, file: !14, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !13)
!16 = !DILocation(line: 3, column: 3, scope: !2, inlinedAt: !17)
!17 = distinct !DILocation(line: 1, column: 25, scope: !15)
!18 = !DILocation(line: 1, column: 18, scope: !15)

;--- input.map
---
triple: 'arm64-apple-darwin'
objects:
  - filename: input.o
    symbols:
      - { sym: _main, objAddr: 0x0, binAddr: 0x1000, size: 0xc }
      - { sym: _f.live, objAddr: 0xc, binAddr: 0x2000, size: 0x4 }
...
