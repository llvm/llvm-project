;; This test checks output of the DWARF embedded source.

; REQUIRES: x86-registered-target

; RUN: llc -filetype=obj -o %t.o %s 

;; Check LLVM style output.
; RUN: llvm-symbolizer --print-source-context-lines=3 --obj=%t.o 0 | \
; RUN:   FileCheck %s --check-prefixes=COMMON,LLVM --strict-whitespace --match-full-lines --implicit-check-not={{.}}

;; Check GNU output style.
; RUN: llvm-symbolizer --print-source-context-lines=3 --obj=%t.o 0 --output-style=GNU | \
; RUN:   FileCheck %s --check-prefixes=COMMON,GNU --strict-whitespace --match-full-lines --implicit-check-not={{.}}

;      COMMON:foo
;   LLVM-NEXT:/source.c:3:13
;    GNU-NEXT:/source.c:3
; COMMON-NEXT:2  : // Line 2
; COMMON-NEXT:3 >: void foo() {}
; COMMON-NEXT:4  : // Line 4

;; Check JSON style output.
; RUN: llvm-symbolizer --print-source-context-lines=3 --obj=%t.o 0 --output-style=JSON | \
; RUN:   FileCheck %s --check-prefix=JSON --strict-whitespace --match-full-lines --implicit-check-not={{.}}
; JSON:[{"Address":"0x0","ModuleName":"{{.*}}.o","Symbol":[{"Column":13,"Discriminator":0,"FileName":"/source.c","FunctionName":"foo","Line":3,"Source":"2  : // Line 2\n3 >: void foo() {}\n4  : // Line 4\n","StartAddress":"0x0","StartFileName":"/source.c","StartLine":3}]}]

;; Generated from the following source:
;; // Line 1
;; // Line 2
;; void foo() {}
;; // Line 4
;; // Line 5
;; clang --target=x86_64-pc-linux -gdwarf-5 -gembed-source -g -emit-llvm -S source.c -o source.ll

source_filename = "source.c"
target triple = "x86_64-pc-linux"

define dso_local void @foo() #0 !dbg !7 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "source.c", directory: "/", source: "// Line 1\0A// Line 2\0Avoid foo() {}\0A// Line 4\0A// Line 5\0A")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 3, column: 13, scope: !7)
