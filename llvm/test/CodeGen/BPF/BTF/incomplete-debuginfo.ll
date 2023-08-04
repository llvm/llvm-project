; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   unsigned long foo(void) {
;     return 42;
;   }
;   unsigned long bar(void) {
;     return 1337;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c -o - | sed -E 's/,? !dbg !(7|11)//'

; Function Attrs: noinline nounwind optnone
define dso_local i64 @foo() #0 {
entry:
  ret i64 42
}

; Function Attrs: noinline nounwind optnone
define dso_local i64 @bar() #0 !dbg !12 {
entry:
  ret i64 1337, !dbg !13
}

; CHECK: .section        .BTF,"",@progbits
; CHECK: .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK: .long   1                               # BTF_KIND_INT(id = 2)
; CHECK: .long   15                              # BTF_KIND_FUNC(id = 3)
; CHECK: .byte   0                               # string offset=0
; CHECK: .ascii  "unsigned long"                 # string offset=1
; CHECK: .ascii  "bar"                           # string offset=15
; CHECK: .ascii  ".text"                         # string offset=19

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (https://github.com/llvm/llvm-project.git 8031b3f2c40d3fe622648b6731a0ae1dc3f37860)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/tamird/src/llvm-project-latest", checksumkind: CSK_MD5, checksum: "20bd72139e533d6aed61683730100513")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git 8031b3f2c40d3fe622648b6731a0ae1dc3f37860)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 2, column: 3, scope: !7)
!12 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 6, column: 3, scope: !12)
