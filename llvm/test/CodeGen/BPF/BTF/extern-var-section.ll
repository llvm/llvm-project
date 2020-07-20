; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern int global_func(char c) __attribute__((section("abc")));
;   extern char ch __attribute__((section("abc")));
;   int test() {
;     return global_func(0) + ch;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@ch = external dso_local local_unnamed_addr global i8, section "abc", align 1, !dbg !0

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !16 {
entry:
  %call = tail call i32 @global_func(i8 signext 0) #2, !dbg !19
  %0 = load i8, i8* @ch, align 1, !dbg !20, !tbaa !21
  %conv = sext i8 %0 to i32, !dbg !20
  %add = add nsw i32 %call, %conv, !dbg !24
  ret i32 %add, !dbg !25
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   128
; CHECK-NEXT:        .long   128
; CHECK-NEXT:        .long   79
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 4)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   55                      # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   60                      # BTF_KIND_FUNC(id = 6)
; CHECK-NEXT:        .long   201326594               # 0xc000002
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   72                      # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   75                      # BTF_KIND_DATASEC(id = 8)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   ch
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "test"                  # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/home/yhs/work/tests/extern/test.c" # string offset=16
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=55
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "global_func"           # string offset=60
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "ch"                    # string offset=72
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "abc"                   # string offset=75
; CHECK-NEXT:        .byte   0

declare !dbg !6 dso_local i32 @global_func(i8 signext) local_unnamed_addr #1 section "abc"

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "ch", scope: !2, file: !3, line: 2, type: !10, isLocal: false, isDefinition: false)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 0ad024346185b3f0b5167438e126568982b1168d)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !11, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/extern")
!4 = !{}
!5 = !{!6}
!6 = !DISubprogram(name: "global_func", scope: !3, file: !3, line: 1, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!11 = !{!0}
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 0ad024346185b3f0b5167438e126568982b1168d)"}
!16 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{!9}
!19 = !DILocation(line: 4, column: 10, scope: !16)
!20 = !DILocation(line: 4, column: 27, scope: !16)
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 4, column: 25, scope: !16)
!25 = !DILocation(line: 4, column: 3, scope: !16)
