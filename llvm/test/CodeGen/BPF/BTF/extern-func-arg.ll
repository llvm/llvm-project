; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern int global_func(char arg);
;   int test() { return global_func(0); }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !13 {
entry:
  %call = tail call i32 @global_func(i8 signext 0) #2, !dbg !16
  ret i32 %call, !dbg !17
}

; CHECK:             .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   88
; CHECK-NEXT:        .long   72
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
; CHECK:             .ascii  "int"                   # string offset=1
; CHECK:             .ascii  "test"                  # string offset=5
; CHECK:             .ascii  "char"                  # string offset=55
; CHECK:             .ascii  "global_func"           # string offset=60

declare !dbg !4 dso_local i32 @global_func(i8 signext) local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 987c5665e81822b32c895fd0c97a9a084b0d3106)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/extern")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "global_func", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 987c5665e81822b32c895fd0c97a9a084b0d3106)"}
!13 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 2, type: !14, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!7}
!16 = !DILocation(line: 2, column: 21, scope: !13)
!17 = !DILocation(line: 2, column: 14, scope: !13)
