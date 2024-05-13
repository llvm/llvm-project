; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   char g __attribute__((weak)) = 2;
;   int test() {
;     return g;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@g = weak dso_local local_unnamed_addr global i8 2, align 1, !dbg !0
; Function Attrs: norecurse nounwind readonly
define dso_local i32 @test() local_unnamed_addr #0 !dbg !11 {
entry:
  %0 = load i8, ptr @g, align 1, !dbg !15, !tbaa !16
  %conv = sext i8 %0 to i32, !dbg !15
  ret i32 %conv, !dbg !19
}

; CHECK:             .long   55                      # BTF_KIND_INT(id = 4)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   60                      # BTF_KIND_VAR(id = 5)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   62                      # BTF_KIND_DATASEC(id = 6)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   g
; CHECK-NEXT:        .long   1

; CHECK:             .ascii  "char"                  # string offset=55
; CHECK:             .byte   103                     # string offset=60
; CHECK:             .ascii  ".data"                 # string offset=62


attributes #0 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git edf6717d8d30034da932b95350898e03c90a5082)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/global")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git edf6717d8d30034da932b95350898e03c90a5082)"}
!11 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 2, type: !12, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, column: 10, scope: !11)
!16 = !{!17, !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 3, column: 3, scope: !11)
