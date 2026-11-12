; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   const volatile char g __attribute__((weak)) = 2;
;   int test() {
;     return g;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@g = weak_odr dso_local constant i8 2, align 1, !dbg !0

; Function Attrs: nofree norecurse nounwind willreturn
define dso_local i32 @test() local_unnamed_addr #0 !dbg !13 {
entry:
  %0 = load volatile i8, ptr @g, align 1, !dbg !17, !tbaa !18
  %conv = sext i8 %0 to i32, !dbg !17
  ret i32 %conv, !dbg !21
}

; CHECK:             .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808                       # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                               # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   5                               # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # BTF_KIND_CONST(id = 4)
; CHECK-NEXT:        .long   167772160                       # 0xa000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                               # BTF_KIND_VOLATILE(id = 5)
; CHECK-NEXT:        .long   150994944                       # 0x9000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   47                              # BTF_KIND_INT(id = 6)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                        # 0x1000008
; CHECK-NEXT:        .long   52                              # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   54                              # BTF_KIND_DATASEC(id = 8)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   g
; CHECK-NEXT:        .long   1

; CHECK:             .ascii  "int"                           # string offset=1
; CHECK:             .ascii  "test"                          # string offset=5
; CHECK:             .ascii  "char"                          # string offset=47
; CHECK:             .byte   103                             # string offset=52
; CHECK:             .ascii  ".rodata"                       # string offset=54

attributes #0 = { nofree norecurse nounwind willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 9cc417cbca1cece0d55fa3d1e15682943a06139e)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/btf/tests")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !8)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 9cc417cbca1cece0d55fa3d1e15682943a06139e)"}
!13 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 2, type: !14, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 3, column: 10, scope: !13)
!18 = !{!19, !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!21 = !DILocation(line: 3, column: 3, scope: !13)
