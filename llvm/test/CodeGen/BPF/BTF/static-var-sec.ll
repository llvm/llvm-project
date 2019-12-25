; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   static volatile char a __attribute__((section("maps")));
;   int foo() {
;     static volatile short b __attribute__((section("maps")));
;     return a + b;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@foo.b = internal global i16 0, section "maps", align 2, !dbg !0
@a = internal global i8 0, section "maps", align 1, !dbg !10

; Function Attrs: norecurse nounwind
define dso_local i32 @foo() local_unnamed_addr #0 !dbg !2 {
  %1 = load volatile i8, i8* @a, align 1, !dbg !20, !tbaa !21
  %2 = sext i8 %1 to i32, !dbg !20
  %3 = load volatile i16, i16* @foo.b, align 2, !dbg !24, !tbaa !25
  %4 = sext i16 %3 to i32, !dbg !24
  %5 = add nsw i32 %4, %2, !dbg !27
  ret i32 %5, !dbg !28
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   164
; CHECK-NEXT:        .long   164
; CHECK-NEXT:        .long   76
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   5                       # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326592               # 0xc000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 4)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   52                      # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   16777232                # 0x1000010
; CHECK-NEXT:        .long   58                      # BTF_KIND_VAR(id = 6)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 7)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   64                      # BTF_KIND_INT(id = 8)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   69                      # BTF_KIND_VAR(id = 9)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   71                      # BTF_KIND_DATASEC(id = 10)
; CHECK-NEXT:        .long   251658242               # 0xf000002
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   foo.b
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   a
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo"                   # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=9
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/home/yhs/work/tests/llvm/bug/test.c" # string offset=15
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "short"                 # string offset=52
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo.b"                 # string offset=58
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=64
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=69
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "maps"                  # string offset=71
; CHECK-NEXT:        .byte   0

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !14, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9, nameTableKind: None)
!8 = !{}
!9 = !{!0, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "a", scope: !7, file: !3, line: 1, type: !12, isLocal: true, isDefinition: true)
!12 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !13)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !15)
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 8.0.20181009 "}
!20 = !DILocation(line: 4, column: 10, scope: !2)
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 4, column: 14, scope: !2)
!25 = !{!26, !26, i64 0}
!26 = !{!"short", !22, i64 0}
!27 = !DILocation(line: 4, column: 12, scope: !2)
!28 = !DILocation(line: 4, column: 3, scope: !2)
