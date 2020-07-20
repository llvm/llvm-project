; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   typedef int * int_ptr;
;   static int * volatile v1;
;   static const int * volatile v2;
;   static volatile int_ptr v3 = 0;
;   static volatile const int_ptr v4 = 0;
;   long foo() { return (long)(v1 - v2 + v3 - v4); }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@v1 = internal global i32* null, align 8, !dbg !0
@v2 = internal global i32* null, align 8, !dbg !8
@v3 = internal global i32* null, align 8, !dbg !14
@v4 = internal constant i32* null, align 8, !dbg !19

; Function Attrs: norecurse nounwind
define dso_local i64 @foo() local_unnamed_addr #0 !dbg !27 {
  %1 = load volatile i32*, i32** @v1, align 8, !dbg !29, !tbaa !30
  %2 = load volatile i32*, i32** @v2, align 8, !dbg !34, !tbaa !30
  %3 = ptrtoint i32* %1 to i64, !dbg !35
  %4 = ptrtoint i32* %2 to i64, !dbg !35
  %5 = sub i64 %3, %4, !dbg !35
  %6 = ashr exact i64 %5, 2, !dbg !35
  %7 = load volatile i32*, i32** @v3, align 8, !dbg !36, !tbaa !30
  %8 = getelementptr inbounds i32, i32* %7, i64 %6, !dbg !37
  %9 = load volatile i32*, i32** @v4, align 8, !dbg !38, !tbaa !30
  %10 = ptrtoint i32* %8 to i64, !dbg !39
  %11 = ptrtoint i32* %9 to i64, !dbg !39
  %12 = sub i64 %10, %11, !dbg !39
  %13 = ashr exact i64 %12, 2, !dbg !39
  ret i64 %13, !dbg !40
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   288
; CHECK-NEXT:        .long   288
; CHECK-NEXT:        .long   95
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:        .long   218103808               # 0xd000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   16777280                # 0x1000040
; CHECK-NEXT:        .long   10                      # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 4)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 5)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   58                      # BTF_KIND_INT(id = 6)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   62                      # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 8)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 9)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   0                       # BTF_KIND_CONST(id = 10)
; CHECK-NEXT:        .long   167772160               # 0xa000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   65                      # BTF_KIND_VAR(id = 11)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 12)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   13
; CHECK-NEXT:        .long   68                      # BTF_KIND_TYPEDEF(id = 13)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   76                      # BTF_KIND_VAR(id = 14)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_CONST(id = 15)
; CHECK-NEXT:        .long   167772160               # 0xa000000
; CHECK-NEXT:        .long   12
; CHECK-NEXT:        .long   79                      # BTF_KIND_VAR(id = 16)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   15
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   82                      # BTF_KIND_DATASEC(id = 17)
; CHECK-NEXT:        .long   251658243               # 0xf000003
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   v1
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   v2
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   14
; CHECK-NEXT:        .long   v3
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   87                      # BTF_KIND_DATASEC(id = 18)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   16
; CHECK-NEXT:        .long   v4
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "long int"              # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "foo"                   # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=14
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/home/yhs/work/tests/llvm/bugs/test.c" # string offset=20
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int"                   # string offset=58
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "v1"                    # string offset=62
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "v2"                    # string offset=65
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "int_ptr"               # string offset=68
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "v3"                    # string offset=76
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "v4"                    # string offset=79
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".bss"                  # string offset=82
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".rodata"               # string offset=87
; CHECK-NEXT:        .byte   0

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!23, !24, !25}
!llvm.ident = !{!26}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v1", scope: !2, file: !3, line: 2, type: !22, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !7, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bugs")
!4 = !{}
!5 = !{!6}
!6 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!7 = !{!0, !8, !14, !19}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "v2", scope: !2, file: !3, line: 3, type: !10, isLocal: true, isDefinition: true)
!10 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "v3", scope: !2, file: !3, line: 4, type: !16, isLocal: true, isDefinition: true)
!16 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "int_ptr", file: !3, line: 1, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "v4", scope: !2, file: !3, line: 5, type: !21, isLocal: true, isDefinition: true)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !18)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{i32 1, !"wchar_size", i32 4}
!26 = !{!"clang version 8.0.20181009 "}
!27 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 6, type: !28, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !2, retainedNodes: !4)
!28 = !DISubroutineType(types: !5)
!29 = !DILocation(line: 6, column: 28, scope: !27)
!30 = !{!31, !31, i64 0}
!31 = !{!"any pointer", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !DILocation(line: 6, column: 33, scope: !27)
!35 = !DILocation(line: 6, column: 31, scope: !27)
!36 = !DILocation(line: 6, column: 38, scope: !27)
!37 = !DILocation(line: 6, column: 36, scope: !27)
!38 = !DILocation(line: 6, column: 43, scope: !27)
!39 = !DILocation(line: 6, column: 41, scope: !27)
!40 = !DILocation(line: 6, column: 14, scope: !27)
