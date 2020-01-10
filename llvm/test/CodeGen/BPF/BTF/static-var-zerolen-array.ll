; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct t {
;     int a;
;     int b;
;     char c[];
;   };
;   static volatile struct t sv = {3, 4, "abcdefghi"};
;   int test() { return sv.a; }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@sv = internal global { i32, i32, [10 x i8] } { i32 3, i32 4, [10 x i8] c"abcdefghi\00" }, align 4, !dbg !0

; Function Attrs: norecurse nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !21 {
  %1 = load volatile i32, i32* getelementptr inbounds ({ i32, i32, [10 x i8] }, { i32, i32, [10 x i8] }* @sv, i64 0, i32 0), align 4, !dbg !24, !tbaa !25
  ret i32 %1, !dbg !29
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   196
; CHECK-NEXT:        .long   196
; CHECK-NEXT:        .long   95
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
; CHECK-NEXT:        .long   0                       # BTF_KIND_VOLATILE(id = 4)
; CHECK-NEXT:        .long   150994944               # 0x9000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   53                      # BTF_KIND_STRUCT(id = 5)
; CHECK-NEXT:        .long   67108867                # 0x4000003
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   55
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .long   57
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   59
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   64                      # 0x40
; CHECK-NEXT:        .long   61                      # BTF_KIND_INT(id = 6)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   16777224                # 0x1000008
; CHECK-NEXT:        .long   0                       # BTF_KIND_ARRAY(id = 7)
; CHECK-NEXT:        .long   50331648                # 0x3000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   66                      # BTF_KIND_INT(id = 8)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   32                      # 0x20
; CHECK-NEXT:        .long   86                      # BTF_KIND_VAR(id = 9)
; CHECK-NEXT:        .long   234881024               # 0xe000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   89                      # BTF_KIND_DATASEC(id = 10)
; CHECK-NEXT:        .long   251658241               # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   sv
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "test"                  # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=10
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/home/yhs/work/tests/llvm/bug/test.c" # string offset=16
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   116                     # string offset=53
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   97                      # string offset=55
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   98                      # string offset=57
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   99                      # string offset=59
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "char"                  # string offset=61
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "__ARRAY_SIZE_TYPE__"   # string offset=66
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "sv"                    # string offset=86
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".data"                 # string offset=89
; CHECK-NEXT:        .byte   0

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sv", scope: !2, file: !3, line: 6, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.20181009 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/yhs/work/tests/llvm/bug")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !3, line: 1, size: 64, elements: !8)
!8 = !{!9, !11, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 2, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !7, file: !3, line: 3, baseType: !10, size: 32, offset: 32)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !7, file: !3, line: 4, baseType: !13, offset: 64)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, elements: !15)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !{!16}
!16 = !DISubrange(count: -1)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{!"clang version 8.0.20181009 "}
!21 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 7, type: !22, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, unit: !2, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{!10}
!24 = !DILocation(line: 7, column: 24, scope: !21)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 7, column: 14, scope: !21)
