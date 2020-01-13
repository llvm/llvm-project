; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   void (*a1)(int p1);
;   struct t1 { void (*a1)(int p1); } b1;
;   void f1(int p2) { }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t1 = type { void (i32)* }

@a1 = common dso_local local_unnamed_addr global void (i32)* null, align 8, !dbg !0
@b1 = common dso_local local_unnamed_addr global %struct.t1 zeroinitializer, align 8, !dbg !6

; Function Attrs: nounwind readnone
define dso_local void @f1(i32 %p2) local_unnamed_addr #0 !dbg !19 {
entry:
  call void @llvm.dbg.value(metadata i32 %p2, metadata !21, metadata !DIExpression()), !dbg !22
  ret void, !dbg !23
}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   104
; CHECK-NEXT:        .long   104
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   1                       # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                # 0x1000020
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 2)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   8                       # BTF_KIND_FUNC(id = 3)
; CHECK-NEXT:        .long   201326593               # 0xc000001
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   0                       # BTF_KIND_FUNC_PROTO(id = 5)
; CHECK-NEXT:        .long   218103809               # 0xd000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   26                      # BTF_KIND_STRUCT(id = 6)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   29
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                       # 0x0
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "int"                   # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "p2"                    # string offset=5
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "f1"                    # string offset=8
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  ".text"                 # string offset=11
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "/tmp/t.c"              # string offset=17
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "t1"                    # string offset=26
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "a1"                    # string offset=29
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   28
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   11                      # FuncInfo section string offset=11
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Lfunc_begin0
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   16                      # LineInfo
; CHECK-NEXT:        .long   11                      # LineInfo section string offset=11
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   17
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   3091                    # Line 3 Col 19

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a1", scope: !2, file: !3, line: 1, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/tmp")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b1", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 2, size: 64, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !8, file: !3, line: 2, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
!19 = distinct !DISubprogram(name: "f1", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !20)
!20 = !{!21}
!21 = !DILocalVariable(name: "p2", arg: 1, scope: !19, file: !3, line: 3, type: !14)
!22 = !DILocation(line: 3, column: 13, scope: !19)
!23 = !DILocation(line: 3, column: 19, scope: !19)
