; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct b { int d; int e; } c;
;   int f() {
;     return __builtin_preserve_field_info(c.e, 0);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.b = type { i32, i32 }

@c = common dso_local global %struct.b zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind readnone
define dso_local i32 @f() local_unnamed_addr #0 !dbg !15 {
entry:
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.bs(%struct.b* nonnull @c, i32 1, i32 1), !dbg !18, !llvm.preserve.access.index !6
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %0, i64 0), !dbg !19
  ret i32 %1, !dbg !20
}

; CHECK:             r0 = 4
; CHECK:             exit

; CHECK:             .long   13                      # BTF_KIND_STRUCT(id = 4)

; CHECK:             .ascii  ".text"                 # string offset=7
; CHECK:             .byte   98                      # string offset=13
; CHECK:             .ascii  "0:1"                   # string offset=19

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   7                       # Field reloc section string offset=7
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   19
; CHECK-NEXT:        .long   0

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.bs(%struct.b*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git d5509db439a1bb3f0822c42398c8b5921a665478)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "b", file: !3, line: 1, size: 64, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !6, file: !3, line: 1, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !6, file: !3, line: 1, baseType: !9, size: 32, offset: 32)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git d5509db439a1bb3f0822c42398c8b5921a665478)"}
!15 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 2, type: !16, scopeLine: 2, isDefinition: true, isOptimized: true, unit: !2, retainedNodes: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{!9}
!18 = !DILocation(line: 3, column: 42, scope: !15)
!19 = !DILocation(line: 3, column: 10, scope: !15)
!20 = !DILocation(line: 3, column: 3, scope: !15)
