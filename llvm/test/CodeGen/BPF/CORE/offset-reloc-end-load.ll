; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
;
; Source Code:
;   #define _(x) (__builtin_preserve_access_index(x))
;   struct s {int a; int b;};
;   int test(struct s *arg) { return *(const int *)_(&arg->b); }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.s = type { i32, i32 }

; Function Attrs: nounwind readonly
define dso_local i32 @test(ptr readonly %arg) local_unnamed_addr #0 !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr elementtype(%struct.s) %arg, i32 1, i32 1), !dbg !22, !llvm.preserve.access.index !15
  %1 = load i32, ptr %0, align 4, !dbg !23, !tbaa !24
  ret i32 %1, !dbg !28
}

; CHECK-LABEL: test
; CHECK-ALU64: r0 = *(u32 *)(r1 + 4)
; CHECK-ALU32: w0 = *(u32 *)(r1 + 4)
; CHECK:       exit
;
; CHECK:       .long   1                       # BTF_KIND_STRUCT(id = 2)
;
; CHECK:       .byte   115                     # string offset=1
; CHECK:       .ascii  ".text"                 # string offset=20
; CHECK:       .ascii  "0:1"                   # string offset=26
;
; CHECK:       .long   16                      # FieldReloc
; CHECK-NEXT:  .long   20                      # Field reloc section string offset=20
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:  .long   2
; CHECK-NEXT:  .long   26
; CHECK-NEXT:  .long   0

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr, i32, i32) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6e353b4df3aa452ed4741a5e5caea02b1a876d8c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6e353b4df3aa452ed4741a5e5caea02b1a876d8c)"}
!11 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !19)
!12 = !DISubroutineType(types: !13)
!13 = !{!6, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 2, size: 64, elements: !16)
!16 = !{!17, !18}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 2, baseType: !6, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !15, file: !1, line: 2, baseType: !6, size: 32, offset: 32)
!19 = !{!20}
!20 = !DILocalVariable(name: "arg", arg: 1, scope: !11, file: !1, line: 3, type: !14)
!21 = !DILocation(line: 0, scope: !11)
!22 = !DILocation(line: 3, column: 48, scope: !11)
!23 = !DILocation(line: 3, column: 34, scope: !11)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DILocation(line: 3, column: 27, scope: !11)
