; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -mcpu=v1 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -mcpu=v1 -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   typedef struct s1 { int a1[10][10]; } __s1;
;   union u1 { int b1; __s1 b2; };
;   enum { FIELD_BYTE_SIZE = 1, };
;   int test(union u1 *arg) {
;     unsigned r1 = __builtin_preserve_field_info(arg->b2.a1[5], FIELD_BYTE_SIZE);
;     unsigned r2 = __builtin_preserve_field_info(arg->b2.a1[5][5], FIELD_BYTE_SIZE);
;     /* r1: 40, r2: 4 */
;     return r1 + r2;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%union.u1 = type { %struct.s1 }
%struct.s1 = type { [10 x [10 x i32]] }

; Function Attrs: nounwind readnone
define dso_local i32 @test(ptr %arg) local_unnamed_addr #0 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !31, metadata !DIExpression()), !dbg !34
  %0 = tail call ptr @llvm.preserve.union.access.index.p0.u1s.p0.u1s(ptr %arg, i32 1), !dbg !35, !llvm.preserve.access.index !22
  %1 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.s1s(ptr elementtype(%struct.s1) %0, i32 0, i32 0), !dbg !36, !llvm.preserve.access.index !27
  %2 = tail call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([10 x [10 x i32]]) %1, i32 1, i32 5), !dbg !37, !llvm.preserve.access.index !8
  %3 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %2, i64 1), !dbg !38
  call void @llvm.dbg.value(metadata i32 %3, metadata !32, metadata !DIExpression()), !dbg !34
  %4 = tail call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([10 x i32]) %2, i32 1, i32 5), !dbg !39, !llvm.preserve.access.index !12
  %5 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %4, i64 1), !dbg !40
  call void @llvm.dbg.value(metadata i32 %5, metadata !33, metadata !DIExpression()), !dbg !34
  %add = add i32 %5, %3, !dbg !41
  ret i32 %add, !dbg !42
}

; CHECK:             r1 = 40
; CHECK:             r0 = 4
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_UNION(id = 2)
; CHECK:             .ascii  "u1"                    # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=54
; CHECK:             .ascii  "0:1:0:5"               # string offset=60
; CHECK:             .ascii  "0:1:0:5:5"             # string offset=105

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   54                      # Field reloc section string offset=54
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   60
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   105
; CHECK-NEXT:        .long   1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.union.access.index.p0.u1s.p0.u1s(ptr, i32) #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.s1s(ptr, i32, i32) #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64) #1

; Function Attrs: nounwind readnone

; Function Attrs: nounwind readnone

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git c1e02f16f1105ffaf1c35ee8bc38b7d6db5c6ea9)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !7, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 3, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FIELD_BYTE_SIZE", value: 1, isUnsigned: true)
!7 = !{!8, !12}
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 3200, elements: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !11}
!11 = !DISubrange(count: 10)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 320, elements: !13)
!13 = !{!11}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git c1e02f16f1105ffaf1c35ee8bc38b7d6db5c6ea9)"}
!18 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !19, scopeLine: 4, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !30)
!19 = !DISubroutineType(types: !20)
!20 = !{!9, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1", file: !1, line: 2, size: 3200, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !22, file: !1, line: 2, baseType: !9, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !22, file: !1, line: 2, baseType: !26, size: 3200)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s1", file: !1, line: 1, baseType: !27)
!27 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 1, size: 3200, elements: !28)
!28 = !{!29}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !27, file: !1, line: 1, baseType: !8, size: 3200)
!30 = !{!31, !32, !33}
!31 = !DILocalVariable(name: "arg", arg: 1, scope: !18, file: !1, line: 4, type: !21)
!32 = !DILocalVariable(name: "r1", scope: !18, file: !1, line: 5, type: !4)
!33 = !DILocalVariable(name: "r2", scope: !18, file: !1, line: 6, type: !4)
!34 = !DILocation(line: 0, scope: !18)
!35 = !DILocation(line: 5, column: 52, scope: !18)
!36 = !DILocation(line: 5, column: 55, scope: !18)
!37 = !DILocation(line: 5, column: 47, scope: !18)
!38 = !DILocation(line: 5, column: 17, scope: !18)
!39 = !DILocation(line: 6, column: 47, scope: !18)
!40 = !DILocation(line: 6, column: 17, scope: !18)
!41 = !DILocation(line: 8, column: 13, scope: !18)
!42 = !DILocation(line: 8, column: 3, scope: !18)
