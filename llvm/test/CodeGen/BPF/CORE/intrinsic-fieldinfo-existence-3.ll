; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   typedef struct s1 { int a1[10][10]; } __s1;
;   union u1 { int b1; __s1 b2; };
;   enum { FIELD_EXISTENCE = 2, };
;   int test(union u1 *arg) {
;     unsigned r1 = __builtin_preserve_field_info(arg->b2.a1[5], FIELD_EXISTENCE);
;     unsigned r2 = __builtin_preserve_field_info(arg->b2.a1[5][5], FIELD_EXISTENCE);
;     return r1 + r2;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%union.u1 = type { %struct.s1 }
%struct.s1 = type { [10 x [10 x i32]] }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%union.u1* %arg) local_unnamed_addr #0 !dbg !18 {
entry:
  call void @llvm.dbg.value(metadata %union.u1* %arg, metadata !31, metadata !DIExpression()), !dbg !34
  %0 = tail call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %arg, i32 1), !dbg !35, !llvm.preserve.access.index !22
  %b2 = getelementptr inbounds %union.u1, %union.u1* %0, i64 0, i32 0, !dbg !35
  %1 = tail call [10 x [10 x i32]]* @llvm.preserve.struct.access.index.p0a10a10i32.p0s_struct.s1s(%struct.s1* %b2, i32 0, i32 0), !dbg !36, !llvm.preserve.access.index !27
  %2 = tail call [10 x i32]* @llvm.preserve.array.access.index.p0a10i32.p0a10a10i32([10 x [10 x i32]]* %1, i32 1, i32 5), !dbg !37, !llvm.preserve.access.index !8
  %3 = tail call i32 @llvm.bpf.preserve.field.info.p0a10i32([10 x i32]* %2, i64 2), !dbg !38
  call void @llvm.dbg.value(metadata i32 %3, metadata !32, metadata !DIExpression()), !dbg !34
  %4 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a10i32([10 x i32]* %2, i32 1, i32 5), !dbg !39, !llvm.preserve.access.index !12
  %5 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %4, i64 2), !dbg !40
  call void @llvm.dbg.value(metadata i32 %5, metadata !33, metadata !DIExpression()), !dbg !34
  %add = add i32 %5, %3, !dbg !41
  ret i32 %add, !dbg !42
}

; CHECK:             r1 = 1
; CHECK:             r0 = 1
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
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   105
; CHECK-NEXT:        .long   2

; Function Attrs: nounwind readnone
declare %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1*, i32) #1

; Function Attrs: nounwind readnone
declare [10 x [10 x i32]]* @llvm.preserve.struct.access.index.p0a10a10i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare [10 x i32]* @llvm.preserve.array.access.index.p0a10i32.p0a10a10i32([10 x [10 x i32]]*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0a10i32([10 x i32]*, i64) #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0a10i32([10 x i32]*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

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
!6 = !DIEnumerator(name: "FIELD_EXISTENCE", value: 2, isUnsigned: true)
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
!41 = !DILocation(line: 7, column: 13, scope: !18)
!42 = !DILocation(line: 7, column: 3, scope: !18)
