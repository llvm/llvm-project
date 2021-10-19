; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   enum A { AA = -1, AB = 0, };
;   enum B { BA = 0, BB = 1, };
;   typedef enum A __A;
;   typedef enum B __B;
;   typedef struct s1 { __A a1[10]; __B a2[10][10]; } __s1;
;   union u1 { int b1; __s1 b2; };
;   enum { FIELD_SIGNEDNESS = 3, };
;   int test(union u1 *arg) {
;     unsigned r1 = __builtin_preserve_field_info(arg->b2.a1[5], FIELD_SIGNEDNESS);
;     unsigned r2 = __builtin_preserve_field_info(arg->b2.a2[5][5], FIELD_SIGNEDNESS);
;     /* r1 : 1, r2 : 0 */
;     return r1 + r2;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%union.u1 = type { %struct.s1 }
%struct.s1 = type { [10 x i32], [10 x [10 x i32]] }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%union.u1* %arg) local_unnamed_addr #0 !dbg !29 {
entry:
  call void @llvm.dbg.value(metadata %union.u1* %arg, metadata !43, metadata !DIExpression()), !dbg !46
  %0 = tail call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %arg, i32 1), !dbg !47, !llvm.preserve.access.index !33
  %b2 = getelementptr inbounds %union.u1, %union.u1* %0, i64 0, i32 0, !dbg !47
  %1 = tail call [10 x i32]* @llvm.preserve.struct.access.index.p0a10i32.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %b2, i32 0, i32 0), !dbg !48, !llvm.preserve.access.index !38
  %2 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a10i32([10 x i32]* elementtype([10 x i32]) %1, i32 1, i32 5), !dbg !49, !llvm.preserve.access.index !17
  %3 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %2, i64 3), !dbg !50
  call void @llvm.dbg.value(metadata i32 %3, metadata !44, metadata !DIExpression()), !dbg !46
  %4 = tail call [10 x [10 x i32]]* @llvm.preserve.struct.access.index.p0a10a10i32.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %b2, i32 1, i32 1), !dbg !51, !llvm.preserve.access.index !38
  %5 = tail call [10 x i32]* @llvm.preserve.array.access.index.p0a10i32.p0a10a10i32([10 x [10 x i32]]* elementtype([10 x [10 x i32]]) %4, i32 1, i32 5), !dbg !52, !llvm.preserve.access.index !21
  %6 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0a10i32([10 x i32]* elementtype([10 x i32]) %5, i32 1, i32 5), !dbg !52, !llvm.preserve.access.index !24
  %7 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %6, i64 3), !dbg !53
  call void @llvm.dbg.value(metadata i32 %7, metadata !45, metadata !DIExpression()), !dbg !46
  %add = add i32 %7, %3, !dbg !54
  ret i32 %add, !dbg !55
}

; CHECK:             r1 = 1
; CHECK:             r0 = 0
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_UNION(id = 2)
; CHECK:             .ascii  "u1"                    # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=81
; CHECK:             .ascii  "0:1:0:5"               # string offset=87
; CHECK:             .ascii  "0:1:1:5:5"             # string offset=132

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   81                      # Field reloc section string offset=81
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   87
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   132
; CHECK-NEXT:        .long   3

; Function Attrs: nounwind readnone
declare %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1*, i32) #1

; Function Attrs: nounwind readnone
declare [10 x i32]* @llvm.preserve.struct.access.index.p0a10i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0a10i32([10 x i32]*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

; Function Attrs: nounwind readnone
declare [10 x [10 x i32]]* @llvm.preserve.struct.access.index.p0a10a10i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare [10 x i32]* @llvm.preserve.array.access.index.p0a10i32.p0a10a10i32([10 x [10 x i32]]*, i32, i32) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26, !27}
!llvm.ident = !{!28}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git c1e02f16f1105ffaf1c35ee8bc38b7d6db5c6ea9)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !16, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3, !8, !13}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "A", file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "AA", value: -1)
!7 = !DIEnumerator(name: "AB", value: 0)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "B", file: !1, line: 2, baseType: !9, size: 32, elements: !10)
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11, !12}
!11 = !DIEnumerator(name: "BA", value: 0, isUnsigned: true)
!12 = !DIEnumerator(name: "BB", value: 1, isUnsigned: true)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 7, baseType: !9, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIEnumerator(name: "FIELD_SIGNEDNESS", value: 3, isUnsigned: true)
!16 = !{!17, !21, !24}
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 320, elements: !19)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "__A", file: !1, line: 3, baseType: !3)
!19 = !{!20}
!20 = !DISubrange(count: 10)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 3200, elements: !23)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "__B", file: !1, line: 4, baseType: !8)
!23 = !{!20, !20}
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 320, elements: !19)
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 4}
!28 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git c1e02f16f1105ffaf1c35ee8bc38b7d6db5c6ea9)"}
!29 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 8, type: !30, scopeLine: 8, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !42)
!30 = !DISubroutineType(types: !31)
!31 = !{!4, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!33 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1", file: !1, line: 6, size: 3520, elements: !34)
!34 = !{!35, !36}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !33, file: !1, line: 6, baseType: !4, size: 32)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !33, file: !1, line: 6, baseType: !37, size: 3520)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s1", file: !1, line: 5, baseType: !38)
!38 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 5, size: 3520, elements: !39)
!39 = !{!40, !41}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !38, file: !1, line: 5, baseType: !17, size: 320)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !38, file: !1, line: 5, baseType: !21, size: 3200, offset: 320)
!42 = !{!43, !44, !45}
!43 = !DILocalVariable(name: "arg", arg: 1, scope: !29, file: !1, line: 8, type: !32)
!44 = !DILocalVariable(name: "r1", scope: !29, file: !1, line: 9, type: !9)
!45 = !DILocalVariable(name: "r2", scope: !29, file: !1, line: 10, type: !9)
!46 = !DILocation(line: 0, scope: !29)
!47 = !DILocation(line: 9, column: 52, scope: !29)
!48 = !DILocation(line: 9, column: 55, scope: !29)
!49 = !DILocation(line: 9, column: 47, scope: !29)
!50 = !DILocation(line: 9, column: 17, scope: !29)
!51 = !DILocation(line: 10, column: 55, scope: !29)
!52 = !DILocation(line: 10, column: 47, scope: !29)
!53 = !DILocation(line: 10, column: 17, scope: !29)
!54 = !DILocation(line: 12, column: 13, scope: !29)
!55 = !DILocation(line: 12, column: 3, scope: !29)
