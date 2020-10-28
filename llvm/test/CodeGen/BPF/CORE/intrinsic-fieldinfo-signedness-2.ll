; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   enum A { AA = -1, AB = 0, }; /* signed */
;   enum B { BA = 0, BB = 1, };  /* unsigned */
;   typedef enum A __A;
;   typedef enum B __B;
;   typedef int __int;           /* signed */
;   struct s1 { __A a1; __B a2:9; __int a3:4; };
;   union u1 { int b1; struct s1 b2; };
;   enum { FIELD_SIGNEDNESS = 3, };
;   int test(union u1 *arg) {
;     unsigned r1 = __builtin_preserve_field_info(arg->b2.a1, FIELD_SIGNEDNESS);
;     unsigned r2 = __builtin_preserve_field_info(arg->b2.a2, FIELD_SIGNEDNESS);
;     unsigned r3 = __builtin_preserve_field_info(arg->b2.a3, FIELD_SIGNEDNESS);
;     return r1 + r2 + r3;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%union.u1 = type { %struct.s1 }
%struct.s1 = type { i32, i16 }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%union.u1* %arg) local_unnamed_addr #0 !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata %union.u1* %arg, metadata !37, metadata !DIExpression()), !dbg !41
  %0 = tail call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %arg, i32 1), !dbg !42, !llvm.preserve.access.index !24
  %b2 = getelementptr inbounds %union.u1, %union.u1* %0, i64 0, i32 0, !dbg !42
  %1 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* %b2, i32 0, i32 0), !dbg !43, !llvm.preserve.access.index !28
  %2 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %1, i64 3), !dbg !44
  call void @llvm.dbg.value(metadata i32 %2, metadata !38, metadata !DIExpression()), !dbg !41
  %3 = tail call i16* @llvm.preserve.struct.access.index.p0i16.p0s_struct.s1s(%struct.s1* %b2, i32 1, i32 1), !dbg !45, !llvm.preserve.access.index !28
  %4 = tail call i32 @llvm.bpf.preserve.field.info.p0i16(i16* %3, i64 3), !dbg !46
  call void @llvm.dbg.value(metadata i32 %4, metadata !39, metadata !DIExpression()), !dbg !41
  %5 = tail call i16* @llvm.preserve.struct.access.index.p0i16.p0s_struct.s1s(%struct.s1* %b2, i32 1, i32 2), !dbg !47, !llvm.preserve.access.index !28
  %6 = tail call i32 @llvm.bpf.preserve.field.info.p0i16(i16* %5, i64 3), !dbg !48
  call void @llvm.dbg.value(metadata i32 %6, metadata !40, metadata !DIExpression()), !dbg !41
  %add = add i32 %4, %2, !dbg !49
  %add3 = add i32 %add, %6, !dbg !50
  ret i32 %add3, !dbg !51
}

; CHECK:             r1 = 1
; CHECK:             r0 = 0
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             r1 = 1
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_UNION(id = 2)
; CHECK:             .ascii  "u1"                    # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=65
; CHECK:             .ascii  "0:1:0"                 # string offset=71
; CHECK:             .ascii  "0:1:1"                 # string offset=114
; CHECK:             .ascii  "0:1:2"                 # string offset=120

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   65                      # Field reloc section string offset=65
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   71
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   114
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   120
; CHECK-NEXT:        .long   3

; Function Attrs: nounwind readnone
declare %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1*, i32) #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

; Function Attrs: nounwind readnone
declare i16* @llvm.preserve.struct.access.index.p0i16.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i16(i16*, i64) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4a60741b74384f14b21fdc0131ede326438840ab)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
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
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 8, baseType: !9, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIEnumerator(name: "FIELD_SIGNEDNESS", value: 3, isUnsigned: true)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4a60741b74384f14b21fdc0131ede326438840ab)"}
!20 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 9, type: !21, scopeLine: 9, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !36)
!21 = !DISubroutineType(types: !22)
!22 = !{!4, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
!24 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1", file: !1, line: 7, size: 64, elements: !25)
!25 = !{!26, !27}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !24, file: !1, line: 7, baseType: !4, size: 32)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !24, file: !1, line: 7, baseType: !28, size: 64)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 6, size: 64, elements: !29)
!29 = !{!30, !32, !34}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !28, file: !1, line: 6, baseType: !31, size: 32)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "__A", file: !1, line: 3, baseType: !3)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !28, file: !1, line: 6, baseType: !33, size: 9, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!33 = !DIDerivedType(tag: DW_TAG_typedef, name: "__B", file: !1, line: 4, baseType: !8)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "a3", scope: !28, file: !1, line: 6, baseType: !35, size: 4, offset: 41, flags: DIFlagBitField, extraData: i64 32)
!35 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int", file: !1, line: 5, baseType: !4)
!36 = !{!37, !38, !39, !40}
!37 = !DILocalVariable(name: "arg", arg: 1, scope: !20, file: !1, line: 9, type: !23)
!38 = !DILocalVariable(name: "r1", scope: !20, file: !1, line: 10, type: !9)
!39 = !DILocalVariable(name: "r2", scope: !20, file: !1, line: 11, type: !9)
!40 = !DILocalVariable(name: "r3", scope: !20, file: !1, line: 12, type: !9)
!41 = !DILocation(line: 0, scope: !20)
!42 = !DILocation(line: 10, column: 52, scope: !20)
!43 = !DILocation(line: 10, column: 55, scope: !20)
!44 = !DILocation(line: 10, column: 17, scope: !20)
!45 = !DILocation(line: 11, column: 55, scope: !20)
!46 = !DILocation(line: 11, column: 17, scope: !20)
!47 = !DILocation(line: 12, column: 55, scope: !20)
!48 = !DILocation(line: 12, column: 17, scope: !20)
!49 = !DILocation(line: 13, column: 13, scope: !20)
!50 = !DILocation(line: 13, column: 18, scope: !20)
!51 = !DILocation(line: 13, column: 3, scope: !20)
