; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -march=bpfel -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; RUN: llc -march=bpfeb -mattr=+alu32 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   typedef unsigned __uint;
;   struct s1 { int a1; __uint a2:9; __uint a3:4; };
;   union u1 { int b1; __uint b2:9; __uint b3:4; };
;   enum { FIELD_SIGNEDNESS = 3, };
;   int test(struct s1 *arg1, union u1 *arg2) {
;     unsigned r1 = __builtin_preserve_field_info(arg1->a1, FIELD_SIGNEDNESS);
;     unsigned r2 = __builtin_preserve_field_info(arg1->a3, FIELD_SIGNEDNESS);
;     unsigned r3 = __builtin_preserve_field_info(arg2->b1, FIELD_SIGNEDNESS);
;     unsigned r4 = __builtin_preserve_field_info(arg2->b3, FIELD_SIGNEDNESS);
;     return r1 + r2 + r3 + r4;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.s1 = type { i32, i16 }
%union.u1 = type { i32 }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%struct.s1* %arg1, %union.u1* %arg2) local_unnamed_addr #0 !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata %struct.s1* %arg1, metadata !29, metadata !DIExpression()), !dbg !35
  call void @llvm.dbg.value(metadata %union.u1* %arg2, metadata !30, metadata !DIExpression()), !dbg !35
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* %arg1, i32 0, i32 0), !dbg !36, !llvm.preserve.access.index !16
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %0, i64 3), !dbg !37
  call void @llvm.dbg.value(metadata i32 %1, metadata !31, metadata !DIExpression()), !dbg !35
  %2 = tail call i16* @llvm.preserve.struct.access.index.p0i16.p0s_struct.s1s(%struct.s1* %arg1, i32 1, i32 2), !dbg !38, !llvm.preserve.access.index !16
  %3 = tail call i32 @llvm.bpf.preserve.field.info.p0i16(i16* %2, i64 3), !dbg !39
  call void @llvm.dbg.value(metadata i32 %3, metadata !32, metadata !DIExpression()), !dbg !35
  %4 = tail call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %arg2, i32 0), !dbg !40, !llvm.preserve.access.index !23
  %b1 = getelementptr inbounds %union.u1, %union.u1* %4, i64 0, i32 0, !dbg !40
  %5 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %b1, i64 3), !dbg !41
  call void @llvm.dbg.value(metadata i32 %5, metadata !33, metadata !DIExpression()), !dbg !35
  %6 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_union.u1s(%union.u1* %arg2, i32 0, i32 2), !dbg !42, !llvm.preserve.access.index !23
  %7 = bitcast i32* %6 to i8*, !dbg !42
  %8 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %7, i64 3), !dbg !43
  call void @llvm.dbg.value(metadata i32 %8, metadata !34, metadata !DIExpression()), !dbg !35
  %add = add i32 %3, %1, !dbg !44
  %add1 = add i32 %add, %5, !dbg !45
  %add2 = add i32 %add1, %8, !dbg !46
  ret i32 %add2, !dbg !47
}

; CHECK:             r1 = 1
; CHECK:             r0 = 0
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             r1 = 1
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             r1 = 0
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_STRUCT(id = 2)
; CHECK:             .long   37                      # BTF_KIND_UNION(id = 7)
; CHECK:             .ascii  "s1"                    # string offset=1
; CHECK:             .ascii  "u1"                    # string offset=37
; CHECK:             .ascii  ".text"                 # string offset=64
; CHECK:             .ascii  "0:0"                   # string offset=70
; CHECK:             .ascii  "0:2"                   # string offset=111

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   64                      # Field reloc section string offset=64
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   70
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   111
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   70
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   111
; CHECK-NEXT:        .long   3

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

; Function Attrs: nounwind readnone
declare i16* @llvm.preserve.struct.access.index.p0i16.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i16(i16*, i64) #1

; Function Attrs: nounwind readnone
declare %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1*, i32) #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_union.u1s(%union.u1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i8(i8*, i64) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4a60741b74384f14b21fdc0131ede326438840ab)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 4, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FIELD_SIGNEDNESS", value: 3, isUnsigned: true)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4a60741b74384f14b21fdc0131ede326438840ab)"}
!11 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !28)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15, !22}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 2, size: 64, elements: !17)
!17 = !{!18, !19, !21}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !16, file: !1, line: 2, baseType: !14, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !16, file: !1, line: 2, baseType: !20, size: 9, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint", file: !1, line: 1, baseType: !4)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "a3", scope: !16, file: !1, line: 2, baseType: !20, size: 4, offset: 41, flags: DIFlagBitField, extraData: i64 32)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1", file: !1, line: 3, size: 32, elements: !24)
!24 = !{!25, !26, !27}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !23, file: !1, line: 3, baseType: !14, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !23, file: !1, line: 3, baseType: !20, size: 9, flags: DIFlagBitField, extraData: i64 0)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "b3", scope: !23, file: !1, line: 3, baseType: !20, size: 4, flags: DIFlagBitField, extraData: i64 0)
!28 = !{!29, !30, !31, !32, !33, !34}
!29 = !DILocalVariable(name: "arg1", arg: 1, scope: !11, file: !1, line: 5, type: !15)
!30 = !DILocalVariable(name: "arg2", arg: 2, scope: !11, file: !1, line: 5, type: !22)
!31 = !DILocalVariable(name: "r1", scope: !11, file: !1, line: 6, type: !4)
!32 = !DILocalVariable(name: "r2", scope: !11, file: !1, line: 7, type: !4)
!33 = !DILocalVariable(name: "r3", scope: !11, file: !1, line: 8, type: !4)
!34 = !DILocalVariable(name: "r4", scope: !11, file: !1, line: 9, type: !4)
!35 = !DILocation(line: 0, scope: !11)
!36 = !DILocation(line: 6, column: 53, scope: !11)
!37 = !DILocation(line: 6, column: 17, scope: !11)
!38 = !DILocation(line: 7, column: 53, scope: !11)
!39 = !DILocation(line: 7, column: 17, scope: !11)
!40 = !DILocation(line: 8, column: 53, scope: !11)
!41 = !DILocation(line: 8, column: 17, scope: !11)
!42 = !DILocation(line: 9, column: 53, scope: !11)
!43 = !DILocation(line: 9, column: 17, scope: !11)
!44 = !DILocation(line: 10, column: 13, scope: !11)
!45 = !DILocation(line: 10, column: 18, scope: !11)
!46 = !DILocation(line: 10, column: 23, scope: !11)
!47 = !DILocation(line: 10, column: 3, scope: !11)
