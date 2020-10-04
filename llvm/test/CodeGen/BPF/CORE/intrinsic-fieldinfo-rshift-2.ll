; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-ALU32 %s
; Source code:
;   typedef struct s1 { int a1; char a2; } __s1;
;   union u1 { int b1; __s1 b2; };
;   enum { FIELD_RSHIFT_U64 = 5, };
;   int test(union u1 *arg) {
;     unsigned r1 = __builtin_preserve_field_info(arg->b2.a1, FIELD_RSHIFT_U64);
;     unsigned r2 = __builtin_preserve_field_info(arg->b2.a2, FIELD_RSHIFT_U64);
;     /* r1: 32, r2: 56 */
;     return r1 + r2;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%union.u1 = type { %struct.s1 }
%struct.s1 = type { i32, i8 }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%union.u1* %arg) local_unnamed_addr #0 !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata %union.u1* %arg, metadata !27, metadata !DIExpression()), !dbg !30
  %0 = tail call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %arg, i32 1), !dbg !31, !llvm.preserve.access.index !16
  %b2 = getelementptr inbounds %union.u1, %union.u1* %0, i64 0, i32 0, !dbg !31
  %1 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* %b2, i32 0, i32 0), !dbg !32, !llvm.preserve.access.index !21
  %2 = tail call i32 @llvm.bpf.preserve.field.info.p0i32(i32* %1, i64 5), !dbg !33
  call void @llvm.dbg.value(metadata i32 %2, metadata !28, metadata !DIExpression()), !dbg !30
  %3 = tail call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1* %b2, i32 1, i32 1), !dbg !34, !llvm.preserve.access.index !21
  %4 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %3, i64 5), !dbg !35
  call void @llvm.dbg.value(metadata i32 %4, metadata !29, metadata !DIExpression()), !dbg !30
  %add = add i32 %4, %2, !dbg !36
  ret i32 %add, !dbg !37
}

; CHECK:             r1 = 32
; CHECK:             r0 = 56
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_UNION(id = 2)
; CHECK:             .ascii  "u1"                    # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=42
; CHECK:             .ascii  "0:1:0"                 # string offset=48
; CHECK:             .ascii  "0:1:1"                 # string offset=91

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   42                      # Field reloc section string offset=42
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   91
; CHECK-NEXT:        .long   5

; Function Attrs: nounwind readnone
declare %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1*, i32) #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i32(i32*, i64) #1

; Function Attrs: nounwind readnone
declare i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1*, i32, i32) #1

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
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 3, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "FIELD_RSHIFT_U64", value: 5, isUnsigned: true)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 4a60741b74384f14b21fdc0131ede326438840ab)"}
!11 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !12, scopeLine: 4, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !26)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !15}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1", file: !1, line: 2, size: 64, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !16, file: !1, line: 2, baseType: !14, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !16, file: !1, line: 2, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s1", file: !1, line: 1, baseType: !21)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1", file: !1, line: 1, size: 64, elements: !22)
!22 = !{!23, !24}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !21, file: !1, line: 1, baseType: !14, size: 32)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a2", scope: !21, file: !1, line: 1, baseType: !25, size: 8, offset: 32)
!25 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!26 = !{!27, !28, !29}
!27 = !DILocalVariable(name: "arg", arg: 1, scope: !11, file: !1, line: 4, type: !15)
!28 = !DILocalVariable(name: "r1", scope: !11, file: !1, line: 5, type: !4)
!29 = !DILocalVariable(name: "r2", scope: !11, file: !1, line: 6, type: !4)
!30 = !DILocation(line: 0, scope: !11)
!31 = !DILocation(line: 5, column: 52, scope: !11)
!32 = !DILocation(line: 5, column: 55, scope: !11)
!33 = !DILocation(line: 5, column: 17, scope: !11)
!34 = !DILocation(line: 6, column: 55, scope: !11)
!35 = !DILocation(line: 6, column: 17, scope: !11)
!36 = !DILocation(line: 8, column: 13, scope: !11)
!37 = !DILocation(line: 8, column: 3, scope: !11)
