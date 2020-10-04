; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-EL,CHECK-ALU64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-EL,CHECK-ALU32 %s
; Source code:
;   struct s {
;     unsigned long long f1;
;     unsigned f2;
;     unsigned f3;
;     unsigned f4;
;     unsigned char f5;
;     unsigned bf1:5,
;              bf2:1;
;   };
;   enum {FIELD_TYPE_OFFSET = 0, FIELD_TYPE_SIZE = 1, FIELD_TYPE_LSHIFT_U64 = 4,};
;   int test(struct s *arg) {
;     return __builtin_preserve_field_info(arg->bf2, FIELD_TYPE_OFFSET) +
;            __builtin_preserve_field_info(arg->bf2, FIELD_TYPE_SIZE) +
;            __builtin_preserve_field_info(arg->bf2, FIELD_TYPE_LSHIFT_U64);
;   }
; Compilation flag:
;   clang -target bpfel -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpfel"

%struct.s = type { i64, i32, i32, i32, i8, i8 }

; Function Attrs: nounwind readnone
define dso_local i32 @test(%struct.s* %arg) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata %struct.s* %arg, metadata !30, metadata !DIExpression()), !dbg !31
  %0 = tail call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.ss(%struct.s* %arg, i32 5, i32 6), !dbg !32, !llvm.preserve.access.index !18
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %0, i64 0), !dbg !33
  %2 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %0, i64 1), !dbg !34
  %add = add i32 %2, %1, !dbg !35
  %3 = tail call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %0, i64 4), !dbg !36
  %add1 = add i32 %add, %3, !dbg !37
  ret i32 %add1, !dbg !38
}

; CHECK:             r1 = 20
; CHECK:             r0 = 4
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK-EL:          r1 = 50
; CHECK-ALU64:       r0 += r1
; CHECK-ALU32:       w0 += w1
; CHECK:             exit

; CHECK:             .long   1                       # BTF_KIND_STRUCT(id = 2)

; CHECK:             .byte   115                     # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=89
; CHECK:             .ascii  "0:6"                   # string offset=95

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   89                      # Field reloc section string offset=89
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   95
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   95
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   95
; CHECK-NEXT:        .long   4

; Function Attrs: nounwind readnone
declare i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.ss(%struct.s*, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0i8(i8*, i64) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git bc6913e314806882e2b537b5b03996800078d2ad)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 10, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6, !7, !8}
!6 = !DIEnumerator(name: "FIELD_TYPE_OFFSET", value: 0, isUnsigned: true)
!7 = !DIEnumerator(name: "FIELD_TYPE_SIZE", value: 1, isUnsigned: true)
!8 = !DIEnumerator(name: "FIELD_TYPE_LSHIFT_U64", value: 4, isUnsigned: true)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git bc6913e314806882e2b537b5b03996800078d2ad)"}
!13 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 11, type: !14, scopeLine: 11, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !29)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !17}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 192, elements: !19)
!19 = !{!20, !22, !23, !24, !25, !27, !28}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !18, file: !1, line: 2, baseType: !21, size: 64)
!21 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !18, file: !1, line: 3, baseType: !4, size: 32, offset: 64)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !18, file: !1, line: 4, baseType: !4, size: 32, offset: 96)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "f4", scope: !18, file: !1, line: 5, baseType: !4, size: 32, offset: 128)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "f5", scope: !18, file: !1, line: 6, baseType: !26, size: 8, offset: 160)
!26 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "bf1", scope: !18, file: !1, line: 7, baseType: !4, size: 5, offset: 168, flags: DIFlagBitField, extraData: i64 168)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "bf2", scope: !18, file: !1, line: 8, baseType: !4, size: 1, offset: 173, flags: DIFlagBitField, extraData: i64 168)
!29 = !{!30}
!30 = !DILocalVariable(name: "arg", arg: 1, scope: !13, file: !1, line: 11, type: !17)
!31 = !DILocation(line: 0, scope: !13)
!32 = !DILocation(line: 12, column: 45, scope: !13)
!33 = !DILocation(line: 12, column: 10, scope: !13)
!34 = !DILocation(line: 13, column: 10, scope: !13)
!35 = !DILocation(line: 12, column: 69, scope: !13)
!36 = !DILocation(line: 14, column: 10, scope: !13)
!37 = !DILocation(line: 13, column: 67, scope: !13)
!38 = !DILocation(line: 12, column: 3, scope: !13)
