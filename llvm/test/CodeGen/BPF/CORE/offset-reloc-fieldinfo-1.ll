; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK32 %s
; Source code:
;   struct s {
;     int a;
;     int b1:9;
;     int b2:4;
;   };
;   enum {
;       FIELD_BYTE_OFFSET = 0,
;       FIELD_BYTE_SIZE,
;       FIELD_EXISTENCE,
;       FIELD_SIGNEDNESS,
;       FIELD_LSHIFT_U64,
;       FIELD_RSHIFT_U64,
;   };
;   void bpf_probe_read(ptr, unsigned, const ptr);
;   int field_read(struct s *arg) {
;     unsigned long long ull;
;     unsigned offset = __builtin_preserve_field_info(arg->b2, FIELD_BYTE_OFFSET);
;     unsigned size = __builtin_preserve_field_info(arg->b2, FIELD_BYTE_SIZE);
;     unsigned lshift;
;
;     bpf_probe_read(&ull, size, (const ptr)arg + offset);
;     lshift = __builtin_preserve_field_info(arg->b2, FIELD_LSHIFT_U64);
;   #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
;     lshift = lshift + (size << 3) - 64;
;   #endif
;     ull <<= lshift;
;     if (__builtin_preserve_field_info(arg->b2, FIELD_SIGNEDNESS))
;       return (long long)ull >> __builtin_preserve_field_info(arg->b2, FIELD_RSHIFT_U64);
;     return ull >> __builtin_preserve_field_info(arg->b2, FIELD_RSHIFT_U64);
;   }
; Compilation flag:
;   clang -target bpfel -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpfel"

%struct.s = type { i32, i16 }

; Function Attrs: nounwind
define dso_local i32 @field_read(ptr %arg) local_unnamed_addr #0 !dbg !20 {
entry:
  %ull = alloca i64, align 8
  call void @llvm.dbg.value(metadata ptr %arg, metadata !31, metadata !DIExpression()), !dbg !37
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ull) #5, !dbg !38
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr elementtype(%struct.s) %arg, i32 1, i32 2), !dbg !39, !llvm.preserve.access.index !25
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 0), !dbg !40
  call void @llvm.dbg.value(metadata i32 %1, metadata !34, metadata !DIExpression()), !dbg !37
  %2 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 1), !dbg !41
  call void @llvm.dbg.value(metadata i32 %2, metadata !35, metadata !DIExpression()), !dbg !37
  %idx.ext = zext i32 %1 to i64, !dbg !43
  %add.ptr = getelementptr i8, ptr %arg, i64 %idx.ext, !dbg !43
  call void @bpf_probe_read(ptr nonnull %ull, i32 %2, ptr %add.ptr) #5, !dbg !44
  %3 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 4), !dbg !45
  call void @llvm.dbg.value(metadata i32 %3, metadata !36, metadata !DIExpression()), !dbg !37
  %4 = load i64, ptr %ull, align 8, !dbg !46, !tbaa !47
  call void @llvm.dbg.value(metadata i64 %4, metadata !32, metadata !DIExpression()), !dbg !37
  %sh_prom = zext i32 %3 to i64, !dbg !46
  %shl = shl i64 %4, %sh_prom, !dbg !46
  call void @llvm.dbg.value(metadata i64 %shl, metadata !32, metadata !DIExpression()), !dbg !37
  %5 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 3), !dbg !51
  %tobool = icmp eq i32 %5, 0, !dbg !51
  %6 = call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 5), !dbg !37
  %sh_prom1 = zext i32 %6 to i64, !dbg !37
  %shr = ashr i64 %shl, %sh_prom1, !dbg !53
  %shr3 = lshr i64 %shl, %sh_prom1, !dbg !53
  %retval.0.in = select i1 %tobool, i64 %shr3, i64 %shr, !dbg !53
  %retval.0 = trunc i64 %retval.0.in to i32, !dbg !37
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ull) #5, !dbg !54
  ret i32 %retval.0, !dbg !54
}

; CHECK:             r{{[0-9]+}} = 4
; CHECK:             r{{[0-9]+}} = 4
; CHECK:             r{{[0-9]+}} <<= 51
; CHECK64:           r{{[0-9]+}} s>>= 60
; CHECK64:           r{{[0-9]+}} >>= 60
; CHECK32:           r{{[0-9]+}} >>= 60
; CHECK32:           r{{[0-9]+}} s>>= 60
; CHECK:             r{{[0-9]+}} = 1

; CHECK:             .byte   115                     # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=30
; CHECK:             .ascii  "0:2"                   # string offset=73

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   30                      # Field reloc section string offset=30
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   73
; CHECK-NEXT:        .long   3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr, i32, i32) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64) #2

declare dso_local void @bpf_probe_read(ptr, i32, ptr) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone speculatable willreturn }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 923aa0ce806f7739b754167239fee2c9a15e2f31)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !12, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !1, line: 6, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6, !7, !8, !9, !10, !11}
!6 = !DIEnumerator(name: "FIELD_BYTE_OFFSET", value: 0, isUnsigned: true)
!7 = !DIEnumerator(name: "FIELD_BYTE_SIZE", value: 1, isUnsigned: true)
!8 = !DIEnumerator(name: "FIELD_EXISTENCE", value: 2, isUnsigned: true)
!9 = !DIEnumerator(name: "FIELD_SIGNEDNESS", value: 3, isUnsigned: true)
!10 = !DIEnumerator(name: "FIELD_LSHIFT_U64", value: 4, isUnsigned: true)
!11 = !DIEnumerator(name: "FIELD_RSHIFT_U64", value: 5, isUnsigned: true)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!15 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 923aa0ce806f7739b754167239fee2c9a15e2f31)"}
!20 = distinct !DISubprogram(name: "field_read", scope: !1, file: !1, line: 15, type: !21, scopeLine: 15, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !30)
!21 = !DISubroutineType(types: !22)
!22 = !{!23, !24}
!23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 64, elements: !26)
!26 = !{!27, !28, !29}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !25, file: !1, line: 2, baseType: !23, size: 32)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !25, file: !1, line: 3, baseType: !23, size: 9, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !25, file: !1, line: 4, baseType: !23, size: 4, offset: 41, flags: DIFlagBitField, extraData: i64 32)
!30 = !{!31, !32, !34, !35, !36}
!31 = !DILocalVariable(name: "arg", arg: 1, scope: !20, file: !1, line: 15, type: !24)
!32 = !DILocalVariable(name: "ull", scope: !20, file: !1, line: 16, type: !33)
!33 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!34 = !DILocalVariable(name: "offset", scope: !20, file: !1, line: 17, type: !4)
!35 = !DILocalVariable(name: "size", scope: !20, file: !1, line: 18, type: !4)
!36 = !DILocalVariable(name: "lshift", scope: !20, file: !1, line: 19, type: !4)
!37 = !DILocation(line: 0, scope: !20)
!38 = !DILocation(line: 16, column: 3, scope: !20)
!39 = !DILocation(line: 17, column: 56, scope: !20)
!40 = !DILocation(line: 17, column: 21, scope: !20)
!41 = !DILocation(line: 18, column: 19, scope: !20)
!42 = !DILocation(line: 21, column: 30, scope: !20)
!43 = !DILocation(line: 21, column: 48, scope: !20)
!44 = !DILocation(line: 21, column: 3, scope: !20)
!45 = !DILocation(line: 22, column: 12, scope: !20)
!46 = !DILocation(line: 26, column: 7, scope: !20)
!47 = !{!48, !48, i64 0}
!48 = !{!"long long", !49, i64 0}
!49 = !{!"omnipotent char", !50, i64 0}
!50 = !{!"Simple C/C++ TBAA"}
!51 = !DILocation(line: 27, column: 7, scope: !52)
!52 = distinct !DILexicalBlock(scope: !20, file: !1, line: 27, column: 7)
!53 = !DILocation(line: 27, column: 7, scope: !20)
!54 = !DILocation(line: 30, column: 1, scope: !20)
