; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-EL,CHECK64 %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK,CHECK-EL,CHECK32 %s
; Source code:
;   struct s {
;     int a;
;     int b1:9;
;     int b2:4;
;   };
;   enum {
;     FIELD_BYTE_OFFSET = 0,
;     FIELD_BYTE_SIZE,
;     FIELD_EXISTENCE,
;     FIELD_SIGNEDNESS,
;     FIELD_LSHIFT_U64,
;     FIELD_RSHIFT_U64,
;   };
;   int field_read(struct s *arg) {
;     unsigned long long ull;
;     unsigned offset = __builtin_preserve_field_info(arg->b2, FIELD_BYTE_OFFSET);
;     unsigned size = __builtin_preserve_field_info(arg->b2, FIELD_BYTE_SIZE);
;     switch(size) {
;     case 1:
;       ull = *(unsigned char *)((ptr)arg + offset); break;
;     case 2:
;       ull = *(unsigned short *)((ptr)arg + offset); break;
;     case 4:
;       ull = *(unsigned int *)((ptr)arg + offset); break;
;     case 8:
;       ull = *(unsigned long long *)((ptr)arg + offset); break;
;     }
;     ull <<= __builtin_preserve_field_info(arg->b2, FIELD_LSHIFT_U64);
;     if (__builtin_preserve_field_info(arg->b2, FIELD_SIGNEDNESS))
;       return ((long long)ull) >>__builtin_preserve_field_info(arg->b2, FIELD_RSHIFT_U64);
;     return ull >> __builtin_preserve_field_info(arg->b2, FIELD_RSHIFT_U64);
;   }
; Compilation flag:
;   clang -target bpfel -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpfel"

%struct.s = type { i32, i16 }

; Function Attrs: nounwind readonly
define dso_local i32 @field_read(ptr %arg) local_unnamed_addr #0 !dbg !26 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !37, metadata !DIExpression()), !dbg !41
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr elementtype(%struct.s) %arg, i32 1, i32 2), !dbg !42, !llvm.preserve.access.index !31
  %1 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 0), !dbg !43
  call void @llvm.dbg.value(metadata i32 %1, metadata !39, metadata !DIExpression()), !dbg !41
  %2 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 1), !dbg !44
  call void @llvm.dbg.value(metadata i32 %2, metadata !40, metadata !DIExpression()), !dbg !41
  switch i32 %2, label %sw.epilog [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 4, label %sw.bb5
    i32 8, label %sw.bb9
  ], !dbg !45

sw.bb:                                            ; preds = %entry
  %idx.ext = zext i32 %1 to i64, !dbg !48
  %add.ptr = getelementptr i8, ptr %arg, i64 %idx.ext, !dbg !48
  %3 = load i8, ptr %add.ptr, align 1, !dbg !49, !tbaa !50
  %conv = zext i8 %3 to i64, !dbg !49
  call void @llvm.dbg.value(metadata i64 %conv, metadata !38, metadata !DIExpression()), !dbg !41
  br label %sw.epilog, !dbg !53

sw.bb1:                                           ; preds = %entry
  %idx.ext2 = zext i32 %1 to i64, !dbg !55
  %add.ptr3 = getelementptr i8, ptr %arg, i64 %idx.ext2, !dbg !55
  %4 = load i16, ptr %add.ptr3, align 2, !dbg !57, !tbaa !58
  %conv4 = zext i16 %4 to i64, !dbg !57
  call void @llvm.dbg.value(metadata i64 %conv4, metadata !38, metadata !DIExpression()), !dbg !41
  br label %sw.epilog, !dbg !60

sw.bb5:                                           ; preds = %entry
  %idx.ext6 = zext i32 %1 to i64, !dbg !62
  %add.ptr7 = getelementptr i8, ptr %arg, i64 %idx.ext6, !dbg !62
  %5 = load i32, ptr %add.ptr7, align 4, !dbg !64, !tbaa !65
  %conv8 = zext i32 %5 to i64, !dbg !64
  call void @llvm.dbg.value(metadata i64 %conv8, metadata !38, metadata !DIExpression()), !dbg !41
  br label %sw.epilog, !dbg !67

sw.bb9:                                           ; preds = %entry
  %idx.ext10 = zext i32 %1 to i64, !dbg !69
  %add.ptr11 = getelementptr i8, ptr %arg, i64 %idx.ext10, !dbg !69
  %6 = load i64, ptr %add.ptr11, align 8, !dbg !71, !tbaa !72
  call void @llvm.dbg.value(metadata i64 %6, metadata !38, metadata !DIExpression()), !dbg !41
  br label %sw.epilog, !dbg !74

sw.epilog:                                        ; preds = %entry, %sw.bb9, %sw.bb5, %sw.bb1, %sw.bb
  %ull.0 = phi i64 [ undef, %entry ], [ %6, %sw.bb9 ], [ %conv8, %sw.bb5 ], [ %conv4, %sw.bb1 ], [ %conv, %sw.bb ]
  call void @llvm.dbg.value(metadata i64 %ull.0, metadata !38, metadata !DIExpression()), !dbg !41
  %7 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 4), !dbg !75
  %sh_prom = zext i32 %7 to i64, !dbg !76
  %shl = shl i64 %ull.0, %sh_prom, !dbg !76
  call void @llvm.dbg.value(metadata i64 %shl, metadata !38, metadata !DIExpression()), !dbg !41
  %8 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 3), !dbg !77
  %tobool = icmp eq i32 %8, 0, !dbg !77
  %9 = tail call i32 @llvm.bpf.preserve.field.info.p0(ptr %0, i64 5), !dbg !41
  %sh_prom12 = zext i32 %9 to i64, !dbg !41
  %shr = ashr i64 %shl, %sh_prom12, !dbg !79
  %shr15 = lshr i64 %shl, %sh_prom12, !dbg !79
  %retval.0.in = select i1 %tobool, i64 %shr15, i64 %shr, !dbg !79
  %retval.0 = trunc i64 %retval.0.in to i32, !dbg !41
  ret i32 %retval.0, !dbg !80
}

; CHECK:             r{{[0-9]+}} = 4
; CHECK:             r{{[0-9]+}} = 4
; CHECK-EL:          r{{[0-9]+}} <<= 51
; CHECK64:           r{{[0-9]+}} s>>= 60
; CHECK64:           r{{[0-9]+}} >>= 60
; CHECK32:           r{{[0-9]+}} >>= 60
; CHECK32:           r{{[0-9]+}} s>>= 60
; CHECK:             r{{[0-9]+}} = 1

; CHECK:             .long   1                       # BTF_KIND_STRUCT(id = 2)
; CHECK:             .byte   115                     # string offset=1
; CHECK:             .ascii  ".text"                 # string offset=30
; CHECK:             .ascii  "0:2"                   # string offset=36

; CHECK:             .long   16                      # FieldReloc
; CHECK-NEXT:        .long   30                      # Field reloc section string offset=30
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   36
; CHECK-NEXT:        .long   3

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}

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
!12 = !{!13, !15, !16, !18, !19, !21}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!21 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"wchar_size", i32 4}
!25 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 923aa0ce806f7739b754167239fee2c9a15e2f31)"}
!26 = distinct !DISubprogram(name: "field_read", scope: !1, file: !1, line: 14, type: !27, scopeLine: 14, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !36)
!27 = !DISubroutineType(types: !28)
!28 = !{!29, !30}
!29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 64, elements: !32)
!32 = !{!33, !34, !35}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !31, file: !1, line: 2, baseType: !29, size: 32)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "b1", scope: !31, file: !1, line: 3, baseType: !29, size: 9, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "b2", scope: !31, file: !1, line: 4, baseType: !29, size: 4, offset: 41, flags: DIFlagBitField, extraData: i64 32)
!36 = !{!37, !38, !39, !40}
!37 = !DILocalVariable(name: "arg", arg: 1, scope: !26, file: !1, line: 14, type: !30)
!38 = !DILocalVariable(name: "ull", scope: !26, file: !1, line: 15, type: !20)
!39 = !DILocalVariable(name: "offset", scope: !26, file: !1, line: 16, type: !4)
!40 = !DILocalVariable(name: "size", scope: !26, file: !1, line: 17, type: !4)
!41 = !DILocation(line: 0, scope: !26)
!42 = !DILocation(line: 16, column: 56, scope: !26)
!43 = !DILocation(line: 16, column: 21, scope: !26)
!44 = !DILocation(line: 17, column: 19, scope: !26)
!45 = !DILocation(line: 18, column: 3, scope: !26)
!46 = !DILocation(line: 20, column: 30, scope: !47)
!47 = distinct !DILexicalBlock(scope: !26, file: !1, line: 18, column: 16)
!48 = !DILocation(line: 20, column: 42, scope: !47)
!49 = !DILocation(line: 20, column: 11, scope: !47)
!50 = !{!51, !51, i64 0}
!51 = !{!"omnipotent char", !52, i64 0}
!52 = !{!"Simple C/C++ TBAA"}
!53 = !DILocation(line: 20, column: 53, scope: !47)
!54 = !DILocation(line: 22, column: 31, scope: !47)
!55 = !DILocation(line: 22, column: 43, scope: !47)
!56 = !DILocation(line: 22, column: 12, scope: !47)
!57 = !DILocation(line: 22, column: 11, scope: !47)
!58 = !{!59, !59, i64 0}
!59 = !{!"short", !51, i64 0}
!60 = !DILocation(line: 22, column: 54, scope: !47)
!61 = !DILocation(line: 24, column: 29, scope: !47)
!62 = !DILocation(line: 24, column: 41, scope: !47)
!63 = !DILocation(line: 24, column: 12, scope: !47)
!64 = !DILocation(line: 24, column: 11, scope: !47)
!65 = !{!66, !66, i64 0}
!66 = !{!"int", !51, i64 0}
!67 = !DILocation(line: 24, column: 52, scope: !47)
!68 = !DILocation(line: 26, column: 35, scope: !47)
!69 = !DILocation(line: 26, column: 47, scope: !47)
!70 = !DILocation(line: 26, column: 12, scope: !47)
!71 = !DILocation(line: 26, column: 11, scope: !47)
!72 = !{!73, !73, i64 0}
!73 = !{!"long long", !51, i64 0}
!74 = !DILocation(line: 26, column: 58, scope: !47)
!75 = !DILocation(line: 28, column: 11, scope: !26)
!76 = !DILocation(line: 28, column: 7, scope: !26)
!77 = !DILocation(line: 29, column: 7, scope: !78)
!78 = distinct !DILexicalBlock(scope: !26, file: !1, line: 29, column: 7)
!79 = !DILocation(line: 29, column: 7, scope: !26)
!80 = !DILocation(line: 32, column: 1, scope: !26)
