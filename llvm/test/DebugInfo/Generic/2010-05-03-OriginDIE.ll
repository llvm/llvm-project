
;RUN: llc < %s -o /dev/null
;Radar 7937109

%struct.anon = type { i64, i32, i32, i32, [1 x i32] }
%struct.gpm_t = type { i32, i8*, [16 x i8], i32, i64, i64, i64, i64, i64, i64, i32, i16, i16, [8 x %struct.gpmr_t] }
%struct.gpmr_t = type { [48 x i8], [48 x i8], [16 x i8], i64, i64, i64, i64, i16 }
%struct.gpt_t = type { [8 x i8], i32, i32, i32, i32, i64, i64, i64, i64, [16 x i8], %struct.anon }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.gpm_t*, %struct.gpt_t*)* @gpt2gpm to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define fastcc void @gpt2gpm(%struct.gpm_t* %gpm, %struct.gpt_t* %gpt) nounwind optsize ssp {
entry:
  %data_addr.i18 = alloca i64, align 8            ; <i64*> [#uses=1]
  %data_addr.i17 = alloca i64, align 8            ; <i64*> [#uses=2]
  %data_addr.i16 = alloca i64, align 8            ; <i64*> [#uses=0]
  %data_addr.i15 = alloca i32, align 4            ; <i32*> [#uses=0]
  %data_addr.i = alloca i64, align 8              ; <i64*> [#uses=0]
  %0 = getelementptr inbounds %struct.gpm_t, %struct.gpm_t* %gpm, i32 0, i32 2, i32 0 ; <i8*> [#uses=1]
  %1 = getelementptr inbounds %struct.gpt_t, %struct.gpt_t* %gpt, i32 0, i32 9, i32 0 ; <i8*> [#uses=1]
  call void @uuid_LtoB(i8* %0, i8* %1) nounwind, !dbg !0
  %a9 = load volatile i64, i64* %data_addr.i18, align 8 ; <i64> [#uses=1]
  %a10 = call i64 @llvm.bswap.i64(i64 %a9) nounwind ; <i64> [#uses=1]
  %a11 = getelementptr inbounds %struct.gpt_t, %struct.gpt_t* %gpt, i32 0, i32 8, !dbg !7 ; <i64*> [#uses=1]
  %a12 = load i64, i64* %a11, align 4, !dbg !7         ; <i64> [#uses=1]
  call void @llvm.dbg.declare(metadata i64* %data_addr.i17, metadata !8, metadata !DIExpression()) nounwind, !dbg !14
  store i64 %a12, i64* %data_addr.i17, align 8
  call void @llvm.dbg.value(metadata !6, metadata !15, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !16)
  call void @llvm.dbg.value(metadata i32 0, metadata !19, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !16)
  call void @llvm.dbg.declare(metadata !6, metadata !23, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !24)
  call void @llvm.dbg.value(metadata i64* %data_addr.i17, metadata !34, metadata !DIExpression(DW_OP_deref)) nounwind, !dbg !DILocation(scope: !24)
  %a13 = load volatile i64, i64* %data_addr.i17, align 8 ; <i64> [#uses=1]
  %a14 = call i64 @llvm.bswap.i64(i64 %a13) nounwind ; <i64> [#uses=2]
  %a15 = add i64 %a10, %a14, !dbg !7              ; <i64> [#uses=1]
  %a16 = sub i64 %a15, %a14                       ; <i64> [#uses=1]
  %a17 = getelementptr inbounds %struct.gpm_t, %struct.gpm_t* %gpm, i32 0, i32 5, !dbg !7 ; <i64*> [#uses=1]
  store i64 %a16, i64* %a17, align 4, !dbg !7
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

declare void @uuid_LtoB(i8*, i8*)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!41}
!0 = !DILocation(line: 808, scope: !1)
!1 = distinct !DILexicalBlock(line: 807, column: 0, file: !39, scope: !2)
!2 = distinct !DISubprogram(name: "gpt2gpm", linkageName: "gpt2gpm", line: 807, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, file: !39, scope: null, type: !5)
!3 = !DIFile(filename: "G.c", directory: "/tmp")
!4 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "llvm-gcc", isOptimized: true, emissionKind: FullDebug, file: !39, enums: !18, retainedTypes: !18)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 810, scope: !1)
!8 = !DILocalVariable(name: "data", line: 201, arg: 1, scope: !9, file: !10, type: !11)
!9 = distinct !DISubprogram(name: "_OSSwapInt64", linkageName: "_OSSwapInt64", line: 202, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, file: !10, scope: null, type: !5)
!10 = !DIFile(filename: "OSByteOrder.h", directory: "/usr/include/libkern/ppc")
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", line: 59, file: !36, scope: !3, baseType: !13)
!12 = !DIFile(filename: "stdint.h", directory: "/usr/4.2.1/include")
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!14 = !DILocation(line: 202, scope: !9, inlinedAt: !7)
!15 = !DILocalVariable(name: "base", line: 92, arg: 2, scope: !16, file: !10, type: !17)
!16 = distinct !DISubprogram(name: "OSReadSwapInt64", linkageName: "OSReadSwapInt64", line: 95, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, file: !38, scope: null, type: !5)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !39, scope: !3, baseType: null)
!18 = !{}
!19 = !DILocalVariable(name: "byteOffset", line: 94, arg: 3, scope: !16, file: !10, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "uintptr_t", line: 114, file: !37, scope: !3, baseType: !22)
!21 = !DIFile(filename: "types.h", directory: "/usr/include/ppc")
!22 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!23 = !DILocalVariable(name: "u", line: 100, scope: !24, file: !10, type: !25)
!24 = distinct !DILexicalBlock(line: 95, column: 0, file: !38, scope: !16)
!25 = !DICompositeType(tag: DW_TAG_union_type, line: 97, size: 64, align: 64, file: !38, scope: !16, elements: !26)
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "u64", line: 98, size: 64, align: 64, file: !38, scope: !25, baseType: !11)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "u32", line: 99, size: 64, align: 32, file: !38, scope: !25, baseType: !29)
!29 = !DICompositeType(tag: DW_TAG_array_type, size: 64, align: 32, file: !39, scope: !3, baseType: !30, elements: !32)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", line: 55, file: !36, scope: !3, baseType: !31)
!31 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!32 = !{!33}
!33 = !DISubrange(count: 2)
!34 = !DILocalVariable(name: "addr", line: 96, scope: !24, file: !10, type: !35)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !39, scope: !3, baseType: !11)
!36 = !DIFile(filename: "stdint.h", directory: "/usr/4.2.1/include")
!37 = !DIFile(filename: "types.h", directory: "/usr/include/ppc")
!38 = !DIFile(filename: "OSByteOrder.h", directory: "/usr/include/libkern/ppc")
!39 = !DIFile(filename: "G.c", directory: "/tmp")
!41 = !{i32 1, !"Debug Info Version", i32 3}
