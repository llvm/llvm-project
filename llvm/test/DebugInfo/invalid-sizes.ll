; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: llvm.dbg.declare has larger fragment size than alloca size
; CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill, metadata !23, metadata !DIExpression())
; CHECK: llvm.dbg.declare has larger fragment size than alloca size
; CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill1, metadata !23, metadata !DIExpression())

%"EndianSlice<'_>" = type { { ptr, i64 }, i32, [1 x i32] }

; example::test
; Function Attrs: nonlazybind uwtable
define void @_ZN7example4test17h64a501af0fe536ddE(ptr align 1 %s.0, i64 %s.1) unnamed_addr #0 !dbg !7 {
start:
  %slice.dbg.spill1 = alloca i32, align 4
  %slice.dbg.spill = alloca { ptr, i64 }, align 8
  %s.dbg.spill = alloca { ptr, i64 }, align 8
  %_2 = alloca %"EndianSlice<'_>", align 8
  %0 = getelementptr inbounds { ptr, i64 }, ptr %s.dbg.spill, i32 0, i32 0
  store ptr %s.0, ptr %0, align 8
  %1 = getelementptr inbounds { ptr, i64 }, ptr %s.dbg.spill, i32 0, i32 1
  store i64 %s.1, ptr %1, align 8
  call void @llvm.dbg.declare(metadata ptr %s.dbg.spill, metadata !22, metadata !DIExpression()), !dbg !33
  %2 = getelementptr inbounds { ptr, i64 }, ptr %slice.dbg.spill, i32 0, i32 0, !dbg !34
  store ptr %s.0, ptr %2, align 8, !dbg !34
  %3 = getelementptr inbounds { ptr, i64 }, ptr %slice.dbg.spill, i32 0, i32 1, !dbg !34
  store i64 %s.1, ptr %3, align 8, !dbg !34
  call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill, metadata !23, metadata !DIExpression()), !dbg !35
  store i32 1, ptr %slice.dbg.spill1, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata ptr %slice.dbg.spill1, metadata !23, metadata !DIExpression()), !dbg !35
  ret void, !dbg !36
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.dbg.cu = !{!5}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"rustc version 1.74.0-nightly (5c6a7e71c 2023-08-20)"}
!5 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !6, producer: "clang LLVM (rustc version 1.74.0-nightly (5c6a7e71c 2023-08-20))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!6 = !DIFile(filename: "/app/example.rs/@/example.a6c375ed18e8f6d3-cgu.0", directory: "/app")
!7 = distinct !DISubprogram(name: "test", linkageName: "_ZN7example4test17h64a501af0fe536ddE", scope: !9, file: !8, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, templateParams: !20, retainedNodes: !21)
!8 = !DIFile(filename: "example.rs", directory: "/app", checksumkind: CSK_MD5, checksum: "bd53c9e80c244adbeae5aa0d57de599d")
!9 = !DINamespace(name: "example", scope: null)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12}
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[u8]", file: !13, size: 128, align: 64, elements: !14, templateParams: !20, identifier: "4f7d759e2003ffb713a77bd933fd0146")
!13 = !DIFile(filename: "<unknown>", directory: "")
!14 = !{!15, !18}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !12, file: !13, baseType: !16, size: 64, align: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64, align: 64, dwarfAddressSpace: 0)
!17 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !12, file: !13, baseType: !19, size: 64, align: 64, offset: 64)
!19 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!20 = !{}
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "s", arg: 1, scope: !7, file: !8, line: 9, type: !12)
!23 = !DILocalVariable(name: "slice", scope: !24, file: !8, line: 10, type: !25, align: 8)
!24 = distinct !DILexicalBlock(scope: !7, file: !8, line: 10, column: 5)
!25 = !DICompositeType(tag: DW_TAG_structure_type, name: "EndianSlice", scope: !9, file: !13, size: 192, align: 64, elements: !26, templateParams: !20, identifier: "f1b6e593370159e9df4228aa26ace4b5")
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "slice", scope: !25, file: !13, baseType: !12, size: 128, align: 64)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "endian", scope: !25, file: !13, baseType: !29, size: 32, align: 32, offset: 128)
!29 = !DICompositeType(tag: DW_TAG_structure_type, name: "Endian", scope: !9, file: !13, size: 32, align: 32, elements: !30, templateParams: !20, identifier: "a76092aada82685a5b963f3da7ae1bd9")
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !29, file: !13, baseType: !32, size: 32, align: 32)
!32 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!33 = !DILocation(line: 9, column: 13, scope: !7)
!34 = !DILocation(line: 10, column: 17, scope: !7)
!35 = !DILocation(line: 10, column: 9, scope: !24)
!36 = !DILocation(line: 11, column: 2, scope: !37)
!37 = !DILexicalBlockFile(scope: !7, file: !8, discriminator: 0)
