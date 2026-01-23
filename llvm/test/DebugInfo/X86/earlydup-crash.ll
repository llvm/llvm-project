; RUN: llc %s -mtriple=i386-apple-macosx10.6.7 -o /dev/null

; This used to crash because early dup was not ignoring debug instructions.

%struct.cpp_dir = type { ptr, ptr, i32, i8, ptr, ptr, ptr, i64, i32, i8 }

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

define internal ptr @framework_construct_pathname(ptr %fname, ptr %dir) nounwind ssp !dbg !2 {
entry:
  br i1 poison, label %bb33, label %bb

bb:                                               ; preds = %entry
  %tmp = icmp eq i32 undef, 0
  %tmp1 = add i32 0, 11
  call void @llvm.dbg.value(metadata i32 %tmp1, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  br i1 poison, label %bb18, label %bb31.preheader

bb31.preheader:                                   ; preds = %bb19, %bb
  br label %bb31

bb18:                                             ; preds = %bb
  %tmp3 = icmp eq i32 undef, 0
  br i1 %tmp3, label %bb19, label %bb33

bb19:                                             ; preds = %bb18
  call void @foobar(i32 0)
  br label %bb31.preheader

bb22:                                             ; preds = %bb31
  %tmp4 = add i32 0, %tmp1
  call void @foobar(i32 %tmp4)
  br i1 poison, label %bb33, label %bb31

bb31:                                             ; preds = %bb22, %bb31.preheader
  br i1 false, label %bb33, label %bb22

bb33:                                             ; preds = %bb31, %bb22, %bb18, %entry
  ret ptr undef
}

declare void @foobar(i32)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!47}
!0 = !DILocalVariable(name: "frname_len", line: 517, scope: !1, file: !3, type: !38)
!1 = distinct !DILexicalBlock(line: 515, column: 0, file: !44, scope: !2)
!2 = distinct !DISubprogram(name: "framework_construct_pathname", line: 515, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !4, file: !44, scope: null, type: !5)
!3 = !DIFile(filename: "darwin-c.c", directory: "/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/config")
!4 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !44, enums: !46, retainedTypes: !46)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !9, !11}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_const_type, size: 8, align: 8, file: !44, scope: !3, baseType: !8)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "cpp_dir", line: 45, file: !41, scope: !13, baseType: !14)
!13 = !DIFile(filename: "cpplib.h", directory: "/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/../libcpp/include")
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "cpp_dir", line: 43, size: 352, align: 32, file: !41, scope: !3, elements: !15)
!15 = !{!16, !18, !19, !21, !23, !25, !27, !29, !33, !36}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "next", line: 572, size: 32, align: 32, file: !41, scope: !14, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !14)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "name", line: 575, size: 32, align: 32, offset: 32, file: !41, scope: !14, baseType: !7)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "len", line: 576, size: 32, align: 32, offset: 64, file: !41, scope: !14, baseType: !20)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "sysp", line: 580, size: 8, align: 8, offset: 96, file: !41, scope: !14, baseType: !22)
!22 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "name_map", line: 584, size: 32, align: 32, offset: 128, file: !41, scope: !14, baseType: !24)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !9)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "header_map", line: 590, size: 32, align: 32, offset: 160, file: !41, scope: !14, baseType: !26)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: null)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "construct", line: 597, size: 32, align: 32, offset: 192, file: !41, scope: !14, baseType: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !44, scope: !3, baseType: !5)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "ino", line: 601, size: 64, align: 64, offset: 224, file: !41, scope: !14, baseType: !30)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "ino_t", line: 141, file: !42, scope: !31, baseType: !32)
!31 = !DIFile(filename: "types.h", directory: "/usr/include/sys")
!32 = !DIBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "dev", line: 602, size: 32, align: 32, offset: 288, file: !41, scope: !14, baseType: !34)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "dev_t", line: 107, file: !42, scope: !31, baseType: !35)
!35 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "user_supplied_p", line: 605, size: 8, align: 8, offset: 320, file: !41, scope: !14, baseType: !37)
!37 = !DIBasicType(tag: DW_TAG_base_type, name: "_Bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!38 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", line: 326, file: !43, scope: !39, baseType: !40)
!39 = !DIFile(filename: "stddef.h", directory: "/Users/espindola/llvm/build-llvm-gcc/./prev-gcc/include")
!40 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!41 = !DIFile(filename: "cpplib.h", directory: "/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/../libcpp/include")
!42 = !DIFile(filename: "types.h", directory: "/usr/include/sys")
!43 = !DIFile(filename: "stddef.h", directory: "/Users/espindola/llvm/build-llvm-gcc/./prev-gcc/include")
!44 = !DIFile(filename: "darwin-c.c", directory: "/Users/espindola/llvm/build-llvm-gcc/gcc/../../llvm-gcc-4.2/gcc/config")
!46 = !{}
!47 = !{i32 1, !"Debug Info Version", i32 3}
