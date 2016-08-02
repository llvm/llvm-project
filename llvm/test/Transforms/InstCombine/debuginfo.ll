; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) nounwind readnone

declare i8* @foo(i8*, i32, i64, i64) nounwind

define hidden i8* @foobar(i8* %__dest, i32 %__val, i64 %__len) nounwind inlinehint ssp !dbg !1 {
entry:
  %__dest.addr = alloca i8*, align 8
  %__val.addr = alloca i32, align 4
  %__len.addr = alloca i64, align 8
  store i8* %__dest, i8** %__dest.addr, align 8
; CHECK-NOT: call void @llvm.dbg.declare
; CHECK: call void @llvm.dbg.value
  call void @llvm.dbg.declare(metadata i8** %__dest.addr, metadata !0, metadata !DIExpression()), !dbg !16
  store i32 %__val, i32* %__val.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %__val.addr, metadata !7, metadata !DIExpression()), !dbg !18
  store i64 %__len, i64* %__len.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %__len.addr, metadata !9, metadata !DIExpression()), !dbg !20
  %tmp = load i8*, i8** %__dest.addr, align 8, !dbg !21
  %tmp1 = load i32, i32* %__val.addr, align 4, !dbg !21
  %tmp2 = load i64, i64* %__len.addr, align 8, !dbg !21
  %tmp3 = load i8*, i8** %__dest.addr, align 8, !dbg !21
  %0 = call i64 @llvm.objectsize.i64.p0i8(i8* %tmp3, i1 false), !dbg !21
  %call = call i8* @foo(i8* %tmp, i32 %tmp1, i64 %tmp2, i64 %0), !dbg !21
  ret i8* %call, !dbg !21
}

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}

!0 = !DILocalVariable(name: "__dest", line: 78, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "foobar", line: 79, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 79, file: !27, scope: !2, type: !4, variables: !25)
!2 = !DIFile(filename: "string.h", directory: "Game")
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 127710)", isOptimized: true, emissionKind: 0, file: !28, enums: !29, retainedTypes: !29, subprograms: !24)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !3, baseType: null)
!7 = !DILocalVariable(name: "__val", line: 78, arg: 2, scope: !1, file: !2, type: !8)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "__len", line: 78, arg: 3, scope: !1, file: !2, type: !10)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", line: 80, file: !27, scope: !3, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "__darwin_size_t", line: 90, file: !27, scope: !3, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!16 = !DILocation(line: 78, column: 28, scope: !1)
!18 = !DILocation(line: 78, column: 40, scope: !1)
!20 = !DILocation(line: 78, column: 54, scope: !1)
!21 = !DILocation(line: 80, column: 3, scope: !22)
!22 = distinct !DILexicalBlock(line: 80, column: 3, file: !27, scope: !23)
!23 = distinct !DILexicalBlock(line: 79, column: 1, file: !27, scope: !1)
!24 = !{!1}
!25 = !{!0, !7, !9}
!26 = !DIFile(filename: "bits.c", directory: "Game")
!27 = !DIFile(filename: "string.h", directory: "Game")
!28 = !DIFile(filename: "bits.c", directory: "Game")
!29 = !{}
!30 = !{i32 1, !"Debug Info Version", i32 3}
