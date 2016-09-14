; RUN: llc < %s -verify-machineinstrs -mtriple=wasm32-unknown-unknown | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=wasm32-unknown-unknown -fast-isel | FileCheck --check-prefix=CHECK-FAST %s
; CHECK: #DEBUG_VALUE: decode:i <- [%vreg
; CHECK: #DEBUG_VALUE: decode:v <- [%vreg
; CHECK: DW_TAG_variable
; CHECK-FAST: DW_TAG_variable

; Test that llvm.dbg.declare() instrinsics do not crash the backend

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%0 = type opaque

@key = external global [15 x i8], align 1

; Function Attrs: nounwind
define internal zeroext i8 @0(i32, i8 zeroext)  !dbg !14 !type !19 {
  %3 = alloca i32, align 4
  %4 = alloca i8, align 1
  store i32 %0, i32* %3, align 4
  call void @llvm.dbg.declare(metadata i32* %3, metadata !20, metadata !21), !dbg !22
  store i8 %1, i8* %4, align 1
  call void @llvm.dbg.declare(metadata i8* %4, metadata !23, metadata !21), !dbg !24
  %5 = load i8, i8* %4, align 1, !dbg !25
  %6 = zext i8 %5 to i32, !dbg !25
  %7 = load i32, i32* %3, align 4, !dbg !26
  %8 = urem i32 %7, 15, !dbg !27
  %9 = getelementptr inbounds [15 x i8], [15 x i8]* @key, i32 0, i32 %8, !dbg !28
  %10 = load i8, i8* %9, align 1, !dbg !28
  %11 = zext i8 %10 to i32, !dbg !28
  %12 = xor i32 %6, %11, !dbg !29
  %13 = trunc i32 %12 to i8, !dbg !30
  ret i8 %13, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "crash.c", directory: "wasm/tests")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "key", scope: !0, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true)
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 120, align: 8, elements: !9)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !7, line: 185, baseType: !8)
!7 = !DIFile(filename: "wasm/emscripten/system/include/libc/bits/alltypes.h", directory: "wasm/tests")
!8 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!9 = !{!10}
!10 = !DISubrange(count: 15)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)"}
!14 = distinct !DISubprogram(name: "decode", scope: !1, file: !1, line: 11, type: !15, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!6, !17, !6}
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !7, line: 124, baseType: !18)
!18 = !DIBasicType(name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!19 = !{i64 0, !"_ZTSFhmhE"}
!20 = !DILocalVariable(name: "i", arg: 1, scope: !14, file: !1, line: 11, type: !17)
!21 = !DIExpression()
!22 = !DILocation(line: 11, column: 23, scope: !14)
!23 = !DILocalVariable(name: "v", arg: 2, scope: !14, file: !1, line: 11, type: !6)
!24 = !DILocation(line: 11, column: 34, scope: !14)
!25 = !DILocation(line: 12, column: 11, scope: !14)
!26 = !DILocation(line: 12, column: 19, scope: !14)
!27 = !DILocation(line: 12, column: 21, scope: !14)
!28 = !DILocation(line: 12, column: 15, scope: !14)
!29 = !DILocation(line: 12, column: 13, scope: !14)
!30 = !DILocation(line: 12, column: 10, scope: !14)
!31 = !DILocation(line: 12, column: 3, scope: !14)
