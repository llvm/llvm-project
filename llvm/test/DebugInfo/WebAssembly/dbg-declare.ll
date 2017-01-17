; RUN: llc < %s -verify-machineinstrs -mtriple=wasm32-unknown-unknown | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=wasm32-unknown-unknown -fast-isel | FileCheck --check-prefix=CHECK-FAST %s
; CHECK: #DEBUG_VALUE: decode:i <- [%vreg
; CHECK: #DEBUG_VALUE: decode:v <- [%vreg
; CHECK: DW_TAG_variable
; CHECK-FAST: DW_TAG_variable

; Test that llvm.dbg.declare() instrinsics do not crash the backend

source_filename = "test/DebugInfo/WebAssembly/dbg-declare.ll"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@key = external global [15 x i8], align 1

define internal zeroext i8 @0(i32, i8 zeroext) !dbg !15 !type !20 {
  %3 = alloca i32, align 4
  %4 = alloca i8, align 1
  store i32 %0, i32* %3, align 4
  call void @llvm.dbg.declare(metadata i32* %3, metadata !21, metadata !22), !dbg !23
  store i8 %1, i8* %4, align 1
  call void @llvm.dbg.declare(metadata i8* %4, metadata !24, metadata !22), !dbg !25
  %5 = load i8, i8* %4, align 1, !dbg !26
  %6 = zext i8 %5 to i32, !dbg !26
  %7 = load i32, i32* %3, align 4, !dbg !27
  %8 = urem i32 %7, 15, !dbg !28
  %9 = getelementptr inbounds [15 x i8], [15 x i8]* @key, i32 0, i32 %8, !dbg !29
  %10 = load i8, i8* %9, align 1, !dbg !29
  %11 = zext i8 %10 to i32, !dbg !29
  %12 = xor i32 %6, %11, !dbg !30
  %13 = trunc i32 %12 to i8, !dbg !31
  ret i8 %13, !dbg !32
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "crash.c", directory: "wasm/tests")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariableExpression(var: !5)
!5 = !DIGlobalVariable(name: "key", scope: !0, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 120, align: 8, elements: !10)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !8, line: 185, baseType: !9)
!8 = !DIFile(filename: "wasm/emscripten/system/include/libc/bits/alltypes.h", directory: "wasm/tests")
!9 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!10 = !{!11}
!11 = !DISubrange(count: 15)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)"}
!15 = distinct !DISubprogram(name: "decode", scope: !1, file: !1, line: 11, type: !16, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!7, !18, !7}
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !8, line: 124, baseType: !19)
!19 = !DIBasicType(name: "long unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!20 = !{i64 0, !"_ZTSFhmhE"}
!21 = !DILocalVariable(name: "i", arg: 1, scope: !15, file: !1, line: 11, type: !18)
!22 = !DIExpression()
!23 = !DILocation(line: 11, column: 23, scope: !15)
!24 = !DILocalVariable(name: "v", arg: 2, scope: !15, file: !1, line: 11, type: !7)
!25 = !DILocation(line: 11, column: 34, scope: !15)
!26 = !DILocation(line: 12, column: 11, scope: !15)
!27 = !DILocation(line: 12, column: 19, scope: !15)
!28 = !DILocation(line: 12, column: 21, scope: !15)
!29 = !DILocation(line: 12, column: 15, scope: !15)
!30 = !DILocation(line: 12, column: 13, scope: !15)
!31 = !DILocation(line: 12, column: 10, scope: !15)
!32 = !DILocation(line: 12, column: 3, scope: !15)

