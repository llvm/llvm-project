; RUN: llc < %s -O0 -verify-machineinstrs -mtriple=wasm32-unknown-unknown | FileCheck %s

; CHECK: BB#0
; CHECK: #DEBUG_VALUE: usage:self <- %vreg4
; CHECK: BB#1
; CHECK: DW_TAG_variable
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%0 = type opaque

@key = external local_unnamed_addr global [15 x i8], align 1
@.str = external unnamed_addr constant [33 x i8], align 1

; Function Attrs: nounwind
define internal i32 @0(i8*) local_unnamed_addr !dbg !14 !type !22 {
  tail call void @llvm.dbg.value(metadata i8* %0, i64 0, metadata !21, metadata !23), !dbg !24
  %2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str, i32 0, i32 0), i8* %0), !dbg !25
  br i1 1, label %a, label %b
a:
  %3 = add i32 %2, %2
  br label %c

b:
  %4 = sub i32 %2, %2
  br label %c

c:
  %5 = phi i32 [ %3, %a ], [ %4, %b ]
  %6 = add i32 ptrtoint (i32 (i8*)* @0 to i32), %5
  ret i32 %6, !dbg !26
}

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "crash.c", directory: "wasm/tests")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "key", scope: !0, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, variable: [15 x i8]* @key)
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 120, align: 8, elements: !9)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !7, line: 185, baseType: !8)
!7 = !DIFile(filename: "wasm/emscripten/system/include/libc/bits/alltypes.h", directory: "wasm/tests")
!8 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!9 = !{!10}
!10 = !DISubrange(count: 15)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)"}
!14 = distinct !DISubprogram(name: "usage", scope: !1, file: !1, line: 15, type: !15, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !20)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 32, align: 32)
!19 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!20 = !{!21}
!21 = !DILocalVariable(name: "self", arg: 1, scope: !14, file: !1, line: 15, type: !18)
!22 = !{i64 0, !"_ZTSFiPcE"}
!23 = !DIExpression()
!24 = !DILocation(line: 15, column: 17, scope: !14)
!25 = !DILocation(line: 16, column: 3, scope: !14)
!26 = !DILocation(line: 17, column: 3, scope: !14)
