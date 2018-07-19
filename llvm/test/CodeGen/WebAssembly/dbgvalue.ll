; RUN: llc < %s -O0 -verify-machineinstrs -mtriple=wasm32-unknown-unknown-wasm | FileCheck %s

; CHECK: %bb.0
; CHECK: #DEBUG_VALUE: usage:self <- %4
; CHECK: %bb.1
; CHECK: DW_TAG_variable
source_filename = "test/CodeGen/WebAssembly/dbgvalue.ll"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

@key = external local_unnamed_addr global [15 x i8], align 1
@.str = external unnamed_addr constant [33 x i8], align 1

define internal i32 @0(i8*) local_unnamed_addr !dbg !15 !type !23 {
  tail call void @llvm.dbg.value(metadata i8* %0, i64 0, metadata !22, metadata !24), !dbg !25
  %2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str, i32 0, i32 0), i8* %0), !dbg !26
  br i1 true, label %a, label %b

a:                                                ; preds = %1
  %3 = add i32 %2, %2
  br label %c

b:                                                ; preds = %1
  %4 = sub i32 %2, %2
  br label %c

c:                                                ; preds = %b, %a
  %5 = phi i32 [ %3, %a ], [ %4, %b ]
  %6 = add i32 ptrtoint (i32 (i8*)* @0 to i32), %5
  ret i32 %6, !dbg !27
}

declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273884) (llvm/trunk 273897)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "crash.c", directory: "wasm/tests")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
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
!15 = distinct !DISubprogram(name: "usage", scope: !1, file: !1, line: 15, type: !16, isLocal: false, isDefinition: true, scopeLine: 15, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !21)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !19}
!18 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 32, align: 32)
!20 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!21 = !{!22}
!22 = !DILocalVariable(name: "self", arg: 1, scope: !15, file: !1, line: 15, type: !19)
!23 = !{i64 0, !"_ZTSFiPcE"}
!24 = !DIExpression()
!25 = !DILocation(line: 15, column: 17, scope: !15)
!26 = !DILocation(line: 16, column: 3, scope: !15)
!27 = !DILocation(line: 17, column: 3, scope: !15)

