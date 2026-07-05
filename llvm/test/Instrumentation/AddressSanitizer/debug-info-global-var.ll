; RUN: opt < %s -passes=asan -S | FileCheck %s
source_filename = "version.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"
; CHECK: @version = constant { [5 x i8], [27 x i8] } {{.*}}, !dbg ![[GV:.*]]

@version = constant [5 x i8] c"4.00\00", align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}
; Should not have an expression:
; CHECK: ![[GV]] = !DIGlobalVariableExpression(var: ![[GVAR:.*]], expr: !DIExpression())
; CHECK: ![[GVAR]] = !DIGlobalVariable(name: "version"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "version", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0 (trunk 281923) (llvm/trunk 281916)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "version.c", directory: "/Volumes/Fusion/Data/radar/24899262")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 40, align: 8, elements: !9)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = !{!10}
!10 = !DISubrange(count: 5)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"PIC Level", i32 2}
!14 = !{!"clang version 4.0.0 (trunk 281923) (llvm/trunk 281916)"}

