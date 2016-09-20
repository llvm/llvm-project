; RUN: opt < %s -asan -asan-module -S | FileCheck %s
source_filename = "version.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK: @version = constant { [5 x i8], [59 x i8] } {{.*}}, !dbg ![[GV:.*]]
@version = constant [5 x i8] c"4.00\00", align 1, !dbg !0

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

; CHECK: ![[GV]] = distinct !DIGlobalVariable(name: "version"
; CHECK-NOT: expr:
!0 = distinct !DIGlobalVariable(name: "version", scope: !1, file: !2, line: 2, type: !5, isLocal: false, isDefinition: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 281923) (llvm/trunk 281916)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "version.c", directory: "/Volumes/Fusion/Data/radar/24899262")
!3 = !{}
!4 = !{!0}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 40, align: 8, elements: !8)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!8 = !{!9}
!9 = !DISubrange(count: 5)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 4.0.0 (trunk 281923) (llvm/trunk 281916)"}
