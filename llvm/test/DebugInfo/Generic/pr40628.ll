; RUN: opt -early-cse -S %s -o - | FileCheck %s

; PR40628: The first load below is determined to be redundant by EarlyCSE.
; During salvaging, the corresponding dbg.value could have a DW_OP_deref used
; in its DIExpression to represent the redundant operation -- however LLVM
; cannot currently determine that the subsequent store should terminate the
; variables location range. A debugger would display zero for the "redundant"
; variable after stepping onto the return instruction.

; Test that the load being removed results in the corresponding dbg.value
; being assigned the 'undef' value.

; CHECK:      @foo
; CHECK-NEXT: dbg.value(metadata i32 undef, metadata ![[DEADVAR:[0-9]+]],
; CHECK-NEXT: load
; CHECK-NEXT: dbg.value(metadata i32 %{{[0-9]+}}, metadata ![[LIVEVAR:[0-9]+]],
; CHECK-NEXT: store
; CHECK-NEXT: ret

; CHECK:      ![[DEADVAR]] = !DILocalVariable(name: "redundant",
; CHECK:      ![[LIVEVAR]] = !DILocalVariable(name: "loaded",

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @foo(i32*) !dbg !7 {
  %2 = load i32, i32* %0, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %2, metadata !16, metadata !DIExpression()), !dbg !23
  %3 = load i32, i32* %0, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %3, metadata !17, metadata !DIExpression()), !dbg !23
  store i32 0, i32* %0, align 4, !dbg !23
  ret i32 %3, !dbg !23
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "pr40628.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !{!16, !17}
!16 = !DILocalVariable(name: "redundant", scope: !7, file: !1, line: 4, type: !10)
!17 = !DILocalVariable(name: "loaded", scope: !7, file: !1, line: 5, type: !10)
!23 = !DILocation(line: 4, column: 7, scope: !7)
