; RUN: opt -S -globalopt < %s | FileCheck %s
; struct {
;   int f0;
; } static a;
; int main() {
;   a.f0++;
;   return 0;
; }

source_filename = "pr34390.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.anon = type { i32 }

; CHECK: @a.0 = internal unnamed_addr global i32 0, align 4, !dbg ![[GVE:.*]]
@a = internal global %struct.anon zeroinitializer, align 4, !dbg !0

define i32 @main() #0 !dbg !15 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @a, i32 0, i32 0), align 4, !dbg !18
  %inc = add nsw i32 %0, 1, !dbg !18
  store i32 %inc, i32* getelementptr inbounds (%struct.anon, %struct.anon* @a, i32 0, i32 0), align 4, !dbg !18
  ret i32 0, !dbg !19
}

attributes #0 = { noinline nounwind optnone ssp uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}

; CHECK: ![[GVE]] = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 3, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 312175) (llvm/trunk 312146)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 32, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f0", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!15 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !16, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !2, retainedNodes: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{!9}
!18 = !DILocation(line: 5, column: 7, scope: !15)
!19 = !DILocation(line: 6, column: 3, scope: !15)
