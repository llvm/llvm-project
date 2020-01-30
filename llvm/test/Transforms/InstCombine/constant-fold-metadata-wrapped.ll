; RUN: opt -S -instcombine %s -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { i32 }

@global = common global %struct.Foo zeroinitializer, align 4

; Verify that we constant fold the GEP in the llvm.dbg.value intrinsic. We
; want it to match with the non-dbg GEP, so that the debug information can be
; emitted later on.

; When constant folding the GEP the index operand types are canonicalized, so
; we get an i64 operand here.
; CHECK: call void @llvm.dbg.value(metadata i32* getelementptr inbounds (%struct.Foo, %struct.Foo* @global, i64 0, i32 0)
; CHECK: call void @ext(i32* getelementptr inbounds (%struct.Foo, %struct.Foo* @global, i64 0, i32 0))

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i32* getelementptr inbounds (%struct.Foo, %struct.Foo* @global, i32 0, i32 0), metadata !17, metadata !DIExpression()), !dbg !18
  call void @ext(i32* getelementptr inbounds (%struct.Foo, %struct.Foo* @global, i32 0, i32 0)), !dbg !19
  ret i32 0, !dbg !20
}

declare !dbg !4 void @ext(i32*)

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "ext", scope: !1, file: !1, line: 3, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0 "}
!13 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !14, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!8}
!16 = !{!17}
!17 = !DILocalVariable(name: "local", scope: !13, file: !1, line: 6, type: !7)
!18 = !DILocation(line: 0, scope: !13)
!19 = !DILocation(line: 7, scope: !13)
!20 = !DILocation(line: 8, scope: !13)
