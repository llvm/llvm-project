; RUN: opt -run-twice -passes=verify -S -o - %s | FileCheck %s

; The ValueMap shared between CloneFunctionInto calls within CloneModule needs
; to contain identity mappings for all of the DISubprogram's to prevent them
; from being duplicated by MapMetadata / RemapInstruction calls, this is
; achieved via DebugInfoFinder collecting all the DISubprogram's. However,
; CloneFunctionInto was missing calls into DebugInfoFinder for functions w/o
; DISubprogram's attached, but still referring DISubprogram's from within.
;
; This is to make sure we don't regress on that.

; Derived from the following C-snippet
;
;   int inlined(int j);
;   __attribute__((nodebug)) int nodebug(int k) { return inlined(k); }
;   __attribute__((always_inline)) int inlined(int j) { return j * 2; }
;
; compiled with `clang -O1 -g3 -emit-llvm -S` by removing
;
;   call void @llvm.dbg.value(metadata i32 %k, metadata !8, metadata !DIExpression()), !dbg !14
;
; line from @nodebug function.

; The @llvm.dbg.value call is manually removed from @nodebug as not having
; it there also may cause an incorrect remapping of the call in a case of a
; regression, not just a duplication of a DISubprogram. Namely, the call's
; metadata !8 2nd argument and the !dbg !14 debug location may get remapped
; to reference different copies of the DISubprogram, which is verified by IR
; Verifier, while having DISubprogram duplicates is not.

; CHECK:     DISubprogram
; CHECK-NOT: DISubprogram

source_filename = "clone-module.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; Function Attrs: nounwind readnone ssp uwtable
define i32 @nodebug(i32 %k) local_unnamed_addr #0 {
entry:
  %mul.i = shl nsw i32 %k, 1, !dbg !15
  ret i32 %mul.i
}

; Function Attrs: alwaysinline nounwind readnone ssp uwtable
define i32 @inlined(i32 %j) local_unnamed_addr #1 !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata i32 %j, metadata !8, metadata !DIExpression()), !dbg !14
  %mul = shl nsw i32 %j, 1, !dbg !15
  ret i32 %mul, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { alwaysinline nounwind readnone ssp uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "clone-module.c", directory: "/somewhere")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 7.0.0"}
!8 = !DILocalVariable(name: "j", arg: 1, scope: !9, file: !1, line: 3, type: !12)
!9 = distinct !DISubprogram(name: "inlined", scope: !1, file: !1, line: 3, type: !10, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!8}
!14 = !DILocation(line: 3, column: 48, scope: !9)
!15 = !DILocation(line: 3, column: 62, scope: !9)
!16 = !DILocation(line: 3, column: 53, scope: !9)
