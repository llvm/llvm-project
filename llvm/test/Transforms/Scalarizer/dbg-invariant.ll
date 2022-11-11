; RUN: opt -strip-debug -passes=scalarizer -S < %s | FileCheck %s
; RUN: opt -passes=scalarizer -S < %s | FileCheck %s

; This input caused the scalarizer to violate a debug info
; invariance by using the wrong insertion point.

; CHECK: %0 = load <8 x i16>

; CHECK: %.i0 = extractelement <8 x i16> %0, i32 0
; CHECK-NEXT: %.i01 = add i16 %.i0, 28690
; CHECK: %.i1 = extractelement <8 x i16> %0, i32 1
; CHECK-NEXT: %.i12 = add i16 %.i1, 28690
; CHECK: %.i2 = extractelement <8 x i16> %0, i32 2
; CHECK-NEXT: %.i23 = add i16 %.i2, 28690
; CHECK: %.i3 = extractelement <8 x i16> %0, i32 3
; CHECK-NEXT: %.i34 = add i16 %.i3, 28690
; CHECK: %.i4 = extractelement <8 x i16> %0, i32 4
; CHECK-NEXT: %.i45 = add i16 %.i4, 28690
; CHECK: %.i5 = extractelement <8 x i16> %0, i32 5
; CHECK-NEXT: %.i56 = add i16 %.i5, 28690
; CHECK: %.i6 = extractelement <8 x i16> %0, i32 6
; CHECK-NEXT: %.i67 = add i16 %.i6, 28690
; CHECK: %.i7 = extractelement <8 x i16> %0, i32 7
; CHECK-NEXT: = add i16 %.i7, 28690

@d = external global [8 x i16], align 1
@e = external global i16, align 1

; Function Attrs: nofree norecurse nounwind
define dso_local void @foo() local_unnamed_addr #0 !dbg !7 {
entry:
  %0 = load <8 x i16>, ptr @d, align 1
  call void @llvm.dbg.value(metadata i16 0, metadata !11, metadata !DIExpression()), !dbg !13
  %1 = add <8 x i16> %0, <i16 28690, i16 28690, i16 28690, i16 28690, i16 28690, i16 28690, i16 28690, i16 28690>, !dbg !13
  store <8 x i16> %1, ptr @d, align 1, !dbg !13
  %2 = extractelement <8 x i16> %1, i32 7, !dbg !13
  store i16 %2, ptr @e, align 1, !dbg !13
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nofree norecurse nounwind }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 1}
!6 = !{!"clang version 11.0.0 "}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !7)
