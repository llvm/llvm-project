; RUN: llc %s -stop-after=finalize-isel -o - | FileCheck %s


; RUN: llc --try-experimental-debuginfo-iterators %s -stop-after=finalize-isel -o - | FileCheck %s

;; Hand written. Check AssignmentTrackingAnalysis doesn't try to get the size
;; of scalable vectors (which causes an assertion failure).

; CHECK: DBG_VALUE %stack.0, $noreg, ![[#]], !DIExpression(DW_OP_deref)
; CHECK: DBG_VALUE i256 0, $noreg, ![[#]], !DIExpression()
;; FIXME: We should reinstate (or poison) the variable at the store. See
;; llvm.org/PR62346.

target triple = "aarch64"

define dso_local void @square(ptr %0) local_unnamed_addr #0 !dbg !9 {
entry:
   %1 = alloca <32 x i8>, !DIAssignID !18
   call void @llvm.dbg.assign(metadata i1 poison, metadata !19, metadata !DIExpression(), metadata !18, metadata ptr %1, metadata !DIExpression()), !dbg !21
   %2 = load <vscale x 8 x i8>, ptr %0, align 1
   call void @llvm.dbg.assign(metadata i256 0, metadata !19, metadata !DIExpression(), metadata !22, metadata ptr %1, metadata !DIExpression()), !dbg !21
   store <vscale x 8 x i8> %2, ptr %1, align 1
   ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

attributes #0 = { vscale_range(1,16) "target-cpu"="generic" "target-features"="+sve" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.c", directory: "/app")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!8 = !{!"clang version 17.0.0"}
!9 = distinct !DISubprogram(name: "square", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!10 = !DIFile(filename: "example.c", directory: "/app")
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14, !16}
!13 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{}
!18 =  distinct !DIAssignID()
!19 = !DILocalVariable(name: "i", scope: !9, file: !10, line: 2, type: !20)
!20 = !DIBasicType(name: "cats", size: 256, encoding: DW_ATE_signed)
!21 = !DILocation(line: 0, scope: !9)
!22 =  distinct !DIAssignID()

