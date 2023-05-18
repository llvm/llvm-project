; RUN: llc -march=hexagon < %s 2>&1 | FileCheck %s

; Check that the misaligned store is diagnosed.
; CHECK: remark: Misaligned constant address: 0x00012345 has alignment 1, but the memory access requires 4, at misaligned-const-store.c:2:10. The instruction has been replaced with a trap.

target triple = "hexagon"

define void @bad_store(i32 %a0) #0 !dbg !10 {
entry:
  store i32 %a0, ptr inttoptr (i32 74565 to ptr), align 4, !dbg !13, !tbaa !14
  ret void, !dbg !18
}

attributes #0 = { norecurse nounwind readonly "target-cpu"="hexagonv60" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "misaligned-const-store.c", directory: "/test")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 32)
!5 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 8.0.0"}
!10 = distinct !DISubprogram(name: "bad_store", scope: !1, file: !1, line: 1, type: !11, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{!5}
!13 = !DILocation(line: 2, column: 10, scope: !10)
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 2, column: 3, scope: !10)
