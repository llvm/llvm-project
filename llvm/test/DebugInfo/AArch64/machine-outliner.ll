; RUN: llc -verify-machineinstrs -filetype=obj -mtriple=aarch64-- \
; RUN: -enable-machine-outliner < %s | llvm-dwarfdump - | FileCheck %s

; Ensure that the MachineOutliner produces valid DWARF when it creates outlined
; functions. Check that the outlined function has a subprogram, and that the
; expected DWARF attributes are present.

; The outlined function appears after bar. Skip past the functions we don't
; care about.
; CHECK: DW_TAG_compile_unit
; CHECK-DAG: DW_AT_name	("foo")
; CHECK-DAG: DW_AT_name	("p")

; Check the high address of bar. This is one past the end of bar. It should be
; the beginning of the outlined function.
; CHECK:      DW_AT_high_pc	([[ONE_PAST_BAR:0x[a-f0-9]+]])
; CHECK-NEXT: DW_AT_frame_base	(DW_OP_reg29 W29)
; CHECK-NEXT: DW_AT_name	("bar")

; Check the outlined function's DWARF.
; CHECK-DAG:  DW_TAG_subprogram
; CHECK-NEXT: DW_AT_low_pc	([[ONE_PAST_BAR]])
; CHECK-NEXT: DW_AT_high_pc	(0x{{[0-9a-f]+}})
; CHECK-NEXT: DW_AT_frame_base	(DW_OP_reg29 W29)
; CHECK-NEXT: DW_AT_MIPS_linkage_name	("[[NAME:OUTLINED_FUNCTION_[0-9]+]]")
; CHECK-NEXT: DW_AT_name	("[[NAME]]")
; CHECK-NEXT: DW_AT_artificial	(0x01)
; CHECK-NEXT: DW_AT_external	(0x01)

define void @foo() #0 !dbg !8 {
entry:
  %p = alloca ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %p, metadata !11, metadata !DIExpression()), !dbg !14
  %0 = load ptr, ptr %p, align 8, !dbg !15
  %incdec.ptr = getelementptr inbounds i32, ptr %0, i32 1, !dbg !15
  store ptr %incdec.ptr, ptr %p, align 8, !dbg !15
  call void @foo(), !dbg !16
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define void @bar() #0 !dbg !18 {
entry:
  %p = alloca ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %p, metadata !19, metadata !DIExpression()), !dbg !20
  %0 = load ptr, ptr %p, align 8, !dbg !21
  %incdec.ptr = getelementptr inbounds i32, ptr %0, i32 1, !dbg !21
  store ptr %incdec.ptr, ptr %p, align 8, !dbg !21
  call void @foo(), !dbg !22
  ret void, !dbg !23
}

attributes #0 = { nounwind ssp uwtable "frame-pointer"="all" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "something.c", directory: "somewhere")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "p", scope: !8, file: !1, line: 2, type: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, column: 8, scope: !8)
!15 = !DILocation(line: 3, column: 4, scope: !8)
!16 = !DILocation(line: 4, column: 3, scope: !8)
!17 = !DILocation(line: 5, column: 1, scope: !8)
!18 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !9, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!19 = !DILocalVariable(name: "p", scope: !18, file: !1, line: 9, type: !12)
!20 = !DILocation(line: 9, column: 8, scope: !18)
!21 = !DILocation(line: 10, column: 4, scope: !18)
!22 = !DILocation(line: 11, column: 3, scope: !18)
!23 = !DILocation(line: 12, column: 1, scope: !18)

