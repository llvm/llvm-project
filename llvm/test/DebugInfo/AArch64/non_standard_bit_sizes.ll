; RUN: llc -mtriple arm64-apple-darwin -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -all %t | FileCheck %s

; Checks that bit sizes that are not byte aligned are rounded up.

; Check that a 1 bit sized type gets emitted as 1 bit long.
; CHECK: DW_AT_name	("one_bit_int")
; CHECK-NEXT: DW_AT_encoding	(DW_ATE_signed)
; CHECK-NEXT: DW_AT_bit_size	(0x01)

; Check that a 9 bit sized type gets emitted as 2 bytes long.
; CHECK: DW_AT_name	("nine_bit_double")
; CHECK-NEXT: DW_AT_encoding	(DW_ATE_float)
; CHECK-NEXT: DW_AT_bit_size	(0x09)

; Check that a 7 bit sized type gets emitted as 1 bytes long.
; CHECK: DW_AT_name	("seven_bit_float")
; CHECK-NEXT: DW_AT_encoding	(DW_ATE_float)
; CHECK-NEXT: DW_AT_bit_size	(0x07)

; Check that a byte aligned bit size is emitted with the same byte size.
; CHECK: DW_AT_name	("four_byte_S")
; CHECK-NEXT: DW_AT_byte_size	(0x04)

%struct.S = type { i32 }

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @func(i32 noundef %a, double noundef %b, float noundef %c, i64 %s.coerce) !dbg !10 {
entry:
  %s = alloca %struct.S, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca double, align 8
  %c.addr = alloca float, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata ptr %b.addr, metadata !22, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata ptr %c.addr, metadata !24, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata ptr %s, metadata !26, metadata !DIExpression()), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) 

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!7}

!2 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !8, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!8 = !DIFile(filename: "t.c", directory: "")
!10 = distinct !DISubprogram(name: "func", scope: !8, file: !8, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !19)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14, !15, !16}
!13 = !DIBasicType(name: "one_bit_int", size: 1, encoding: DW_ATE_signed)
!14 = !DIBasicType(name: "nine_bit_double", size: 9, encoding: DW_ATE_float)
!15 = !DIBasicType(name: "seven_bit_float", size: 7, encoding: DW_ATE_float)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "four_byte_S", file: !8, line: 2, size: 32, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !16, file: !8, line: 3, baseType: !13, size: 32)
!19 = !{}
!20 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !8, line: 6, type: !13)
!21 = !DILocation(line: 6, column: 15, scope: !10)
!22 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !8, line: 6, type: !14)
!23 = !DILocation(line: 6, column: 25, scope: !10)
!24 = !DILocalVariable(name: "c", arg: 3, scope: !10, file: !8, line: 6, type: !15)
!25 = !DILocation(line: 6, column: 34, scope: !10)
!26 = !DILocalVariable(name: "s", arg: 4, scope: !10, file: !8, line: 6, type: !16)
!27 = !DILocation(line: 6, column: 46, scope: !10)
!28 = !DILocation(line: 7, column: 1, scope: !10)

