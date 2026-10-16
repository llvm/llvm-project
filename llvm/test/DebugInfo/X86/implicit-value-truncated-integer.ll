; RUN: llc -mtriple=i386-unknown-linux-gnu -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; A constant debug value may use a carrier integer whose bits do not fit in the
; declared source type. When emitting source-sized DW_OP_implicit_value bytes,
; truncate the carrier value to the source type instead of asserting.

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location
; CHECK: DW_OP_implicit_value 0x6 0x01 0x00 0x00 0x00 0x00 0x00
; CHECK: DW_OP_implicit_value 0x6 0xff 0xff 0xff 0xff 0xff 0xff
; CHECK: DW_AT_name {{.*}}"x"

target triple = "i386-unknown-linux-gnu"

define void @test(ptr %p) !dbg !4 {
entry:
  #dbg_value(i64 281474976710657, !6, !DIExpression(), !8)
  store volatile i32 0, ptr %p, !dbg !8
  #dbg_value(i64 -1, !6, !DIExpression(), !9)
  store volatile i32 1, ptr %p, !dbg !9
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "x", scope: !4, file: !1, type: !7)
!7 = !DIBasicType(name: "unsigned _BitInt(48)", size: 48, encoding: DW_ATE_unsigned)
!8 = !DILocation(line: 1, scope: !4)
!9 = !DILocation(line: 2, scope: !4)
!10 = !DILocation(line: 3, scope: !4)
!11 = !{i32 7, !"Dwarf Version", i32 4}
