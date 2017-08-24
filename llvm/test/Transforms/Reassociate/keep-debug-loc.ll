; RUN: opt -S -reassociate < %s | FileCheck %s

; PR34231
;
; Verify that the original debug location is kept if the
; replacement debug location is missing when
; reassociating expressions.

define i16 @fn1() !dbg !3 {
  ret i16 undef
}

define void @fn2() !dbg !6 {
; CHECK-LABEL: @fn2
; CHECK: call i16 @fn1(), !dbg ![[LOC1:[0-9]+]]
; CHECK-NOT: or i16
  %inlinable_call = call i16 @fn1(), !dbg !7
  %dbgless_instruction = or i16 %inlinable_call, 0
  store i16 %dbgless_instruction, i16* undef, align 1
  unreachable
}

define void @fn3() !dbg !8 {
; CHECK-LABEL: @fn3
; CHECK: load i16, i16* undef, !dbg ![[LOC2:[0-9]+]]
; CHECK-NOT: or i16
  %instruction = load i16, i16* undef, !dbg !9
  %dbgless_instruction = or i16 %instruction, 0
  store i16 %dbgless_instruction, i16* undef, align 1
  unreachable
}

; CHECK: ![[LOC1]] = !DILocation(line: 7
; CHECK: ![[LOC2]] = !DILocation(line: 9

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 2, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 3, type: !4, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0)
!7 = !DILocation(line: 7, column: 10, scope: !6)
!8 = distinct !DISubprogram(name: "fn3", scope: !1, file: !1, line: 8, type: !4, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0)
!9 = !DILocation(line: 9, column: 10, scope: !8)
