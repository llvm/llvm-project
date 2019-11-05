; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj -o - %s | llvm-dwarfdump -debug-loc - | FileCheck %s

; The test inlines the function F four times, with each inlined variable for
; "i4" sharing the same virtual register. This means the live interval of the
; register spans all of the inlined callsites, extending beyond the lexical
; scope of each. Later during register allocation the live interval is split
; into multiple intervals. Check that this does not generate multiple entries
; within the debug location (see PR33730).
;
; Generated from:
;
; extern int foobar(int, int, int, int, int);
;
; int F(int i1, int i2, int i3, int i4, int i5) {
;   return foobar(i1, i2, i3, i4, i5);
; }
;
; int foo(int a, int b, int c, int d, int e) {
;   return F(a,b,c,d,e) +
;          F(a,b,c,d,e) +
;          F(a,b,c,d,e) +
;          F(a,b,c,d,e);
; }

; CHECK:      .debug_loc contents:
; CHECK-NEXT: 0x00000000:
;   We currently emit an entry for the function prologue, too, which could be optimized away.
; CHECK:              (0x0000000000000010, 0x0000000000000072): DW_OP_reg3 RBX
;   We should only have one entry inside the function.
; CHECK-NOT: :

declare i32 @foobar(i32, i32, i32, i32, i32)

define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) !dbg !25 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %d, i64 0, metadata !15, metadata !17) #3, !dbg !41
  %call.i = tail call i32 @foobar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #3, !dbg !43
  %call.i21 = tail call i32 @foobar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #3, !dbg !50
  %add = add nsw i32 %call.i21, %call.i, !dbg !51
  %call.i22 = tail call i32 @foobar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #3, !dbg !58
  %add3 = add nsw i32 %add, %call.i22, !dbg !59
  %call.i23 = tail call i32 @foobar(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) #3, !dbg !66
  %add5 = add nsw i32 %add3, %call.i23, !dbg !67
  ret i32 %add5, !dbg !68
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 308976)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 308976)"}
!7 = distinct !DISubprogram(name: "F", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!15}
!15 = !DILocalVariable(name: "i4", arg: 4, scope: !7, file: !1, line: 3, type: !10)
!17 = !DIExpression()
!25 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 7, type: !8, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !26)
!26 = !{}
!38 = distinct !DILocation(line: 8, column: 10, scope: !25)
!41 = !DILocation(line: 3, column: 35, scope: !7, inlinedAt: !38)
!43 = !DILocation(line: 4, column: 10, scope: !7, inlinedAt: !38)
!45 = distinct !DILocation(line: 9, column: 10, scope: !25)
!50 = !DILocation(line: 4, column: 10, scope: !7, inlinedAt: !45)
!51 = !DILocation(line: 8, column: 23, scope: !25)
!53 = distinct !DILocation(line: 10, column: 10, scope: !25)
!58 = !DILocation(line: 4, column: 10, scope: !7, inlinedAt: !53)
!59 = !DILocation(line: 9, column: 23, scope: !25)
!61 = distinct !DILocation(line: 11, column: 10, scope: !25)
!66 = !DILocation(line: 4, column: 10, scope: !7, inlinedAt: !61)
!67 = !DILocation(line: 10, column: 23, scope: !25)
!68 = !DILocation(line: 8, column: 3, scope: !25)
