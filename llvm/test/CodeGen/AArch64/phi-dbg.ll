; RUN: llc -O0 %s -mtriple=aarch64 -o - | FileCheck %s

; Test that a DEBUG_VALUE node is create for variable c after the phi has been
; converted to a ldr.    The DEBUG_VALUE must be *after* the ldr and not before it.

; Created from the C code, compiled with -O0 -g and then passed through opt -mem2reg:
;
; int func(int a)
; {
;         int c = 1;
;         if (a < 0 ) {
;                 c = 12;
;         }
;         return c;
; }
;
; Function Attrs: nounwind
define i32 @func(i32) #0 !dbg !8 {
  call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !12, metadata !13), !dbg !14
  call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !15, metadata !13), !dbg !16
  %2 = icmp slt i32 %0, 0, !dbg !17
  br i1 %2, label %3, label %4, !dbg !19

; <label>:3:                                      ; preds = %1
  call void @llvm.dbg.value(metadata i32 12, i64 0, metadata !15, metadata !13), !dbg !16
  br label %4, !dbg !20

; <label>:4:                                      ; preds = %3, %1
  %.0 = phi i32 [ 12, %3 ], [ 1, %1 ]
; CHECK: ldr     w[[REG:[0-9]+]], [sp, #8]
; CHECK-NEXT: .Ltmp
  call void @llvm.dbg.value(metadata i32 %.0, i64 0, metadata !15, metadata !13), !dbg !16
; CHECK-NEXT:  //DEBUG_VALUE: func:c <- %w[[REG]]
  %5 = add nsw i32 %.0, %0, !dbg !22
  call void @llvm.dbg.value(metadata i32 %5, i64 0, metadata !15, metadata !13), !dbg !16
  ret i32 %5, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 14, scope: !8)
!15 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 3, type: !11)
!16 = !DILocation(line: 3, column: 13, scope: !8)
!17 = !DILocation(line: 4, column: 15, scope: !18)
!18 = distinct !DILexicalBlock(scope: !8, file: !1, line: 4, column: 13)
!19 = !DILocation(line: 4, column: 13, scope: !8)
!20 = !DILocation(line: 6, column: 9, scope: !21)
!21 = distinct !DILexicalBlock(scope: !18, file: !1, line: 4, column: 21)
!22 = !DILocation(line: 7, column: 4, scope: !8)
!23 = !DILocation(line: 8, column: 9, scope: !8)
