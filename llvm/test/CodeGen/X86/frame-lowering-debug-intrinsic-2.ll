; Test ensuring debug intrinsics do not affect generated function prologue.
;
; RUN: llc -O1 -mtriple=x86_64-unknown-unknown -o - %s | FileCheck %s

@a = local_unnamed_addr global i64 0, align 8

define void @noDebug() {
entry:
  %0 = load i64, i64* @a, align 8
  %1 = load i64, i64* @a, align 8
  %2 = load i64, i64* @a, align 8
  %3 = tail call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %0, i64 %1)
  %4 = extractvalue { i64, i1 } %3, 0
  %5 = tail call i64 @fn1(i64 %4, i64 %2)
  tail call void (...) @printf()
  tail call void (...) @printf(i64 1, i64 2, i64 3, i64 4, i32 0, i64 0, i64 %4, i64 %5)
  ret void
}

; CHECK-LABEL: noDebug
; CHECK:        addq	$16, %rsp
; CHECK-NEXT: 	  .cfi_adjust_cfa_offset -16
; CHECK-NEXT: 	addq	$8, %rsp
; CHECK-NEXT: 	  .cfi_def_cfa_offset 24
; CHECK-NEXT: 	popq	%rbx
; CHECK-NEXT: 	  .cfi_def_cfa_offset 16
; CHECK-NEXT: 	popq	%r14
; CHECK-NEXT: 	  .cfi_def_cfa_offset 8
; CHECK-NEXT: 	retq

define void @withDebug() !dbg !18 {
entry:
  %0 = load i64, i64* @a, align 8
  %1 = load i64, i64* @a, align 8
  %2 = load i64, i64* @a, align 8
  %3 = tail call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %0, i64 %1)
  %4 = extractvalue { i64, i1 } %3, 0
  %5 = tail call i64 @fn1(i64 %4, i64 %2)
  tail call void @llvm.dbg.value(metadata i64 %4, i64 0, metadata !23, metadata !33), !dbg !34
  tail call void @llvm.dbg.value(metadata i64 %5, i64 0, metadata !22, metadata !33), !dbg !35
  tail call void (...) @printf()
  tail call void (...) @printf(i64 1, i64 2, i64 3, i64 4, i32 0, i64 0, i64 %4, i64 %5)
  ret void
}

; CHECK-LABEL: withDebug
; CHECK:       callq printf
; CHECK:       callq printf
; CHECK-NEXT: addq $16, %rsp
; CHECK:       popq  %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:  popq  %r14
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:  retq

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @fn1(i64, i64)

declare void @printf(...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)


!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!15, !16}

!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0")
!2 = !DIFile(filename: "test.cpp", directory: "")
!11 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!18 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 5, unit: !1)
!22 = !DILocalVariable(name: "i", scope: !18, file: !2, line: 6, type: !11)
!23 = !DILocalVariable(name: "j", scope: !18, file: !2, line: 7, type: !11)
!33 = !DIExpression()
!34 = !DILocation(line: 7, column: 17, scope: !18)
!35 = !DILocation(line: 6, column: 8, scope: !18)
!36 = !DILocation(line: 9, column: 3, scope: !18)
!37 = !DILocation(line: 10, column: 10, scope: !18)
