; RUN: llc -O0 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"

;CHECK-LABEL: foo:
;CHECK: .loc	1 2 11 prologue_end
define i32 @foo(i32 %i) nounwind ssp !dbg !1 {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %i.addr, metadata !7, metadata !DIExpression()), !dbg !8
  call void @llvm.dbg.declare(metadata ptr %j, metadata !9, metadata !DIExpression()), !dbg !11
  store i32 2, ptr %j, align 4, !dbg !12
  %tmp = load i32, ptr %j, align 4, !dbg !13
  %inc = add nsw i32 %tmp, 1, !dbg !13
  store i32 %inc, ptr %j, align 4, !dbg !13
  %tmp1 = load i32, ptr %j, align 4, !dbg !14
  %tmp2 = load i32, ptr %i.addr, align 4, !dbg !14
  %add = add nsw i32 %tmp1, %tmp2, !dbg !14
  store i32 %add, ptr %j, align 4, !dbg !14
  %tmp3 = load i32, ptr %j, align 4, !dbg !15
  ret i32 %tmp3, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

;CHECK-LABEL: main:
;CHECK: .loc 1 8 2 prologue_end

define i32 @main() nounwind ssp !dbg !6 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, !dbg !22
  %call = call i32 @foo(i32 21), !dbg !16
  ret i32 %call, !dbg !16
}

;; int foo(int arg) {
;;   while (arg)
;;    arg--;
;;  return 0;
;; }
;;
;; In this function, the entry block will fall through to while.cond, with no
;; instructions having source-locations. The expectations at -O0 is that we'll
;; put prologue_end on the first instruction of the loop, after %arg.addr is
;; initialized.

; CHECK:      _bar:
; CHECK-NEXT: Lfunc_begin2:
; CHECK-NEXT:     .loc    1 11 0 is_stmt 1
; CHECK-NEXT:     .cfi_startproc
; CHECK-NEXT: ## %bb.0:
; CHECK-NEXT:     movl    %edi, -4(%rsp)
; CHECK-NEXT: LBB2_1:
; CHECK-NEXT:                  ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT: Ltmp4:
; CHECK-NEXT:     .loc    1 12 3 prologue_end
; CHECK-NEXT:     cmpl    $0, -4(%rsp)

define dso_local i32 @bar(i32 noundef %arg) !dbg !30 {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, ptr %arg.addr, align 4
  br label %while.cond, !dbg !37

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, ptr %arg.addr, align 4, !dbg !38
  %tobool = icmp ne i32 %0, 0, !dbg !37
  br i1 %tobool, label %while.body, label %while.end, !dbg !37

while.body:                                       ; preds = %while.cond
  %1 = load i32, ptr %arg.addr, align 4, !dbg !39
  %dec = add nsw i32 %1, -1, !dbg !39
  store i32 %dec, ptr %arg.addr, align 4, !dbg !39
  br label %while.cond, !dbg !37

while.end:                                        ; preds = %while.cond
  ret i32 0, !dbg !42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}
!18 = !{!1, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 131100)", isOptimized: false, emissionKind: FullDebug, file: !19, enums: !20, retainedTypes: !20, imports:  null)
!1 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !19, scope: !2, type: !3)
!2 = !DIFile(filename: "/tmp/a.c", directory: "/private/tmp")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 7, file: !19, scope: !2, type: !3)
!7 = !DILocalVariable(name: "i", line: 1, arg: 1, scope: !1, file: !2, type: !5)
!8 = !DILocation(line: 1, column: 13, scope: !1)
!9 = !DILocalVariable(name: "j", line: 2, scope: !10, file: !2, type: !5)
!10 = distinct !DILexicalBlock(line: 1, column: 16, file: !19, scope: !1)
!11 = !DILocation(line: 2, column: 6, scope: !10)
!12 = !DILocation(line: 2, column: 11, scope: !10)
!13 = !DILocation(line: 3, column: 2, scope: !10)
!14 = !DILocation(line: 4, column: 2, scope: !10)
!15 = !DILocation(line: 5, column: 2, scope: !10)
!16 = !DILocation(line: 8, column: 2, scope: !17)
!17 = distinct !DILexicalBlock(line: 7, column: 12, file: !19, scope: !6)
!19 = !DIFile(filename: "/tmp/a.c", directory: "/private/tmp")
!20 = !{}
!21 = !{i32 1, !"Debug Info Version", i32 3}
!22 = !DILocation(line: 0, column: 0, scope: !17)
!30 = distinct !DISubprogram(name: "bar", scope: !2, file: !2, line: 10, type: !3, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
!34 = !{}
!36 = !DILocation(line: 11, column: 13, scope: !30)
!37 = !DILocation(line: 12, column: 3, scope: !30)
!38 = !DILocation(line: 12, column: 10, scope: !30)
!39 = !DILocation(line: 13, column: 8, scope: !30)
!42 = !DILocation(line: 14, column: 3, scope: !30)
