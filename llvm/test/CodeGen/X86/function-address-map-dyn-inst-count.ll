; RUN: llc < %s -mtriple=x86_64 -function-sections -func-map=dyn-inst-count | FileCheck %s


;; Check we add SHF_LINK_ORDER for .llvm_func_map and link it with the corresponding .text sections.
; CHECK:  .section .text.foo,"ax",@progbits
; CHECK-LABEL: foo:
; CHECK-NEXT:  [[FOO_BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:	.section .llvm_func_map,"o",@llvm_func_map,.text.foo{{$}}
; CHECK-NEXT:  .byte 1			            # version
; CHECK-NEXT:  .byte 1			            # feature
; CHECK-NEXT:  .quad [[FOO_BEGIN]]	    # function address
; CHECK-NEXT:  .ascii  "\252\001"       # dynamic instruction count


; CHECK:  .section .text.main,"ax",@progbits
; CHECK-LABEL: main:
; CHECK-NEXT:  [[MAIN_BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:  .section .llvm_func_map,"o",@llvm_func_map,.text.main{{$}}
; CHECK-NEXT:  .byte 1			            # version
; CHECK-NEXT:  .byte 1			            # feature
; CHECK-NEXT:  .quad [[MAIN_BEGIN]]	    # function address
; CHECK-NEXT:  .ascii  "\265\003"       # dynamic instruction count


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %x) !dbg !4 !prof !6 {
entry:
    #dbg_value(i32 %x, !7, !DIExpression(), !9)
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  %rem = mul i32 %x, 5
  %tobool.not = icmp eq i32 %rem, 0, !dbg !10
  br i1 %tobool.not, label %if.end, label %if.then, !prof !12

if.then:                                          ; preds = %entry
  %inc = add i32 0, 0
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 2, i32 0, i64 -1)
    #dbg_value(i32 %inc, !7, !DIExpression(), !9)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %x.addr.0 = phi i32 [ 0, %if.then ], [ 1, %entry ]
    #dbg_value(i32 %x.addr.0, !7, !DIExpression(), !9)
  ret i32 %x.addr.0
}

define i32 @main() #0 !dbg !13 !prof !15 {
entry:
    #dbg_value(i32 0, !16, !DIExpression(), !17)
  br label %while.cond

while.cond:                                       ; preds = %if.then, %if.else, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.else ], [ %inc, %if.then ]
    #dbg_value(i32 %i.0, !16, !DIExpression(), !17)
  %inc = add i32 %i.0, 1
    #dbg_value(i32 %inc, !16, !DIExpression(), !17)
  %cmp = icmp ult i32 %i.0, 1600000
  br i1 %cmp, label %while.body, label %while.end, !prof !18

while.body:                                       ; preds = %while.cond
  %rem = urem i32 %inc, 11
  %tobool.not = icmp eq i32 %rem, 0
  br i1 %tobool.not, label %if.else, label %if.then, !prof !19

if.then:                                          ; preds = %while.body
  %call = call i32 @foo(i32 0), !dbg !20
  %0 = load volatile i32, ptr null, align 4
  br label %while.cond

if.else:                                          ; preds = %while.body
  store i32 0, ptr null, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 0

; uselistorder directives
  uselistorder label %while.cond, { 1, 0, 2 }
  uselistorder i32 %inc, { 2, 1, 0 }
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home", checksumkind: CSK_MD5, checksum: "920887ee2258042655d8340f78e732e9")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = distinct !DISubroutineType(types: !2)
!6 = !{!"function_entry_count", i64 20}
!7 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 3, type: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 0, scope: !4)
!10 = !DILocation(line: 4, column: 9, scope: !11)
!11 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 7)
!12 = !{!"branch_weights", i32 15, i32 5}
!13 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 9, type: !14, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !2)
!15 = !{!"function_entry_count", i64 1}
!16 = !DILocalVariable(name: "i", scope: !13, file: !1, line: 10, type: !8)
!17 = !DILocation(line: 0, scope: !13)
!18 = !{!"branch_weights", i32 22, i32 1}
!19 = !{!"branch_weights", i32 2, i32 20}
!20 = !DILocation(line: 12, column: 22, scope: !21)
!21 = !DILexicalBlockFile(scope: !22, file: !1, discriminator: 455082031)
!22 = distinct !DILexicalBlock(scope: !13, file: !1, line: 12, column: 9)
