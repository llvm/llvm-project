; Test dwarf codegen of DW_OP_minus.
; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; This was built by compiling the following source with SafeStack and
; simplifying the result a little.
; extern "C" {
; void Capture(int *);
; void f() {
;   int buf[100];
;   Capture(buf);
; }
; }
; The interesting part is !DIExpression(DW_OP_deref, DW_OP_minus, 400)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__safestack_unsafe_stack_ptr = external thread_local(initialexec) global i8*

define void @f() !dbg !4 {
entry:
  %unsafe_stack_ptr = load i8*, i8** @__safestack_unsafe_stack_ptr
  %unsafe_stack_static_top = getelementptr i8, i8* %unsafe_stack_ptr, i32 -400
  store i8* %unsafe_stack_static_top, i8** @__safestack_unsafe_stack_ptr
  %0 = getelementptr i8, i8* %unsafe_stack_ptr, i32 -400
  %buf = bitcast i8* %0 to [100 x i32]*
  %1 = bitcast [100 x i32]* %buf to i8*, !dbg !16
  call void @llvm.dbg.declare(metadata i8* %unsafe_stack_ptr, metadata !8, metadata !17), !dbg !18
  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %buf, i64 0, i64 0, !dbg !19
  call void @Capture(i32* %arraydecay), !dbg !20
  store i8* %unsafe_stack_ptr, i8** @__safestack_unsafe_stack_ptr, !dbg !21
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @Capture(i32*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "1.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, variables: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "buf", scope: !4, file: !1, line: 5, type: !9)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 3200, align: 32, elements: !11)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 100)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)"}
!16 = !DILocation(line: 5, column: 3, scope: !4)
!17 = !DIExpression(DW_OP_deref, DW_OP_minus, 400)
!18 = !DILocation(line: 5, column: 7, scope: !4)
!19 = !DILocation(line: 6, column: 11, scope: !4)
!20 = !DILocation(line: 6, column: 3, scope: !4)
!21 = !DILocation(line: 7, column: 1, scope: !4)

; RCX - 400
; CHECK:      .short	6                       # Loc expr size
; CHECK-NEXT: .byte	114                     # DW_OP_breg2
; CHECK-NEXT: .byte	0                       # 0
; CHECK-NEXT: .byte	16                      # DW_OP_constu
; CHECK-NEXT: .byte	144                     # 400
; CHECK-NEXT: .byte	3                       # DW_OP_minus
; CHECK-NEXT: .byte	28

; RCX is clobbered in call @Capture, but there is a spilled copy.
; *(RSP + 8) - 400
; CHECK:      .short	7                       # Loc expr size
; CHECK-NEXT: .byte	119                     # DW_OP_breg7
; CHECK-NEXT: .byte	8                       # 8
; CHECK-NEXT: .byte	6                       # DW_OP_deref
; CHECK-NEXT: .byte	16                      # DW_OP_constu
; CHECK-NEXT: .byte	144                     # 400
; CHECK-NEXT: .byte	3                       # DW_OP_minus
; CHECK-NEXT: .byte	28
