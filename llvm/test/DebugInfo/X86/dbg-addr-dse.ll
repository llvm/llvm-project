; RUN: llc %s -o %t.s
; RUN: llvm-mc %t.s -filetype=obj -triple=x86_64-windows-msvc -o %t.o
; RUN: FileCheck %s < %t.s --check-prefix=ASM
; RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix=DWARF

; In this example, the variable lives mostly in memory, but at the point of the
; assignment to global, it lives nowhere, and is described as the constant
; value 1.

; C source:
;
; void escape(int *);
; extern int global;
; void f(int x) {
;   escape(&x);
;   x = 1; // DSE should delete and insert dbg.value(i32 1)
;   global = x;
;   x = 2; // DSE should insert dbg.addr
;   escape(&x);
; }

; ModuleID = 'dse.c'
source_filename = "dse.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @llvm.dbg.addr(metadata, metadata, metadata) #2
declare void @llvm.dbg.value(metadata, metadata, metadata) #2
declare void @escape(i32*)

@global = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @f(i32 %x) #0 !dbg !8 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.addr(metadata i32* %x.addr, metadata !13, metadata !DIExpression()), !dbg !18
  call void @escape(i32* %x.addr), !dbg !19
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !20
  store i32 1, i32* @global, align 4, !dbg !22
  call void @llvm.dbg.addr(metadata i32* %x.addr, metadata !13, metadata !DIExpression()), !dbg !23
  store i32 2, i32* %x.addr, align 4, !dbg !23
  call void @escape(i32* %x.addr), !dbg !24
  ret void, !dbg !25
}

; ASM-LABEL: f: # @f
; ASM: #DEBUG_VALUE: f:x <- [DW_OP_plus_uconst [[OFF_X:[0-9]+]]] [$rsp+0]
; ASM: movl    %ecx, [[OFF_X]](%rsp)
; ASM: callq   escape
; ASM: #DEBUG_VALUE: f:x <- 1
; ASM: movl    $1, global(%rip)
; ASM: #DEBUG_VALUE: f:x <- [DW_OP_plus_uconst [[OFF_X]]] [$rsp+0]
; ASM: movl    $2, [[OFF_X]](%rsp)
; ASM: callq   escape
; ASM: retq

; DWARF:      DW_TAG_formal_parameter
; DWARF-NEXT:   DW_AT_location        (0x00000000
; DWARF-NEXT:      {{[^:]*}}: DW_OP_breg7 RSP+{{[0-9]+}}
; DWARF-NEXT:      {{[^:]*}}: DW_OP_consts +1, DW_OP_stack_value
; DWARF-NEXT:      {{[^:]*}}: DW_OP_breg7 RSP+{{[0-9]+}})
; DWARF-NEXT:   DW_AT_name    ("x")

attributes #0 = { nounwind uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dse.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 3, column: 12, scope: !8)
!19 = !DILocation(line: 4, column: 3, scope: !8)
!20 = !DILocation(line: 5, column: 5, scope: !8)
!21 = !DILocation(line: 6, column: 12, scope: !8)
!22 = !DILocation(line: 6, column: 10, scope: !8)
!23 = !DILocation(line: 7, column: 5, scope: !8)
!24 = !DILocation(line: 8, column: 3, scope: !8)
!25 = !DILocation(line: 9, column: 1, scope: !8)
