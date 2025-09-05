; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/import-metadata.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t-out \
; RUN:    -r=%t1.bc,main,plx \
; RUN:    -r=%t1.bc,foo,l \
; RUN:    -r=%t2.bc,foo,pl
; RUN: llvm-dis %t-out.1.3.import.bc -o - | FileCheck %s

;; Check the imported DICompileUnit doesn't have the enums operand.
;; Also check the imported md metadata that shares a node with the 
;; enums operand originally is not null.

; CHECK: !llvm.dbg.cu = !{![[#CU1:]], ![[#CU2:]]}
;; Note that MD1 comes from the current module. MD2 is from the imported module. 
;; We are checking that MD2 is unified with MD1 and the imported MD2 doesn't
;; end up having a null operand.
; CHECK: !llvm.md = !{![[#MD1:]]}
; CHECK: ![[#MD3:]] = !{}
; CHECK: ![[#CU2]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: ![[#FILE2:]], isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
; CHECK: ![[#MD1]] = !{![[#MD3]]}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

declare i32 @foo(i32 %goo)

define i32 @main() {
  call i32 @foo(i32 0)
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.md = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, enums: !4)
!1 = !DIFile(filename: "main.cpp", directory: "tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = !{!4}
