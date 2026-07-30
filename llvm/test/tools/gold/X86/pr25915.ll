; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2
; RUN: llvm-dis %t2 -o - | FileCheck %s
; CHECK-NOT: subprograms

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256170) (llvm/trunk 256171)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "pr25915.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.8.0 (trunk 256170) (llvm/trunk 256171)"}
