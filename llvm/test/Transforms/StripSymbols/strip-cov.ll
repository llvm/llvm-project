; RUN: opt -S %s -passes=strip -o - | FileCheck %s

; CHECK-NOT: !llvm.dbg.cu
; CHECK-NOT: !llvm.gcov

; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.gcov = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 284352) (llvm/trunk 284353)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/dev/null", directory: "/home/davide/work/llvm/build/bin")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"/scratch/patatino/build/bin/null.gcno", !"/scratch/patatino/build/bin/null.gcda", !0}
