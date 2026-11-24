; Check that cross-TU ODR DICompositeType size mismatches do not collapse into a
; single ODR-uniqued node and trip the IR verifier on DW_OP_LLVM_fragment bounds
; (including under -disable-auto-upgrade-debug-info).

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary -module-hash %t/small.ll -o %t/small.bc
; RUN: opt -module-summary -module-hash %t/large.ll -o %t/large.bc
; RUN: opt -module-summary -module-hash %t/main.ll -o %t/main.bc
; RUN: opt -module-summary -module-hash %t/dest_small.ll -o %t/dest_small.bc

; Case 1: DestModule has no definition of ProdEval and imports both small.bc
; (size: 192) and large.bc (size: 320, with a fragment at [192, 256)).
; RUN: llvm-lto2 run -disable-auto-upgrade-debug-info -save-temps \
; RUN:   %t/main.bc %t/small.bc %t/large.bc -o %t/out1 \
; RUN:   -r %t/main.bc,run,plx \
; RUN:   -r %t/main.bc,eval_small,l \
; RUN:   -r %t/main.bc,eval_large,l \
; RUN:   -r %t/small.bc,eval_small,pl \
; RUN:   -r %t/large.bc,eval_large,pl
; RUN: llvm-dis < %t/out1.1.3.import.bc | FileCheck %s --check-prefix=IMPORT

; Case 1b: Same as Case 1 with lazy metadata loading enabled
; (--bitcode-mdindex-threshold=0), verifying SizeInBits is resolved before
; ODR uniquing rather than compared as an unresolved placeholder.
; RUN: opt --bitcode-mdindex-threshold=0 -module-summary -module-hash %t/small.ll -o %t/small.lazy.bc
; RUN: opt --bitcode-mdindex-threshold=0 -module-summary -module-hash %t/large.ll -o %t/large.lazy.bc
; RUN: opt --bitcode-mdindex-threshold=0 -module-summary -module-hash %t/main.ll -o %t/main.lazy.bc
; RUN: llvm-lto2 run -disable-auto-upgrade-debug-info -save-temps \
; RUN:   %t/main.lazy.bc %t/small.lazy.bc %t/large.lazy.bc -o %t/out1_lazy \
; RUN:   -r %t/main.lazy.bc,run,plx \
; RUN:   -r %t/main.lazy.bc,eval_small,l \
; RUN:   -r %t/main.lazy.bc,eval_large,l \
; RUN:   -r %t/small.lazy.bc,eval_small,pl \
; RUN:   -r %t/large.lazy.bc,eval_large,pl
; RUN: llvm-dis < %t/out1_lazy.1.3.import.bc | FileCheck %s --check-prefix=IMPORT

; Case 2: DestModule itself defines ProdEval with the smaller size (192) and
; imports eval_large (size: 320, with a fragment at [192, 256)).
; RUN: llvm-lto2 run -disable-auto-upgrade-debug-info -save-temps \
; RUN:   %t/dest_small.bc %t/large.bc -o %t/out2 \
; RUN:   -r %t/dest_small.bc,run,plx \
; RUN:   -r %t/dest_small.bc,eval_large,l \
; RUN:   -r %t/large.bc,eval_large,pl
; RUN: llvm-dis < %t/out2.1.3.import.bc | FileCheck %s --check-prefix=DESTSMALL

; Case 3: Full LTO module linking (llvm-link) with mismatched ODR sizes keeps
; both DICompositeTypes distinct and passes verification.
; RUN: llvm-link -disable-auto-upgrade-debug-info -S %t/small.ll %t/large.ll \
; RUN:   | opt -passes=verify -S | FileCheck %s --check-prefix=FULLLTO

; IMPORT-DAG: !DILocalVariable(name: "v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, type: ![[LARGE:[0-9]+]])
; IMPORT-DAG: ![[LARGE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", {{.*}}size: 320, flags: DIFlagFwdDecl, identifier: "_ZTS8ProdEval")

; DESTSMALL-DAG: !DILocalVariable(name: "v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, type: ![[LARGE2:[0-9]+]])
; DESTSMALL-DAG: ![[LARGE2]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", {{.*}}size: 320, flags: DIFlagFwdDecl, identifier: "_ZTS8ProdEval")
; DESTSMALL-DAG: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", {{.*}}size: 192, elements: !{{[0-9]+}}, identifier: "_ZTS8ProdEval")

; FULLLTO-DAG: !DILocalVariable(name: "v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: 2, type: ![[LARGE3:[0-9]+]])
; FULLLTO-DAG: ![[LARGE3]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", {{.*}}size: 320, elements: !{{[0-9]+}}, identifier: "_ZTS8ProdEval")
; FULLLTO-DAG: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", {{.*}}size: 192, elements: !{{[0-9]+}}, identifier: "_ZTS8ProdEval")

;--- small.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @eval_small() !dbg !6 {
  ret i64 1, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !DIFile(filename: "small.cc", directory: "/repro")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", file: !1, line: 1, size: 192, elements: !9, identifier: "_ZTS8ProdEval")
!6 = distinct !DISubprogram(name: "eval_small", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocation(line: 2, column: 3, scope: !6)

;--- large.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @eval_large(i64 %x) !dbg !6 {
  #dbg_value(i64 %x, !10, !DIExpression(DW_OP_LLVM_fragment, 192, 64), !11)
  ret i64 %x, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !DIFile(filename: "large.cc", directory: "/repro")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", file: !1, line: 1, size: 320, elements: !9, identifier: "_ZTS8ProdEval")
!6 = distinct !DISubprogram(name: "eval_large", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocalVariable(name: "v", scope: !6, file: !1, line: 2, type: !5)
!11 = !DILocation(line: 2, column: 3, scope: !6)
!12 = !{!10}

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @eval_small()
declare i64 @eval_large(i64)

define i64 @run(i64 %x) !dbg !6 {
  %a = call i64 @eval_small(), !dbg !10
  %b = call i64 @eval_large(i64 %x), !dbg !10
  %c = add i64 %a, %b, !dbg !10
  ret i64 %c, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.cc", directory: "/repro")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "run", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!10 = !DILocation(line: 2, column: 3, scope: !6)

;--- dest_small.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @eval_large(i64)

define i64 @run(i64 %x) !dbg !6 {
  %b = call i64 @eval_large(i64 %x), !dbg !10
  ret i64 %b, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !DIFile(filename: "dest_small.cc", directory: "/repro")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ProdEval", file: !1, line: 1, size: 192, elements: !9, identifier: "_ZTS8ProdEval")
!6 = distinct !DISubprogram(name: "run", scope: !1, file: !1, line: 2, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{}
!10 = !DILocation(line: 2, column: 3, scope: !6)
