;; -mllvm options that are part of the target options reach libLTO's code
;; generators, even when they are given to libLTO after the generator is
;; created.
; RUN: rm -rf %t && mkdir %t
; RUN: llvm-as %s -o %t/a.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/default.dylib %t/a.o -object_path_lto %t/default.o -exported_symbol _zero -exported_symbol _f -lSystem
; RUN: llvm-nm %t/default.o | FileCheck %s --check-prefix=DEFAULT --implicit-check-not=Lline_table_start0
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/opts.dylib %t/a.o -object_path_lto %t/opts.o -exported_symbol _zero -exported_symbol _f -lSystem -mllvm -nozero-initialized-in-bss -mllvm -save-temp-labels
; RUN: llvm-nm %t/opts.o | FileCheck %s --check-prefix=OPTS

; RUN: opt -module-summary %s -o %t/thin.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/thin.dylib %t/thin.o -object_path_lto %t/thin -exported_symbol _zero -exported_symbol _f -lSystem -mllvm -nozero-initialized-in-bss -mllvm -save-temp-labels
; RUN: llvm-nm %t/thin/0.x86_64.thinlto.o | FileCheck %s --check-prefix=OPTS

;; Zero-initialized globals are in the bss section by default, but not with
;; -nozero-initialized-in-bss (a TargetOptions flag of the code generator).
;; -save-temp-labels (a MCTargetOptions flag) keeps the temporary labels.
; DEFAULT: S _zero
; OPTS-DAG: D _zero
; OPTS-DAG: s Lline_table_start0

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@zero = global i32 0, align 4

define void @f() !dbg !6 {
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "t", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !DILocation(line: 1, column: 1, scope: !6)
