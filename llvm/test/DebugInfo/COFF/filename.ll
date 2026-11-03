; RUN: sed -e 's|_FILENAME_|/abs/path/to/test.cpp|;s|_DIR_|/ignored/directory|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-POSIX
;
; RUN: sed -e 's|_FILENAME_|test.cpp|;s|_DIR_|/abs/path/to|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-POSIX
;
; RUN: sed -e 's|_FILENAME_|./to/to2/../test.cpp|;s|_DIR_|/abs/path/subdir/../|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-POSIX-DOTS
;
; Windows - paths are canonicalized
; RUN: sed -e 's|_FILENAME_|test.cpp|;s|_DIR_|rel/path/to/|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-REL
;
; RUN: sed -e 's|_FILENAME_|test.cpp|;s|_DIR_|C:/abs/path/to|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-DRV
;
; RUN: sed -e 's|_FILENAME_|C:/abs/path/to/test.cpp|;s|_DIR_|X:/ignored/directory/|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-DRV
;
; RUN: sed -e 's|_FILENAME_|./skipped_subdir2/../test.cpp|;s|_DIR_|C:/skipped_subdir1/../abs/path/to/|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-DRV
;
; RUN: sed -e 's|_FILENAME_|./skipped_subdir2//../test.cpp|;s|_DIR_|C:/skipped_subdir1//../abs/path/to/|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-DRV
;
; RUN: sed -e 's|_FILENAME_|./../test.cpp|;s|_DIR_|C:/abs/path/to/skipped_subdir|' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-DRV
;
; RUN: sed -e 's|_FILENAME_|./../test.cpp|' -e 's|_DIR_|<BSLASH><BSLASH>servername/path/to/skipped_subdir/|' -e 's|<BSLASH>|\\\\|g' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-UNC
;
; RUN: sed -e 's|_FILENAME_|<BSLASH><BSLASH>servername/path/to/skipped_subdir/../test.cpp|' -e 's|_DIR_|X:/ignored/directory|' -e 's|<BSLASH>|\\\\|g' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-UNC
;
; RUN: sed -e 's|_FILENAME_|./../test.cpp|' -e 's|_DIR_|<BSLASH><BSLASH>\?<BSLASH>C:/long/path/to/skipped_subdir/|' -e 's|<BSLASH>|\\\\|g' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-EXT
;
; RUN: sed -e 's|_FILENAME_|<BSLASH><BSLASH>\?<BSLASH>C:/long/path/to/skipped_subdir/../test.cpp|' -e 's|_DIR_|X:/ignored/directory|' -e 's|<BSLASH>|\\\\|g' %s > %t
; RUN: llc -filetype=obj -o - %t | llvm-readobj --codeview - | FileCheck %s -check-prefixes CHECK,CHECK-WIN-EXT

; C++ source to regenerate:
; # 1 "_FILENAME_"
; void f() {}
;
; $clang++ --target=x86_64-pc-windows-msvc -g -O0 -fdebug-compilation-dir=_DIR_

; CHECK: FunctionLineTable [
; CHECK-NEXT:   LinkageName: ?f@@YAXXZ
; CHECK-NEXT:   Flags:
; CHECK-NEXT:   CodeSize:
; CHECK-NEXT:   FilenameSegment [
;
; CHECK-POSIX-NEXT:     Filename: /abs/path/to/test.cpp (0x0)
; CHECK-POSIX-DOTS-NEXT:     Filename: /abs/path/subdir/.././to/to2/../test.cpp (0x0)
;
; CHECK-WIN-REL-NEXT:     Filename: rel\path\to\test.cpp (0x0)
; CHECK-WIN-DRV-NEXT:     Filename: C:\abs\path\to\test.cpp (0x0)
; CHECK-WIN-UNC-NEXT:     Filename: \\servername\path\to\test.cpp (0x0)
; CHECK-WIN-EXT-NEXT:     Filename: \\?\C:\long\path\to\test.cpp (0x0)

; ModuleID = '/app/example.cpp'
source_filename = "/app/example.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.33.0"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @"?f@@YAXXZ"() #0 !dbg !9 {
entry:
  ret void, !dbg !13
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.1.0 (https://github.com/llvm/llvm-project.git 4434dabb69916856b824f68a64b029c67175e532)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.cpp", directory: "_DIR_")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 22.1.0 (https://github.com/llvm/llvm-project.git)"}
!9 = distinct !DISubprogram(name: "f", linkageName: "?f@@YAXXZ", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DIFile(filename: "_FILENAME_", directory: "_DIR_")
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 1, scope: !9)
