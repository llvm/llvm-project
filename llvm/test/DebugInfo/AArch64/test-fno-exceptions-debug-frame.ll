; This test is added to provide test coverage for the 
; .cfi_sections .debug_frame intrinsic. It aims to make sure that for a AArch64
; output compiled with -fno-exceptions, a .cfi_sections .debug_frame is emitted.

; RUN: llc --filetype=asm %s -o - | FileCheck %s
; CHECK: .cfi_sections .debug_frame

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx13.0.0"

; Function Attrs: noinline norecurse nounwind optnone ssp
define i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 1, !dbg !14
}

attributes #0 = { noinline norecurse nounwind optnone ssp }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.dbg.cu = !{!6}
!llvm.ident = !{!8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 0]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 1}
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/Applications/Xcode.app", sdk: "MacOSX.sdk")
!7 = !DIFile(filename: "test.cpp", directory: "/Users/shubham/Development")
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocation(line: 2, column: 5, scope: !9)
