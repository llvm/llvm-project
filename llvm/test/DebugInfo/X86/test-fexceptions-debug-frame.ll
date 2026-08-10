; This test is added to provide test coverage for the 
; .cfi_sections .debug_frame intrinsic. It aims to make sure that for an X86_64
; output compiled with -fexceptions, no .cfi_sections .debug_frame is emitted.

; RUN: llc --filetype=asm %s -o - | FileCheck %s
; CHECK-NOT: .cfi_sections .debug_frame
; CHECK: .cfi_startproc

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx13.0.0"

; Function Attrs: noinline norecurse nounwind optnone ssp uwtable
define i32 @main() #0 !dbg !10 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 1, !dbg !15
}

attributes #0 = { noinline norecurse nounwind optnone ssp uwtable  }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
!llvm.dbg.cu = !{!7}
!llvm.ident = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 0]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !8, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, sysroot: "/Applications/Xcode.app", sdk: "MacOSX.sdk")
!8 = !DIFile(filename: "test.cpp", directory: "/Users/shubham/Development")
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocation(line: 2, column: 5, scope: !10)
