; RUN: rm -rf %t && mkdir -p %t
; RUN: llc --filetype=obj --mccas-verify --cas-backend --cas=%t/cas %s -o %t/watchos-cas.o 
; RUN: llvm-dwarfdump %t/watchos-cas.o | FileCheck %s
; CHECK: .debug_info contents:
; CHECK-NEXT: 0x{{[0-9a-f]+}}: Compile Unit: length = 0x{{[0-9a-f]+}}, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x04
; REQUIRES: arm-registered-target

; ModuleID = '/tmp/a.cpp'
source_filename = "/tmp/a.cpp"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

@a = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0git (git@github.com:apple/llvm-project.git 6fa917db4566cab002b856a66e8ebba16f0e20a4)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "/tmp/a.cpp", directory: "/Users/shubham/Development/llvm-project-cas/llvm-project/build_ninja")
!4 = !{!0}
!5 = !DIFile(filename: "/tmp/a.cpp", directory: "")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"min_enum_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 19.0.0git (git@github.com:apple/llvm-project.git 6fa917db4566cab002b856a66e8ebba16f0e20a4)"}