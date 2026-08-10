; RUN: opt -S -passes=consthoist < %s | FileCheck %s
; ModuleID = 'test-hoist-debug.cpp'
source_filename = "test-hoist-debug.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define i32 @_Z3foov() !dbg !7 {
; CHECK: bitcast
; CHECK: !dbg !11
; CHECK: inttoptr
  %a0 = inttoptr i64 4646526064 to ptr, !dbg !11
  %v0 = load i32, ptr %a0, align 16, !dbg !11

  %a1 = inttoptr i64 4646526080 to ptr
  %v1 = load i32, ptr %a1, align 16, !dbg !11

  %a2 = inttoptr i64 4646526096 to ptr
  %v2 = load i32, ptr %a2, align 16, !dbg !11

  %r0 = add i32 %v0, %v1
  %r1 = add i32 %r0, %v2
  ret i32 %r1
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 313291)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test-hoist-debug.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 313291)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 3, scope: !7)
