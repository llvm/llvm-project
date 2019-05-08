; Verify the emission of accelerator tables for various targets for the DWARF<=4 case

; Darwin has the apple tables unless we specifically tune for gdb
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=APPLE %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj -debugger-tune=gdb < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=PUB %s

; Linux does has debug_names tables only if we explicitly tune for lldb
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=PUB %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -debugger-tune=lldb < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=DEBUG_NAMES %s

; No accelerator tables if type units are enabled, as DWARF v4 type units are
; not compatible with accelerator tables.
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -generate-type-units -debugger-tune=lldb < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=NONE %s

; Debug types are ignored for non-ELF targets which means it shouldn't affect
; accelerator table generation.
; RUN: llc -mtriple=x86_64-apple-darwin12 -generate-type-units -filetype=obj < %s \
; RUN:   | llvm-readobj --sections - | FileCheck --check-prefix=APPLE %s

; APPLE-NOT: debug_names
; APPLE-NOT: debug{{.*}}pub
; APPLE: apple_names
; APPLE-NOT: debug_names
; APPLE-NOT: debug{{.*}}pub

; PUB-NOT: apple_names
; PUB-NOT: debug_names
; PUB: pubnames
; PUB-NOT: apple_names
; PUB-NOT: debug_names

; NONE-NOT: apple_names
; NONE-NOT: debug_names

; DEBUG_NAMES-NOT: apple_names
; DEBUG_NAMES-NOT: pubnames
; DEBUG_NAMES: debug_names
; DEBUG_NAMES-NOT: apple_names
; DEBUG_NAMES-NOT: pubnames

@var = thread_local global i32 0, align 4, !dbg !0

; Function Attrs: norecurse nounwind readnone uwtable
define void @_Z3funv() local_unnamed_addr #0 !dbg !11 {
  ret void, !dbg !14
}

; Function Attrs: norecurse uwtable
define weak_odr hidden i32* @_ZTW3var() local_unnamed_addr #1 {
  ret i32* @var
}

attributes #0 = { norecurse nounwind readnone uwtable }
attributes #1 = { norecurse uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 322268) (llvm/trunk 322267)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "debugger-tune.cpp", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0 (trunk 322268) (llvm/trunk 322267)"}
!11 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 2, type: !12, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 2, column: 13, scope: !11)

