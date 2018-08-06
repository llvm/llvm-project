; RUN: llvm-as < %s -o %t.bc 2>&1 >/dev/null | FileCheck -check-prefix=WARN %s
; RUN: llvm-dis < %t.bc | FileCheck %s
; RUN: verify-uselistorder < %t.bc

define i32 @main() !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 (trunk 195495) (llvm/trunk 195495:195504M)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "../llvm/tools/clang/test/CodeGen/debug-info-version.c", directory: "/Users/manmanren/llvm_gmail/release")
!2 = !{i32 0}
!4 = distinct !DISubprogram(name: "main", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "../llvm/tools/clang/test/CodeGen/debug-info-version.c", directory: "/Users/manmanren/llvm_gmail/release")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !DILocation(line: 4, scope: !4)

; WARN: warning: ignoring debug info with an invalid version (0)
; CHECK-NOT: !dbg
; CHECK-NOT: !llvm.dbg.cu
