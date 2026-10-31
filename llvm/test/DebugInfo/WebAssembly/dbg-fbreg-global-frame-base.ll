;; A function whose stack pointer is never referenced has no virtual frame base
;; local, so its frame base is the __stack_pointer global. A stack local must
;; still be described relative to that frame base with DW_OP_fbreg: the stack
;; pointer is not a DWARF register, but a frame-relative location needs no
;; register of its own. Otherwise the base is lost and the offset underflows
;; the DWARF stack when the location is evaluated.

; RUN: llc < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_frame_base ({{.*}}0x3 0x0, DW_OP_stack_value)
; CHECK:        DW_TAG_variable
; CHECK-NEXT:     DW_AT_location (DW_OP_fbreg +12)
; CHECK-NEXT:     DW_AT_name ("t1")

target triple = "wasm32-unknown-unknown"

define hidden i32 @main() #0 !dbg !6 {
  %1 = alloca i32, align 4
    #dbg_declare(ptr %1, !11, !DIExpression(), !13)
  ret i32 0, !dbg !13
}

attributes #0 = { noinline nounwind optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "t1", scope: !6, file: !1, line: 1, type: !9)
!13 = !DILocation(line: 1, column: 1, scope: !6)
