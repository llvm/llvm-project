; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info --name=static_local --show-parents - | FileCheck %s --implicit-check-not=DW_TAG

;; Derived from a CUDA program like:
;;
;;   __forceinline__ __device__ void foo(int *p) {
;;     {  // lexical block -- scope of the static local
;;       static __shared__ int static_local;
;;       *p = static_local;
;;     }
;;   }
;;
;;   __global__ void bar(int *q) {
;;     {
;;       foo(q);  // foo is inlined here
;;     }
;;   }
;;
;; __forceinline__ causes foo to be inlined, so an abstract subprogram
;; origin is created for it.  The `static __shared__` variable compiles to an
;; addrspace(3) global (not a local variable) whose DI scope is the
;; DILexicalBlock inside foo.  When the abstract scope tree is built,
;; skipLexicalScope() may elide that lexical block because it has no local
;; variables -- but the global still needs the block as its context DIE.

;; The abstract "foo" subprogram must contain a DW_TAG_lexical_block holding
;; the "static_local" variable.

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name ("foo")
; CHECK:       DW_AT_inline (DW_INL_inlined)
; CHECK:         DW_TAG_lexical_block
; CHECK:           DW_TAG_variable
; CHECK:             DW_AT_name ("static_local")

@static_local = internal addrspace(3) global i32 0, align 4, !dbg !0

define void @_Z3barPi(ptr %q) !dbg !14 {
entry:
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "static_local", linkageName: "_ZZ3fooPiE12static_local", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DILexicalBlock(scope: !4, file: !3, line: 2, column: 1)
!3 = !DIFile(filename: "test.cu", directory: "/")
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPi", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !10, retainedNodes: !12)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64, dwarfAddressSpace: 12)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "lgenfe: EDG 6.8", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !11)
!11 = !{!0}
!12 = !{}
!13 = !{i32 1, !"Debug Info Version", i32 3}
!14 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barPi", scope: !3, file: !3, line: 9, type: !5, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!15 = distinct !DILocation(line: 11, column: 3, scope: !16)
!16 = distinct !DILexicalBlock(scope: !14, file: !3, line: 10, column: 1)
!17 = !DILocation(line: 4, column: 3, scope: !2, inlinedAt: !15)
