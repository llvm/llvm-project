; REQUIRES: x86-registered-target
; RUN: opt < %s -passes='pseudo-probe,always-inline' -S -o %t
; RUN: FileCheck %s --check-prefix=IR < %t
; RUN: llc %t -mtriple=x86_64-unknown-linux-gnu -stop-after=pseudo-probe-inserter -o - | FileCheck %s --check-prefix=MIR
; RUN: opt < %t -passes='cgscc(inline)' -S -o %t.inlined
; RUN: llc %t.inlined -mtriple=x86_64-unknown-linux-gnu -filetype=asm -o - | FileCheck %s --check-prefix=ASM

; IR-LABEL: @caller(

; This call came from the callee without debug metadata. It keeps the caller's
; source location but must not inherit the caller's callsite probe.
; IR: call void @middle(){{.*}}!dbg ![[MIDDLE_CALL_LOC:[0-9]+]]

; This instruction did not have debug metadata in the callee but gets a debug
; location after inlining.
; IR: store volatile i32 1, {{.*}}, !dbg ![[#]]

; This pseudo probe came from callee without a !dbg metadata.
; IR-NOT: call void @llvm.pseudoprobe({{.*}}), !dbg ![[#]]
; IR: call void @llvm.pseudoprobe({{.*}})

; IR-DAG: ![[MIDDLE_CALL_LOC]] = !DILocation(line: 4, scope: ![[MIDDLE_CALL_SCOPE:[0-9]+]])
; IR-DAG: ![[MIDDLE_CALL_SCOPE]] = !DILexicalBlockFile({{.*}}discriminator: 3)

; MIR-LABEL: name: caller
; MIR-NOT: PSEUDO_PROBE {{.*}}, 2, 2

; After middle and inner are inlined, the caller-to-middle edge has no probe
; identity. Keep that edge as reserved probe zero so the decoder can reset the
; unavailable context without promoting the inlinees to top-level probe groups.
; ASM-LABEL: caller:
; ASM: .pseudoprobe 13491010695890359370 1 0 0 @ 16677772384402303968:0 caller
; ASM: .pseudoprobe 4738244524459464682 1 0 0 @ 16677772384402303968:0 @ 13491010695890359370:2 caller

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4

; Function Attrs: inlinehint nounwind
define internal void @inner() #0 !dbg !7 {
entry:
  store volatile i32 3, ptr @c, align 4, !dbg !14
  ret void, !dbg !15
}

; Function Attrs: inlinehint nounwind
define internal void @middle() #0 !dbg !16 {
entry:
  call void @inner(), !dbg !19
  store volatile i32 2, ptr @b, align 4, !dbg !20
  ret void, !dbg !21
}

; Function Attrs: alwaysinline nounwind
define internal void @callee() #1 {
entry:
  call void @middle()
  store volatile i32 1, ptr @a, align 4
  ret void
}

define void @caller() !dbg !4 {
entry:
  call void @callee(), !dbg !12
  ret void, !dbg !12
}

attributes #0 = { inlinehint nounwind }
attributes #1 = { alwaysinline nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (210174)", isOptimized: true, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/code/llvm/build0")
!2 = !{}
!4 = distinct !DISubprogram(name: "caller", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test.c", directory: "/code/llvm/build0")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "inner", linkageName: "inner", line: 20, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 20, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 (210174)"}
!12 = !DILocation(line: 4, scope: !13)
!13 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 6)
!14 = !DILocation(line: 21, scope: !7)
!15 = !DILocation(line: 22, scope: !7)
!16 = distinct !DISubprogram(name: "middle", linkageName: "middle", line: 10, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 10, file: !1, scope: !5, type: !6, retainedNodes: !2)
!19 = !DILocation(line: 11, scope: !16)
!20 = !DILocation(line: 12, scope: !16)
!21 = !DILocation(line: 13, scope: !16)
