; RUN: %llc_dwarf -mtriple=x86_64-linux -O0 -filetype=obj < %s              \
; RUN:  | llvm-dwarfdump --show-children --name=f -                         \
; RUN:  | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; Ensure that if a local type belongs to a function that is neither emitted nor
; inlined, its DIE is emitted inside an abstract subprogram, and that subprogram
; is not attributeless.

; Compiled with -O2 and reduced from the source below.
; Note that although f() is not emitted as a separate or inlined function in
; LLVM IR (compiled with high optimization level), its local type B leaks out
; of the scope of subprogram f() when it is used as a template parameter in
; use<B>(). Therefore, an abstract subprogram for f() must be emitted in DWARF
; to serve as a container for B.
;
; template <typename T> void use() {}
; inline auto f() {
;   struct B {};
;   return &use<B>;
; }
; void (*p)() = f();

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("f")
; CHECK:   DW_AT_inline (DW_INL_inlined)
; CHECK:   DW_TAG_structure_type
; CHECK:     DW_AT_name ("B")
; CHECK:     DW_AT_declaration (true)
; CHECK:   NULL

@p = dso_local local_unnamed_addr global ptr @_Z3useIZ1fvE1BEvv, align 8, !dbg !0

define dso_local void @_Z3useIZ1fvE1BEvv() !dbg !10 {
entry:
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 6, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "use<B>", linkageName: "_Z3useIZ1fvE1BEvv", scope: !3, file: !3, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, templateParams: !11, keyInstructions: true)
!11 = !{!12}
!12 = !DITemplateTypeParameter(name: "T", type: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "B", scope: !14, file: !3, line: 3, size: 8, flags: DIFlagFwdDecl, identifier: "_ZTSZ1fvE1B")
!14 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !3, file: !3, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17, keyInstructions: true)
!15 = !DISubroutineType(types: !16)
!16 = !{!5}
!17 = !{!13}
!18 = !DILocation(line: 1, column: 35, scope: !10, atomGroup: 1, atomRank: 1)
