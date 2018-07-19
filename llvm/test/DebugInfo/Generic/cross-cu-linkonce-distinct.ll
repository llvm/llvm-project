; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck %s

; Testing that two distinct (distinct by writing them in separate files, while
; still fulfilling C++'s ODR by having identical token sequences) functions,
; linked under LTO, get plausible debug info (and don't crash).

; Built from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc

; This change is intended to tickle a case where the subprogram MDNode
; associated with the llvm::Function will differ from the subprogram
; referenced by the DbgLocs in the function.

; $ sed -ie "s/!12, !0/!0, !12/" ab.ll
; $ cat a.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*x)(int) = &func;
; $ cat b.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*y)(int) = &func;

; The DISubprogram should show up in compile unit a.
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}}"b.cpp"
; CHECK-NOT: DW_TAG_subprogram

; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}}"a.cpp"
; CHECK:     DW_AT_name {{.*}} "func"

source_filename = "test/DebugInfo/Generic/cross-cu-linkonce-distinct.ll"

@x = global i32 (i32)* @_Z4funci, align 8, !dbg !0
@y = global i32 (i32)* @_Z4funci, align 8, !dbg !7

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 !dbg !18 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !19, metadata !20), !dbg !21
  %2 = load i32, i32* %1, align 4, !dbg !22
  %3 = mul nsw i32 %2, 2, !dbg !22
  ret i32 %3, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!10, !13}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17, !17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !6}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "y", scope: null, file: !9, line: 4, type: !3, isLocal: false, isDefinition: true)
!9 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !9, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !12, imports: !11)
!11 = !{}
!12 = !{!7}
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !11, globals: !14, imports: !11)
!14 = !{!0}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 1, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.5.0 "}
!18 = distinct !DISubprogram(name: "func", linkageName: "_Z4funci", scope: !2, file: !2, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !13, variables: !11)
!19 = !DILocalVariable(name: "i", arg: 1, scope: !18, file: !2, line: 1, type: !6)
!20 = !DIExpression()
!21 = !DILocation(line: 1, scope: !18)
!22 = !DILocation(line: 2, scope: !18)

