; struct A {
;   void foo();
; };
;  
; void (A::*p)() = &A::foo;
;
; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump -debug-info - | FileCheck %s
; Check that the member function pointer is emitted without a DW_AT_size attribute.
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NOT: DW_AT_{{.*}}size
; CHECK: DW_TAG
;
; ModuleID = 'memberfnptr.cpp'
source_filename = "test/DebugInfo/X86/memberfnptr.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.A = type { i8 }

@p = global { i64, i64 } { i64 ptrtoint (ptr @_ZN1A3fooEv to i64), i64 0 }, align 8, !dbg !0

declare void @_ZN1A3fooEv(ptr)

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "p", scope: null, file: !2, line: 5, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "memberfnptr.cpp", directory: "")
!3 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !4, size: 64, extraData: !7)
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !2, line: 1, size: 8, align: 8, elements: !8, identifier: "_ZTS1A")
!8 = !{!9}
!9 = !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", scope: !7, file: !2, line: 2, type: !4, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.6.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !11, retainedTypes: !12, globals: !13, imports: !11)
!11 = !{}
!12 = !{!7}
!13 = !{!0}
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"PIC Level", i32 2}
!17 = !{!"clang version 3.6.0 "}

