
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj -generate-arange-section < %s | llvm-dwarfdump -debug-aranges - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj -generate-arange-section < %s | llvm-readobj --relocations - | FileCheck --check-prefix=OBJ %s

; extern int i;
; template<int *x>
; struct foo {
; };
;
; foo<&i> f;

; Check that we only have one arange in this compilation unit (it will be for 'f'), and not an extra one (for 'i' - since it isn't actually defined in this CU)

; CHECK: Address Range Header
; CHECK-NEXT: [0x
; CHECK-NOT: [0x

; Check that we have a relocation back to the debug_info section from the debug_aranges section
; OBJ: debug_aranges
; OBJ-NEXT: R_X86_64_32 .debug_info 0x0

source_filename = "test/DebugInfo/X86/arange.ll"

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1, !dbg !0
@i = external global i32

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 6, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "simple.cpp", directory: "/tmp/dbginfo")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo<&i>", file: !2, line: 3, size: 8, align: 8, elements: !4, templateParams: !5, identifier: "_ZTS3fooIXadL_Z1iEEE")
!4 = !{}
!5 = !{!6}
!6 = !DITemplateValueParameter(name: "x", type: !7, value: ptr @i)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !10, globals: !11, imports: !4)
!10 = !{!3}
!11 = !{!0}
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 1, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.5 "}

