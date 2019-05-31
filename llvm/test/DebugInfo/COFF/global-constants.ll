; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; const float TestConst1 = 3.14;
; struct S {
;   static const int TestConst2 = -10;
; }
; enum TestEnum : int {
;    ENUM_A = 2147000000,
;    ENUM_B = -2147000000,
; };
; void useConst(int);
; void foo() {
;   useConst(TestConst1);
;   useConst(S::TestConst2);
;   useConst(ENUM_B);
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM-LABEL:  .long 241                     # Symbol subsection for globals
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4099                    # Type
; ASM-NEXT:   .byte 0x04, 0x80, 0xc3, 0xf5  # Value
; ASM-NEXT:   .byte 0x48, 0x40
; ASM-NEXT:   .asciz "TestConst1"           # Name
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4100                    # Type
; ASM-NEXT:   .byte 0x61, 0x00              # Value
; ASM-NEXT:   .asciz "S::TestConst2"        # Name
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4102                    # Type
; ASM-NEXT:   .byte 0x0a, 0x80, 0x40, 0x61  # Value
; ASM-NEXT:   .byte 0x07, 0x80, 0xff, 0xff
; ASM-NEXT:   .byte 0xff, 0xff
; ASM-NEXT:   .asciz "ENUM_B"               # Name

; OBJ:        CodeViewDebugInfo [
; OBJ:          Section: .debug$S
; OBJ:          Magic: 0x4
; OBJ:          Subsection [
; OBJ:            SubSectionType: Symbols (0xF1)
; OBJ:            ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const float (0x1003)
; OBJ-NEXT:         Value: 1078523331
; OBJ-NEXT:         Name: TestConst1
; OBJ-NEXT:       }
; OBJ-NEXT:       ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const char (0x1004)
; OBJ-NEXT:         Value: 97
; OBJ-NEXT:         Name: S::TestConst2
; OBJ-NEXT:       }
; OBJ-NEXT:       ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: TestEnum (0x1006)
; OBJ-NEXT:         Value: 18446744071562551616
; OBJ-NEXT:         Name: ENUM_B
; OBJ-NEXT:       }


; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

; Function Attrs: noinline nounwind optnone
define dso_local void @_Z3foov() #0 !dbg !28 {
entry:
  call void @_Z8useConsti(i32 3), !dbg !32
  call void @_Z8useConsti(i32 97), !dbg !33
  call void @_Z8useConsti(i32 -2147000000), !dbg !34
  ret void, !dbg !35
}

declare dso_local void @_Z8useConsti(i32) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25, !26}
!llvm.ident = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git dee1891507401f396290b5d9cb5717d6b0755337)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !9, globals: !15, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "6d700c7d582557a012214ac1f1f8721b")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "TestEnum", file: !4, line: 5, baseType: !5, size: 32, elements: !6, identifier: "_ZTS8TestEnum")
!4 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "6d700c7d582557a012214ac1f1f8721b")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{!7, !8}
!7 = !DIEnumerator(name: "ENUM_A", value: 2147000000)
!8 = !DIEnumerator(name: "ENUM_B", value: -2147000000)
!9 = !{!10}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !11, identifier: "_ZTS1S")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "TestConst2", scope: !10, file: !4, line: 3, baseType: !13, flags: DIFlagStaticMember, extraData: i8 97)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !{!16, !20, !22}
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression(DW_OP_constu, 1078523331, DW_OP_stack_value))
!17 = distinct !DIGlobalVariable(name: "TestConst1", scope: !0, file: !4, line: 1, type: !18, isLocal: true, isDefinition: true)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !19)
!19 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression(DW_OP_constu, 97, DW_OP_stack_value))
!21 = distinct !DIGlobalVariable(name: "TestConst2", scope: !0, file: !4, line: 3, type: !13, isLocal: true, isDefinition: true, declaration: !12)
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression(DW_OP_constu, 18446744071562551616, DW_OP_stack_value))
!23 = distinct !DIGlobalVariable(name: "ENUM_B", scope: !0, file: !4, line: 7, type: !3, isLocal: true, isDefinition: true)
!24 = !{i32 2, !"CodeView", i32 1}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{i32 1, !"wchar_size", i32 2}
!27 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git dee1891507401f396290b5d9cb5717d6b0755337)"}
!28 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !4, file: !4, line: 10, type: !29, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
!29 = !DISubroutineType(types: !30)
!30 = !{null}
!31 = !{}
!32 = !DILocation(line: 11, scope: !28)
!33 = !DILocation(line: 12, scope: !28)
!34 = !DILocation(line: 13, scope: !28)
!35 = !DILocation(line: 14, scope: !28)
