; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; const int Test1 = 1;
; int main() {
;   return Test1;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM-LABEL:  .long 241              # Symbol subsection for globals

; ASM:        .short {{.*-.*}}       # Record length
; ASM:        .short 4359            # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4099             # Type
; ASM-NEXT:   .byte 0x01, 0x00       # Value
; ASM-NEXT:   .asciz "Test1"         # Name

; OBJ:        CodeViewDebugInfo [
; OBJ:          Section: .debug$S
; OBJ:          Magic: 0x4
; OBJ:          Subsection [
; OBJ:            SubSectionType: Symbols (0xF1)
; OBJ:            ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const int (0x1003)
; OBJ-NEXT:         Value: 1
; OBJ-NEXT:         Name: Test1
; OBJ-NEXT:       }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: noinline norecurse nounwind optnone
define dso_local i32 @main() #0 !dbg !13 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 1, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4a1902b6739e3087a03c0ac7ab85b640764e9335)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "0d5ef00bdd80bdb409a3deac9938f20d")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
!5 = distinct !DIGlobalVariable(name: "Test1", scope: !0, file: !6, line: 1, type: !7, isLocal: true, isDefinition: true)
!6 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "0d5ef00bdd80bdb409a3deac9938f20d")
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"CodeView", i32 1}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 2}
!12 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4a1902b6739e3087a03c0ac7ab85b640764e9335)"}
!13 = distinct !DISubprogram(name: "main", scope: !6, file: !6, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!8}
!16 = !DILocation(line: 4, scope: !13)
