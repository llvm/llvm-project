; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; const int Test1 = 1;
; struct Foo { static const int Test2 = 2; };
; int main() {
;   return Test1 + Foo::Test2;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM-LABEL:  .long 241             # Symbol subsection for globals

; ASM:        .short {{.*-.*}}      # Record length
; ASM:        .short 4359           # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4099            # Type
; ASM-NEXT:   .byte 0x01, 0x00      # Value
; ASM-NEXT:   .asciz "Test1"        # Name

; ASM:        .short {{.*-.*}}      # Record length
; ASM:        .short 4359           # Record kind: S_CONSTANT
; ASM:        .long 4099            # Type
; ASM:        .byte 0x02, 0x00      # Value
; ASM:        .asciz "Foo::Test2"   # Name

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
; OBJ:            ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const int (0x1003)
; OBJ-NEXT:         Value: 2
; OBJ-NEXT:         Name: Foo::Test2
; OBJ-NEXT:       }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: noinline norecurse nounwind optnone
define dso_local i32 @main() #0 !dbg !19 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 3, !dbg !22
}

attributes #0 = { noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git 2b66a49044196d8b90d95d7d3b5246ccbe3abc05)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !10, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "77cff5e1c7b260440ed03b23c18809c3")
!2 = !{}
!3 = !{!4}
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !5, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: ".?AUFoo@@")
!5 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "77cff5e1c7b260440ed03b23c18809c3")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "Test2", scope: !4, file: !5, line: 4, baseType: !8, flags: DIFlagStaticMember, extraData: i32 2)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !13}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
!12 = distinct !DIGlobalVariable(name: "Test1", scope: null, file: !5, line: 1, type: !8, isLocal: true, isDefinition: true)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_constu, 2, DW_OP_stack_value))
!14 = distinct !DIGlobalVariable(name: "Test2", scope: !0, file: !5, line: 4, type: !8, isLocal: true, isDefinition: true, declaration: !7)
!15 = !{i32 2, !"CodeView", i32 1}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 2}
!18 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 2b66a49044196d8b90d95d7d3b5246ccbe3abc05)"}
!19 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 7, type: !20, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{!9}
!22 = !DILocation(line: 8, scope: !19)
