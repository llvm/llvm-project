; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; namespace Test1 {
; const float TestConst1 = 3.14;
; }
; struct S {
;   static const int TestConst2 = -10;
;   enum { SEnum = 42 };
; }
; enum TestEnum : int {
;    ENUM_A = 2147000000,
;    ENUM_B = -2147000000,
; };
; void useConst(int);
; void foo() {
;   useConst(Test1::TestConst1);
;   useConst(S::TestConst2);
;   useConst(ENUM_B);
;   useConst(S::SEnum);
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM-LABEL:  .long 241                     # Symbol subsection for globals
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4102                    # Type
; ASM-NEXT:   .byte 0x04, 0x80, 0xc3, 0xf5  # Value
; ASM-NEXT:   .byte 0x48, 0x40
; ASM-NEXT:   .asciz "Test1::TestConst1"    # Name
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4103                    # Type
; ASM-NEXT:   .byte 0x61, 0x00              # Value
; ASM-NEXT:   .asciz "S::TestConst2"        # Name
; ASM:        .short {{.*-.*}}              # Record length
; ASM:        .short 4359                   # Record kind: S_CONSTANT
; ASM-NEXT:   .long 4105                    # Type
; ASM-NEXT:   .byte 0x0a, 0x80, 0x40, 0x61  # Value
; ASM-NEXT:   .byte 0x07, 0x80, 0xff, 0xff
; ASM-NEXT:   .byte 0xff, 0xff
; ASM-NEXT:   .asciz "ENUM_B"               # Name
; ASM-NOT:    .asciz "S::SEnum"             # Name

; OBJ:        CodeViewDebugInfo [
; OBJ:          Section: .debug$S
; OBJ:          Magic: 0x4
; OBJ:          Subsection [
; OBJ:            SubSectionType: Symbols (0xF1)
; OBJ:            ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const float (0x1006)
; OBJ-NEXT:         Value: 1078523331
; OBJ-NEXT:         Name: Test1::TestConst1
; OBJ-NEXT:       }
; OBJ-NEXT:       ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: const char (0x1007)
; OBJ-NEXT:         Value: 97
; OBJ-NEXT:         Name: S::TestConst2
; OBJ-NEXT:       }
; OBJ-NEXT:       ConstantSym {
; OBJ-NEXT:         Kind: S_CONSTANT (0x1107)
; OBJ-NEXT:         Type: TestEnum (0x1009)
; OBJ-NEXT:         Value: 18446744071562551616
; OBJ-NEXT:         Name: ENUM_B
; OBJ-NEXT:       }
; OBJ-NOT:          Name: S::SEnum

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27030"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?useConst@@YAXH@Z"(i32) #0 !dbg !32 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !36, metadata !DIExpression()), !dbg !37
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #2 !dbg !38 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @"?useConst@@YAXH@Z"(i32 3), !dbg !41
  call void @"?useConst@@YAXH@Z"(i32 97), !dbg !42
  call void @"?useConst@@YAXH@Z"(i32 -2147000000), !dbg !43
  call void @"?useConst@@YAXH@Z"(i32 42), !dbg !44
  call void @llvm.debugtrap(), !dbg !45
  ret i32 0, !dbg !46
}

; Function Attrs: nounwind
declare void @llvm.debugtrap() #3

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27, !28, !29, !30}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git f60f863075c7056f26e701b0405fc5752f0db576)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !16, globals: !17, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Ctesting", checksumkind: CSK_MD5, checksum: "70da26ef1009521e2127bf71f8d532a2")
!2 = !{!3, !12}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !4, file: !1, line: 6, baseType: !9, size: 32, elements: !10, identifier: ".?AW4<unnamed-enum-SEnum>@S@@")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: ".?AUS@@")
!5 = !{!6, !3}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "TestConst2", scope: !4, file: !1, line: 5, baseType: !7, flags: DIFlagStaticMember, extraData: i8 97)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DIEnumerator(name: "SEnum", value: 42)
!12 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "TestEnum", file: !1, line: 8, baseType: !9, size: 32, elements: !13, identifier: ".?AW4TestEnum@@")
!13 = !{!14, !15}
!14 = !DIEnumerator(name: "ENUM_A", value: 2147000000)
!15 = !DIEnumerator(name: "ENUM_B", value: -2147000000)
!16 = !{!4}
!17 = !{!18, !23, !25}
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression(DW_OP_constu, 1078523331, DW_OP_stack_value))
!19 = distinct !DIGlobalVariable(name: "TestConst1", scope: !20, file: !1, line: 2, type: !21, isLocal: true, isDefinition: true)
!20 = !DINamespace(name: "Test1", scope: null)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!22 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression(DW_OP_constu, 97, DW_OP_stack_value))
!24 = distinct !DIGlobalVariable(name: "TestConst2", scope: !0, file: !1, line: 5, type: !7, isLocal: true, isDefinition: true, declaration: !6)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression(DW_OP_constu, 18446744071562551616, DW_OP_stack_value))
!26 = distinct !DIGlobalVariable(name: "ENUM_B", scope: !0, file: !1, line: 10, type: !12, isLocal: true, isDefinition: true)
!27 = !{i32 2, !"CodeView", i32 1}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 2}
!30 = !{i32 7, !"PIC Level", i32 2}
!31 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git f60f863075c7056f26e701b0405fc5752f0db576)"}
!32 = distinct !DISubprogram(name: "useConst", linkageName: "?useConst@@YAXH@Z", scope: !1, file: !1, line: 12, type: !33, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !9}
!35 = !{}
!36 = !DILocalVariable(arg: 1, scope: !32, file: !1, line: 12, type: !9)
!37 = !DILocation(line: 12, scope: !32)
!38 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !39, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
!39 = !DISubroutineType(types: !40)
!40 = !{!9}
!41 = !DILocation(line: 14, scope: !38)
!42 = !DILocation(line: 15, scope: !38)
!43 = !DILocation(line: 16, scope: !38)
!44 = !DILocation(line: 17, scope: !38)
!45 = !DILocation(line: 18, scope: !38)
!46 = !DILocation(line: 19, scope: !38)
