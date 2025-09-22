; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | FileCheck %s --check-prefix=ASM

; C++ source to regenerate:
; $ cat numeric-leaves.cpp
; const long long Minus1 = -1;
; const long long Minus128 = -128;
; const long long Minus129 = -129;
; const long long Minus32768 = -32768;
; const long long Minus32769 = -32769;
; const long long Minus2147483648 = -2147483648;
; const long long Minus2147483649 = -2147483649;
;
; const long long Zero = 0;
; const long long Plus32767 = 32767;
; const long long Plus32768 = 32768;
; const long long Plus2147483647 = 2147483647;
; const long long Plus2147483648 = 2147483648;
;
; int main(){
;   long long iDebug1 = Minus1 + Minus128 + Minus129 +
;                       Minus32768 + Minus32769 +
;                       Minus2147483648 + Minus2147483649;
;   long long iDebug2 = Zero + Plus32767 + Plus32768 +
;                       Plus2147483647 + Plus2147483648;
;
;   return 0;
;}
;
; $ clang numeric-leaves.cpp -S -emit-llvm -g -gcodeview -o nl.ll

; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -1
; CHECK-NEXT:      Name: Minus1
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -128
; CHECK-NEXT:      Name: Minus128
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -129
; CHECK-NEXT:      Name: Minus129
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -32768
; CHECK-NEXT:      Name: Minus32768
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -32769
; CHECK-NEXT:      Name: Minus32769
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -2147483648
; CHECK-NEXT:      Name: Minus2147483648
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: -2147483649
; CHECK-NEXT:      Name: Minus2147483649
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: 0
; CHECK-NEXT:      Name: Zero
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: 32767
; CHECK-NEXT:      Name: Plus32767
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: 32768
; CHECK-NEXT:      Name: Plus32768
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: 2147483647
; CHECK-NEXT:      Name: Plus2147483647
; CHECK-NEXT:    }
; CHECK:         ConstantSym {
; CHECK-NEXT:      Kind: S_CONSTANT (0x1107)
; CHECK-NEXT:      Type: const __int64 (0x1003)
; CHECK-NEXT:      Value: 2147483648
; CHECK-NEXT:      Name: Plus2147483648

; ASM-LABEL:    long     241                       # Symbol subsection for globals

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x00, 0x80, 0xff          # Value
; ASM-NEXT:     .asciz   "Minus1"                  # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x00, 0x80, 0x80          # Value
; ASM-NEXT:     .asciz   "Minus128"                # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x01, 0x80, 0x7f, 0xff    # Value
; ASM-NEXT:     .asciz   "Minus129"                # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x01, 0x80, 0x00, 0x80    # Value
; ASM-NEXT:     .asciz   "Minus32768"              # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x03, 0x80, 0xff, 0x7f    # Value
; ASM-NEXT:     .byte    0xff, 0xff
; ASM-NEXT:     .asciz   "Minus32769"              # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x03, 0x80, 0x00, 0x00    # Value
; ASM-NEXT:     .byte    0x00, 0x80
; ASM-NEXT:     .asciz   "Minus2147483648"         # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x09, 0x80, 0xff, 0xff    # Value
; ASM-NEXT:     .byte    0xff, 0x7f, 0xff, 0xff
; ASM-NEXT:     .byte    0xff, 0xff
; ASM-NEXT:     .asciz   "Minus2147483649"         # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x00, 0x00                # Value
; ASM-NEXT:     .asciz   "Zero"                    # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0xff, 0x7f                # Value
; ASM-NEXT:     .asciz   "Plus32767"               # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x03, 0x80, 0x00, 0x80    # Value
; ASM-NEXT:     .byte    0x00, 0x00
; ASM-NEXT:     .asciz   "Plus32768"               # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x03, 0x80, 0xff, 0xff    # Value
; ASM-NEXT:     .byte    0xff, 0x7f
; ASM-NEXT:     .asciz   "Plus2147483647"          # Name

; ASM:          .short   4359                      # Record kind: S_CONSTANT
; ASM-NEXT:     .long    4099                      # Type
; ASM-NEXT:     .byte    0x09, 0x80, 0x00, 0x00    # Value
; ASM-NEXT:     .byte    0x00, 0x80, 0x00, 0x00
; ASM-NEXT:     .byte    0x00, 0x00
; ASM-NEXT:     .asciz   "Plus2147483648"          # Name

; ModuleID = 'numeric-leaves.cpp'
source_filename = "numeric-leaves.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.29.30133"

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !35 {
entry:
  %retval = alloca i32, align 4
  %iDebug1 = alloca i64, align 8
  %iDebug2 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %iDebug1, metadata !40, metadata !DIExpression()), !dbg !41
  store i64 -4295033092, ptr %iDebug1, align 8, !dbg !41
  call void @llvm.dbg.declare(metadata ptr %iDebug2, metadata !42, metadata !DIExpression()), !dbg !43
  store i64 4295032830, ptr %iDebug2, align 8, !dbg !43
  ret i32 0, !dbg !44
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30, !31, !32, !33}
!llvm.ident = !{!34}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "numeric-leaves.cpp", directory: "d:\\tmp", checksumkind: CSK_MD5, checksum: "9b1d86040d3b979a9b2b3e86c7bea0b4")
!2 = !{!3, !7, !9, !11, !13, !15, !17, !19, !21, !23, !25, !27}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression(DW_OP_constu, 18446744073709551615, DW_OP_stack_value))
!4 = distinct !DIGlobalVariable(name: "Minus1", scope: !0, file: !1, line: 1, type: !5, isLocal: true, isDefinition: true)
!5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!6 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_constu, 18446744073709551488, DW_OP_stack_value))
!8 = distinct !DIGlobalVariable(name: "Minus128", scope: !0, file: !1, line: 2, type: !5, isLocal: true, isDefinition: true)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression(DW_OP_constu, 18446744073709551487, DW_OP_stack_value))
!10 = distinct !DIGlobalVariable(name: "Minus129", scope: !0, file: !1, line: 3, type: !5, isLocal: true, isDefinition: true)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression(DW_OP_constu, 18446744073709518848, DW_OP_stack_value))
!12 = distinct !DIGlobalVariable(name: "Minus32768", scope: !0, file: !1, line: 4, type: !5, isLocal: true, isDefinition: true)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_constu, 18446744073709518847, DW_OP_stack_value))
!14 = distinct !DIGlobalVariable(name: "Minus32769", scope: !0, file: !1, line: 5, type: !5, isLocal: true, isDefinition: true)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression(DW_OP_constu, 18446744071562067968, DW_OP_stack_value))
!16 = distinct !DIGlobalVariable(name: "Minus2147483648", scope: !0, file: !1, line: 6, type: !5, isLocal: true, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression(DW_OP_constu, 18446744071562067967, DW_OP_stack_value))
!18 = distinct !DIGlobalVariable(name: "Minus2147483649", scope: !0, file: !1, line: 7, type: !5, isLocal: true, isDefinition: true)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
!20 = distinct !DIGlobalVariable(name: "Zero", scope: !0, file: !1, line: 9, type: !5, isLocal: true, isDefinition: true)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression(DW_OP_constu, 32767, DW_OP_stack_value))
!22 = distinct !DIGlobalVariable(name: "Plus32767", scope: !0, file: !1, line: 10, type: !5, isLocal: true, isDefinition: true)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression(DW_OP_constu, 32768, DW_OP_stack_value))
!24 = distinct !DIGlobalVariable(name: "Plus32768", scope: !0, file: !1, line: 11, type: !5, isLocal: true, isDefinition: true)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression(DW_OP_constu, 2147483647, DW_OP_stack_value))
!26 = distinct !DIGlobalVariable(name: "Plus2147483647", scope: !0, file: !1, line: 12, type: !5, isLocal: true, isDefinition: true)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression(DW_OP_constu, 2147483648, DW_OP_stack_value))
!28 = distinct !DIGlobalVariable(name: "Plus2147483648", scope: !0, file: !1, line: 13, type: !5, isLocal: true, isDefinition: true)
!29 = !{i32 2, !"CodeView", i32 1}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 1, !"wchar_size", i32 2}
!32 = !{i32 7, !"PIC Level", i32 2}
!33 = !{i32 7, !"uwtable", i32 2}
!34 = !{!"clang version 15.0.0"}
!35 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 15, type: !36, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
!36 = !DISubroutineType(types: !37)
!37 = !{!38}
!38 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!39 = !{}
!40 = !DILocalVariable(name: "iDebug1", scope: !35, file: !1, line: 16, type: !6)
!41 = !DILocation(line: 16, scope: !35)
!42 = !DILocalVariable(name: "iDebug2", scope: !35, file: !1, line: 19, type: !6)
!43 = !DILocation(line: 19, scope: !35)
!44 = !DILocation(line: 22, scope: !35)
