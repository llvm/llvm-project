; RUN: llc -filetype=obj %s -o - | llvm-readobj -r -S --symbols - | FileCheck %s

; CHECK:      Format: WASM
; CHECK-NEXT: Arch: wasm64
; CHECK-NEXT: AddressSize: 64bit
; CHECK-NEXT: Sections [
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: TYPE (0x1)
; CHECK-NEXT:     Size: 4
; CHECK-NEXT:     Offset: 8
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: IMPORT (0x2)
; CHECK-NEXT:     Size: 81
; CHECK-NEXT:     Offset: 18
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: FUNCTION (0x3)
; CHECK-NEXT:     Size: 2
; CHECK-NEXT:     Offset: 105
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: ELEM (0x9)
; CHECK-NEXT:     Size: 7
; CHECK-NEXT:     Offset: 113
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: DATACOUNT (0xC)
; CHECK-NEXT:     Size: 1
; CHECK-NEXT:     Offset: 126
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CODE (0xA)
; CHECK-NEXT:     Size: 4
; CHECK-NEXT:     Offset: 133
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: DATA (0xB)
; CHECK-NEXT:     Size: 27
; CHECK-NEXT:     Offset: 143
; CHECK-NEXT:     Segments [
; CHECK-NEXT:       Segment {
; CHECK-NEXT:         Name: .data.foo
; CHECK-NEXT:         Size: 8
; CHECK-NEXT:         Offset: 0
; CHECK-NEXT:       }
; CHECK-NEXT:       Segment {
; CHECK-NEXT:         Name: .data.ptr2
; CHECK-NEXT:         Size: 8
; CHECK-NEXT:         Offset: 8
; CHECK-NEXT:       }
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 86
; CHECK-NEXT:     Offset: 176
; CHECK-NEXT:     Name: .debug_abbrev
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 130
; CHECK-NEXT:     Offset: 282
; CHECK-NEXT:     Name: .debug_info
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 121
; CHECK-NEXT:     Offset: 430
; CHECK-NEXT:     Name: .debug_str
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 42
; CHECK-NEXT:     Offset: 568
; CHECK-NEXT:     Name: .debug_pubnames
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 26
; CHECK-NEXT:     Offset: 632
; CHECK-NEXT:     Name: .debug_pubtypes
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 61
; CHECK-NEXT:     Offset: 680
; CHECK-NEXT:     Name: .debug_line
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 91
; CHECK-NEXT:     Offset: 759
; CHECK-NEXT:     Name: linking
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 9
; CHECK-NEXT:     Offset: 864
; CHECK-NEXT:     Name: reloc.DATA
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 61
; CHECK-NEXT:     Offset: 890
; CHECK-NEXT:     Name: reloc..debug_info
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 6
; CHECK-NEXT:     Offset: 975
; CHECK-NEXT:     Name: reloc..debug_pubnames
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 6
; CHECK-NEXT:     Offset: 1009
; CHECK-NEXT:     Name: reloc..debug_pubtypes
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 6
; CHECK-NEXT:     Offset: 1043
; CHECK-NEXT:     Name: reloc..debug_line
; CHECK-NEXT:   }
; CHECK-NEXT:   Section {
; CHECK-NEXT:     Type: CUSTOM (0x0)
; CHECK-NEXT:     Size: 77
; CHECK-NEXT:     Offset: 1073
; CHECK-NEXT:     Name: producers
; CHECK-NEXT:   }
; CHECK-NEXT: ]
; CHECK-NEXT: Relocations [
; CHECK-NEXT:   Section (7) DATA {
; CHECK-NEXT:     0x6 R_WASM_MEMORY_ADDR_I64 myextern 0
; CHECK-NEXT:     0x13 R_WASM_TABLE_INDEX_I64 f2
; CHECK-NEXT:   }
; CHECK-NEXT:   Section (9) .debug_info {
; CHECK-NEXT:     0x6 R_WASM_SECTION_OFFSET_I32 .debug_abbrev 0
; CHECK-NEXT:     0xC R_WASM_SECTION_OFFSET_I32 .debug_str 0
; CHECK-NEXT:     0x12 R_WASM_SECTION_OFFSET_I32 .debug_str 55
; CHECK-NEXT:     0x16 R_WASM_SECTION_OFFSET_I32 .debug_line 0
; CHECK-NEXT:     0x1A R_WASM_SECTION_OFFSET_I32 .debug_str 62
; CHECK-NEXT:     0x1E R_WASM_FUNCTION_OFFSET_I64 f2 0
; CHECK-NEXT:     0x2B R_WASM_SECTION_OFFSET_I32 .debug_str 105
; CHECK-NEXT:     0x37 R_WASM_MEMORY_ADDR_I64 foo 0
; CHECK-NEXT:     0x45 R_WASM_SECTION_OFFSET_I32 .debug_str 109
; CHECK-NEXT:     0x4C R_WASM_SECTION_OFFSET_I32 .debug_str 113
; CHECK-NEXT:     0x58 R_WASM_MEMORY_ADDR_I64 ptr2 0
; CHECK-NEXT:     0x67 R_WASM_FUNCTION_OFFSET_I64 f2 0
; CHECK-NEXT:     0x76 R_WASM_GLOBAL_INDEX_I32 __stack_pointer
; CHECK-NEXT:     0x7B R_WASM_SECTION_OFFSET_I32 .debug_str 118
; CHECK-NEXT:   }
; CHECK-NEXT:   Section (11) .debug_pubnames {
; CHECK-NEXT:     0x6 R_WASM_SECTION_OFFSET_I32 .debug_info 0
; CHECK-NEXT:   }
; CHECK-NEXT:   Section (12) .debug_pubtypes {
; CHECK-NEXT:     0x6 R_WASM_SECTION_OFFSET_I32 .debug_info 0
; CHECK-NEXT:   }
; CHECK-NEXT:   Section (13) .debug_line {
; CHECK-NEXT:     0x2B R_WASM_FUNCTION_OFFSET_I64 f2 0
; CHECK-NEXT:   }
; CHECK-NEXT: ]
; CHECK-NEXT: Symbols [
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: f2
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x4)
; CHECK-NEXT:       VISIBILITY_HIDDEN (0x4)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: foo
; CHECK-NEXT:     Type: DATA (0x1)
; CHECK-NEXT:     Flags [ (0x4)
; CHECK-NEXT:       VISIBILITY_HIDDEN (0x4)
; CHECK-NEXT:     ]
; CHECK-NEXT:     Offset: 0x0
; CHECK-NEXT:     Segment: 0x0
; CHECK-NEXT:     Size: 0x8
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: myextern
; CHECK-NEXT:     Type: DATA (0x1)
; CHECK-NEXT:     Flags [ (0x10)
; CHECK-NEXT:       UNDEFINED (0x10)
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: ptr2
; CHECK-NEXT:     Type: DATA (0x1)
; CHECK-NEXT:     Flags [ (0x4)
; CHECK-NEXT:       VISIBILITY_HIDDEN (0x4)
; CHECK-NEXT:     ]
; CHECK-NEXT:     Offset: 0x0
; CHECK-NEXT:     Segment: 0x1
; CHECK-NEXT:     Size: 0x8
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: .debug_abbrev
; CHECK-NEXT:     Type: SECTION (0x3)
; CHECK-NEXT:     Flags [ (0x2)
; CHECK-NEXT:       BINDING_LOCAL (0x2)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x7
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: .debug_info
; CHECK-NEXT:     Type: SECTION (0x3)
; CHECK-NEXT:     Flags [ (0x2)
; CHECK-NEXT:       BINDING_LOCAL (0x2)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x8
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: __stack_pointer
; CHECK-NEXT:     Type: GLOBAL (0x2)
; CHECK-NEXT:     Flags [ (0x10)
; CHECK-NEXT:       UNDEFINED (0x10)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ImportName: __stack_pointer
; CHECK-NEXT:     ImportModule: env
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: .debug_str
; CHECK-NEXT:     Type: SECTION (0x3)
; CHECK-NEXT:     Flags [ (0x2)
; CHECK-NEXT:       BINDING_LOCAL (0x2)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x9
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Name: .debug_line
; CHECK-NEXT:     Type: SECTION (0x3)
; CHECK-NEXT:     Flags [ (0x2)
; CHECK-NEXT:       BINDING_LOCAL (0x2)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0xC
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; generated from the following C code using: clang --target=wasm64 -g -O0 -S -emit-llvm test.c
; extern int myextern;
; void f2(void) { return; }
;
; int* foo = &myextern;
; void (*ptr2)(void) = f2;

target triple = "wasm64-unknown-unknown"

source_filename = "test.c"

@myextern = external global i32, align 4
@foo = hidden global i32* @myextern, align 4, !dbg !0
@ptr2 = hidden global void ()* @f2, align 4, !dbg !6

; Function Attrs: noinline nounwind optnone
define hidden void @f2() #0 !dbg !17 {
entry:
  ret void, !dbg !18
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 4, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 332303) (llvm/trunk 332406)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/usr/local/google/home/sbc/dev/wasm/simple")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "ptr2", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 7.0.0 (trunk 332303) (llvm/trunk 332406)"}
!17 = distinct !DISubprogram(name: "f2", scope: !3, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!18 = !DILocation(line: 2, column: 17, scope: !17)
