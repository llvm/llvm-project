; RUN: llc -generate-arange-section < %s | FileCheck %s


; -- header --
; CHECK: .short 2 # DWARF Arange version number
; CHECK-NEXT: .long .Lcu_begin0
; CHECK-NEXT: .byte 8 # Address Size (in bytes)
; CHECK-NEXT: .byte 0 # Segment Size (in bytes)
; -- alignment --
; CHECK-NEXT: .zero 4,255

; <data section> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_data
; CHECK-NEXT: .quad .Lsec_end0-some_data

; <other sections> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_other
; CHECK-NEXT: .quad .Lsec_end1-some_other

; <common symbols> - it should have made one span for each symbol.
; CHECK-NEXT: .quad some_bss
; CHECK-NEXT: .quad 4

; <text section> - it should have made one span covering all functions in this CU.
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .quad .Lsec_end2-.Lfunc_begin0

; -- finish --
; CHECK-NEXT: # ARange terminator



; -- source code --
; Generated from: "clang -c -g -emit-llvm"
;
; int some_data = 4;
; int some_bss;
; int some_other __attribute__ ((section ("strange+section"))) = 5;
; 
; void some_code()
; {
;    some_bss += some_data + some_other;
; }

target triple = "x86_64-unknown-linux-gnu"

@some_data = global i32 4, align 4, !dbg !9
@some_other = global i32 5, section "strange+section", align 4, !dbg !11
@some_bss = common global i32 0, align 4, !dbg !12

define void @some_code() !dbg !4 {
entry:
  %0 = load i32, i32* @some_data, align 4, !dbg !14
  %1 = load i32, i32* @some_other, align 4, !dbg !14
  %add = add nsw i32 %0, %1, !dbg !14
  %2 = load i32, i32* @some_bss, align 4, !dbg !14
  %add1 = add nsw i32 %2, %add, !dbg !14
  store i32 %add1, i32* @some_bss, align 4, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !8, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/kayamon")
!2 = !{}
!4 = distinct !DISubprogram(name: "some_code", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 6, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "test.c", directory: "/home/kayamon")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9, !11, !12}
!9 = !DIGlobalVariable(name: "some_data", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIGlobalVariable(name: "some_other", line: 3, isLocal: false, isDefinition: true, scope: null, file: !5, type: !10)
!12 = !DIGlobalVariable(name: "some_bss", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !10)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !DILocation(line: 7, scope: !4)
!15 = !DILocation(line: 8, scope: !4)
!16 = !{i32 1, !"Debug Info Version", i32 3}
