; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK-NEXT: 0x00000000: Compile Unit: length = 0x0000006e, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x04 (next unit at 0x00000072)

; CHECK: 0x0000000b: DW_TAG_compile_unit
; CHECK-NEXT:              DW_AT_producer	("clang version 6.0.0 (trunk 315924) (llvm/trunk 315960)")
; CHECK-NEXT:              DW_AT_language	(DW_LANG_C99)
; CHECK-NEXT:              DW_AT_name	("test.c")
; CHECK-NEXT:              DW_AT_stmt_list	(0x00000000)
; CHECK-NEXT:              DW_AT_comp_dir	("/usr/local/google/home/sbc/dev/wasm/simple")
; CHECK-NEXT:              DW_AT_GNU_pubnames	(true)
; CHECK-NEXT:              DW_AT_low_pc		(0x0000000000000002)
; CHECK-NEXT:              DW_AT_high_pc		(0x0000000000000004)

; CHECK: 0x00000026:   DW_TAG_variable
; CHECK-NEXT:                DW_AT_name	("foo")
; CHECK-NEXT:                DW_AT_type	(0x00000037 "int*")
; CHECK-NEXT:                DW_AT_external	(true)
; CHECK-NEXT:                DW_AT_decl_file	("/usr/local/google/home/sbc/dev/wasm/simple{{[/\\]}}test.c")
; CHECK-NEXT:                DW_AT_decl_line	(4)
; CHECK-NEXT:                DW_AT_location	(DW_OP_addr 0x0)

; CHECK: 0x00000037:   DW_TAG_pointer_type
; CHECK-NEXT:                DW_AT_type	(0x0000003c "int")

; CHECK: 0x0000003c:   DW_TAG_base_type
; CHECK-NEXT:                DW_AT_name	("int")
; CHECK-NEXT:                DW_AT_encoding	(DW_ATE_signed)
; CHECK-NEXT:                DW_AT_byte_size	(0x04)

; CHECK: 0x00000043:   DW_TAG_variable
; CHECK-NEXT:                DW_AT_name	("ptr2")
; CHECK-NEXT:                DW_AT_type	(0x00000054 "void()*")
; CHECK-NEXT:                DW_AT_external	(true)
; CHECK-NEXT:                DW_AT_decl_file	("/usr/local/google/home/sbc/dev/wasm/simple{{[/\\]}}test.c")
; CHECK-NEXT:                DW_AT_decl_line	(5)
; CHECK-NEXT:                DW_AT_location	(DW_OP_addr 0x4)

; CHECK: 0x00000054:   DW_TAG_pointer_type
; CHECK-NEXT:                DW_AT_type	(0x00000059 "void()")

; CHECK: 0x00000059:   DW_TAG_subroutine_type
; CHECK-NEXT:                DW_AT_prototyped	(true)

; CHECK: 0x0000005a:   DW_TAG_subprogram
; CHECK-NEXT:                DW_AT_low_pc	(0x0000000000000002)
; CHECK-NEXT:                DW_AT_high_pc	(0x0000000000000004)
; CHECK-NEXT:                DW_AT_frame_base	(DW_OP_WASM_location 0x3 0x0, DW_OP_stack_value)
; CHECK-NEXT:                DW_AT_name	("f2")
; CHECK-NEXT:                DW_AT_decl_file	("/usr/local/google/home/sbc/dev/wasm/simple{{[/\\]}}test.c")
; CHECK-NEXT:                DW_AT_decl_line	(2)
; CHECK-NEXT:                DW_AT_prototyped	(true)
; CHECK-NEXT:                DW_AT_external		(true)

; CHECK: 0x00000071:   NULL

target triple = "wasm32-unknown-unknown"

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
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 315924) (llvm/trunk 315960)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
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
!16 = !{!"clang version 6.0.0 (trunk 315924) (llvm/trunk 315960)"}
!17 = distinct !DISubprogram(name: "f2", scope: !3, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!18 = !DILocation(line: 2, column: 16, scope: !17)
