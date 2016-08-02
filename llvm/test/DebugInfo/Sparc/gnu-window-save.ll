; RUN: llc -filetype=obj -O0 < %s -mtriple sparc64-unknown-linux-gnu | llvm-dwarfdump - | FileCheck %s --check-prefix=SPARC64
; RUN: llc -filetype=obj -O0 < %s -mtriple sparc-unknown-linux-gnu   | llvm-dwarfdump - | FileCheck %s --check-prefix=SPARC32

; Check for DW_CFA_GNU_Window_save in debug_frame. Also, Ensure that relocations
; are performed correctly in debug_info.

; SPARC64: file format ELF64-sparc

; SPARC64: .debug_info
; SPARC64:      DW_TAG_compile_unit
; SPARC64:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "hello.c")
; SPARC64:      DW_TAG_subprogram
; SPARC64:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "main")
; SPARC64:      DW_TAG_base_type
; SPARC64:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "int")

; SPARC64: .debug_frame
; SPARC64:      DW_CFA_def_cfa_register
; SPARC64-NEXT: DW_CFA_GNU_window_save
; SPARC64-NEXT: DW_CFA_register


; SPARC32: file format ELF32-sparc

; SPARC32: .debug_info
; SPARC32:      DW_TAG_compile_unit
; SPARC32:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "hello.c")
; SPARC32:      DW_TAG_subprogram
; SPARC32:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "main")
; SPARC32:      DW_TAG_base_type
; SPARC32:        DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9,A-F,a-f]+}}] = "int")

; SPARC32: .debug_frame
; SPARC32:      DW_CFA_def_cfa_register
; SPARC32-NEXT: DW_CFA_GNU_window_save
; SPARC32-NEXT: DW_CFA_register

@.str = private unnamed_addr constant [14 x i8] c"hello, world\0A\00", align 1

; Function Attrs: nounwind
define signext i32 @main() #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i32 0, i32 0)), !dbg !12
  ret i32 0, !dbg !13
}

declare signext i32 @printf(i8*, ...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 (http://llvm.org/git/clang.git 6a0714fee07fb7c4e32d3972b4fe2ce2f5678cf4) (llvm/ 672e88e934757f76d5c5e5258be41e7615094844)", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "hello.c", directory: "/home/venkatra/work/benchmarks/test/hello")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "main", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "hello.c", directory: "/home/venkatra/work/benchmarks/test/hello")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5 (http://llvm.org/git/clang.git 6a0714fee07fb7c4e32d3972b4fe2ce2f5678cf4) (llvm/ 672e88e934757f76d5c5e5258be41e7615094844)"}
!12 = !DILocation(line: 5, scope: !4)
!13 = !DILocation(line: 6, scope: !4)
