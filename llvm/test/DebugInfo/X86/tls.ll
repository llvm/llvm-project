; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-64 --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-32 --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -split-dwarf-file=foo.dwo \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=FISSION --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-scei-ps4 \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-64 --check-prefix=STDOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-apple-darwin \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=DARWIN --check-prefix=STDOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-freebsd \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-64 --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -emulated-tls \
; RUN:   | FileCheck --check-prefix=SINGLE --check-prefix=EMUSINGLE-64 \
; RUN:     --check-prefix=EMUGNUOP --check-prefix=EMU %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu -emulated-tls \
; RUN:   | FileCheck --check-prefix=SINGLE --check-prefix=EMUSINGLE-32 \
; RUN:     --check-prefix=EMUGNUOP --check-prefix=EMU %s

; TODO: Add expected output for -emulated-tls tests.

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; FISSION: .section    .debug_info.dwo,
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; FISSION: .byte 3{{ *}}# DW_AT_location
; DW_OP_GNU_const_index (0xfx == 252) to refer to the debug_addr table
; FISSION-NEXT: .byte 252
; an index of zero into the debug_addr table
; FISSION-NEXT: .byte 0

; SINGLE: .section     .debug_info,
; DARWIN: .section     {{.*}}debug_info,

; 10 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; SINGLE-64: .byte     10 # DW_AT_location
; DW_OP_const8u (0x0e == 14) of address
; SINGLE-64-NEXT: .byte        14
; SINGLE-64-NEXT: .quad tls@DTPOFF

; DARWIN: .byte     10 ## DW_AT_location
; DW_OP_const8u (0x0e == 14) of address
; DARWIN-NEXT: .byte        14
; DARWIN-NEXT: .quad _tls

; 6 bytes of data in 32-bit mode
; SINGLE-32: .byte     6 # DW_AT_location
; DW_OP_const4u (0x0e == 12) of address
; SINGLE-32-NEXT: .byte        12
; SINGLE-32-NEXT: .long tls@DTPOFF

; DW_OP_GNU_push_tls_address
; GNUOP-NEXT: .byte 224
; DW_OP_form_tls_address
; STDOP-NEXT: .byte 155

; FISSION: DW_TAG_variable
; FISSION: .byte 2 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1

; FISSION: DW_TAG_template_value_parameter
; FISSION: .byte 3 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1
; DW_OP_stack_value
; FISSION-NEXT: .byte 159

; check that the expected TLS address description is the first thing in the debug_addr section
; FISSION: .section    .debug_addr
; FISSION-NEXT: .quad  tls@DTPOFF
; FISSION-NEXT: .quad  glbl
; FISSION-NOT: .quad  glbl

; Generated from:

; __thread int tls;
; int glbl;
;
; template <int *I>
; int func() {
;   return 0;
; }
;
; template int func<&glbl>(); // create a second reference to 'glbl'

source_filename = "test/DebugInfo/X86/tls.ll"

@tls = thread_local global i32 0, align 4, !dbg !0
@glbl = global i32 0, align 4, !dbg !4

; Function Attrs: nounwind uwtable
define weak_odr i32 @_Z4funcIXadL_Z4glblEEEiv() #0 !dbg !12 {
entry:
  ret i32 0, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "tls", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tls.cpp", directory: "/tmp/dbginfo")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5)
!5 = !DIGlobalVariable(name: "glbl", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "-.dwo", emissionKind: FullDebug, enums: !7, retainedTypes: !7, globals: !8, imports: !7)
!7 = !{}
!8 = !{!0, !4}
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5 "}
!12 = distinct !DISubprogram(name: "func<&glbl>", linkageName: "_Z4funcIXadL_Z4glblEEEiv", scope: !2, file: !2, line: 5, type: !13, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !6, templateParams: !15, variables: !7)
!13 = !DISubroutineType(types: !14)
!14 = !{!3}
!15 = !{!16}
!16 = !DITemplateValueParameter(name: "I", type: !17, value: i32* @glbl)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, align: 64)
!18 = !DILocation(line: 6, scope: !12)

