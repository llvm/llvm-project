; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-64 --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu \
; RUN:   | FileCheck --check-prefix=NOEMU --check-prefix=SINGLE --check-prefix=SINGLE-32 --check-prefix=GNUOP %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -split-dwarf=Enable \
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
; FISSION: addr_sec:
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


@tls = thread_local global i32 0, align 4, !dbg !13
@glbl = global i32 0, align 4, !dbg !14

; Function Attrs: nounwind uwtable
define weak_odr i32 @_Z4funcIXadL_Z4glblEEEiv() #0 !dbg !4 {
entry:
  ret i32 0, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, splitDebugFilename: "-.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !12, imports: !2)
!1 = !DIFile(filename: "tls.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "func<&glbl>", linkageName: "_Z4funcIXadL_Z4glblEEEiv", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, templateParams: !9, variables: !2)
!5 = !DIFile(filename: "tls.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DITemplateValueParameter(tag: DW_TAG_template_value_parameter, name: "I", type: !11, value: i32* @glbl)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!12 = !{!13, !14}
!13 = !DIGlobalVariable(name: "tls", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8)
!14 = !DIGlobalVariable(name: "glbl", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 1, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.5 "}
!18 = !DILocation(line: 6, scope: !4)
