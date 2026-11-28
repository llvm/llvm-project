# RUN: llvm-mc -triple=x86_64 -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o | FileCheck %s --check-prefix=VALID
# RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix=DUMP
# RUN: llvm-mc -triple=x86_64 -filetype=obj --defsym CASE=1 %s -o %t.bad.o
# RUN: not llvm-dwarfdump --verify %t.bad.o | FileCheck %s --check-prefix=WRONG-TAG
# RUN: llvm-mc -triple=x86_64 -filetype=obj --defsym CASE=2 %s -o %t.bad.o
# RUN: not llvm-dwarfdump --verify %t.bad.o | FileCheck %s --check-prefix=SPECIFICATION
# RUN: llvm-mc -triple=x86_64 -filetype=obj --defsym CASE=3 %s -o %t.bad.o
# RUN: not llvm-dwarfdump --verify %t.bad.o | FileCheck %s --check-prefix=STANDARD
# RUN: llvm-mc -triple=x86_64 -filetype=obj --defsym CASE=4 %s -o %t.bad.o
# RUN: not llvm-dwarfdump --verify %t.bad.o | FileCheck %s --check-prefix=DANGLING

# VALID: No errors.
# DUMP: DW_TAG_GNU_call_site_parameter
# DUMP-NEXT: DW_AT_abstract_origin {{.*}} "tstate"
# WRONG-TAG: DW_TAG_GNU_call_site_parameter has DW_AT_abstract_origin that points to DIE with incompatible tag DW_TAG_base_type
# SPECIFICATION: DW_TAG_GNU_call_site_parameter has DW_AT_specification that points to DIE with incompatible tag DW_TAG_formal_parameter
# STANDARD: DW_TAG_call_site_parameter has DW_AT_abstract_origin that points to DIE with incompatible tag DW_TAG_formal_parameter
# DANGLING: DW_FORM_ref4 CU offset 0x12345678 is invalid

.ifndef CASE
.set CASE, 0
.endif

.section .text,"ax",@progbits
.Lcaller_begin:
callq .Lcallee_begin
.Lreturn_pc:
retq
.Lcaller_end:
.Lcallee_begin:
retq

.section .debug_abbrev,"",@progbits
.byte 1, 0x11, 1, 0, 0
# Subprogram with complete GNU call-site information.
.byte 2, 0x2e, 1
.uleb128 0x2117
.byte 0x19, 0x11, 0x01, 0x12, 0x06, 0, 0
.byte 3
.uleb128 0x4109
.byte 1, 0x31, 0x13, 0x11, 0x01, 0, 0
.byte 4
.if CASE == 3
.uleb128 0x49
.else
.uleb128 0x410a
.endif
.byte 0
.if CASE == 2
.byte 0x47
.else
.byte 0x31
.endif
.byte 0x13
.uleb128 0x2111
.byte 0x18, 0, 0
# Callee declaration and its formal parameter.
.byte 5, 0x2e, 1, 0x3f, 0x19, 0x3c, 0x19, 0, 0
.byte 6, 0x05, 0, 0x03, 0x08, 0, 0
.byte 7, 0x24, 0, 0x3e, 0x0b, 0x0b, 0x0b, 0, 0
.byte 0

.section .debug_info,"",@progbits
.Lcu:
.long .Lend-.Lheader
.Lheader:
.short 4
.long 0
.byte 8
.byte 1
.byte 2
.quad .Lcaller_begin
.long .Lcaller_end-.Lcaller_begin
.byte 3
.long .Lcallee-.Lcu
.quad .Lreturn_pc
.byte 4
.if CASE == 1
.long .Ltype-.Lcu
.elseif CASE == 4
.long 0x12345678
.else
.long .Lparameter-.Lcu
.endif
.byte 3, 0xf3, 1, 0x54
.byte 0, 0
.Lcallee:
.byte 5
.Lparameter:
.byte 6
.asciz "tstate"
.byte 0
.Ltype:
.byte 7, 0x07, 8
.byte 0
.Lend:
