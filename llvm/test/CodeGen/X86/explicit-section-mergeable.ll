; RUN: llc < %s -mtriple=x86_64 -unique-section-names=0 -data-sections 2>&1 \
; RUN:     | FileCheck %s

;; Several sections are created via inline assembly. We add checks
;; for these lines as we want to use --implicit-check-not to reduce the
;; number of checks in this file.
; CHECK: .section .asm_mergeable1,"aMS",@progbits,2{{$}}
; CHECK-NEXT: .section .asm_nonmergeable1,"a",@progbits{{$}}
; CHECK-NEXT: .section .asm_mergeable2,"aMS",@progbits,2{{$}}
; CHECK-NEXT: .section .asm_nonmergeable2,"a",@progbits{{$}}

;; Test implicit section assignment for symbols
; CHECK: .section .data,"aw",@progbits,unique,1
; CHECK: uniquified:

;; Create a uniquified symbol (as -unique-section-names=0) to test the uniqueID
;; interaction with mergeable symbols.
@uniquified = global i32 1

;; Test implicit section assignment for symbols to ensure that the symbols
;; have the expected properties.
; CHECK: .section .rodata,"a",@progbits,unique,2
; CHECK: implicit_nonmergeable:
; CHECK: .section .rodata.cst4,"aM",@progbits,4{{$}}
; CHECK: implicit_rodata_cst4:
; CHECK: .section .rodata.cst8,"aM",@progbits,8{{$}}
; CHECK: implicit_rodata_cst8:
; CHECK: .section .rodata.str4.4,"aMS",@progbits,4{{$}}
; CHECK: implicit_rodata_str4_4:

@implicit_nonmergeable  =              constant [2 x i16] [i16 1, i16 1]
@implicit_rodata_cst4   = unnamed_addr constant [2 x i16] [i16 1, i16 1]
@implicit_rodata_cst8   = unnamed_addr constant [2 x i32] [i32 1, i32 1]
@implicit_rodata_str4_4 = unnamed_addr constant [2 x i32] [i32 1, i32 0]

;; Basic checks that mergeable globals are placed into multiple distinct
;; sections with the same name and a compatible entry size.

; CHECK: .section .explicit_basic,"aM",@progbits,4,unique,3
; CHECK: explicit_basic_1:
; CHECK: explicit_basic_2:

;; Assign a mergeable global to a non-existing section.
@explicit_basic_1 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_basic"
;; Assign a compatible mergeable global to the previous section.
@explicit_basic_2 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_basic"

; CHECK: .section .explicit_basic,"aM",@progbits,8,unique,4
; CHECK: explicit_basic_3:
; CHECK: explicit_basic_4:

;; Assign a symbol with an incompatible entsize (different size) to a section with the same name.
@explicit_basic_3 = unnamed_addr constant [2 x i32] [i32 1, i32 1], section ".explicit_basic"
;; Assign a compatible mergeable global to the previous section.
@explicit_basic_4 = unnamed_addr constant [2 x i32] [i32 1, i32 1], section ".explicit_basic"

; CHECK: .section .explicit_basic,"aMS",@progbits,4,unique,5
; CHECK: explicit_basic_5:
; CHECK: explicit_basic_6:

;; Assign a symbol with an incompatible entsize (string vs non-string) to a section with the same name.
@explicit_basic_5 = unnamed_addr constant [2 x i32] [i32 1, i32 0], section ".explicit_basic"
;; Assign a compatible mergeable global to the previous section.
@explicit_basic_6 = unnamed_addr constant [2 x i32] [i32 1, i32 0], section ".explicit_basic"

; CHECK: .section .explicit_basic,"a",@progbits{{$}}
; CHECK: explicit_basic_7:

;; Assign a symbol with an incompatible entsize (non-mergeable) to a mergeable section created explicitly.
@explicit_basic_7 = constant [2 x i16] [i16 1, i16 1], section ".explicit_basic"

; CHECK: .section .explicit_initially_nonmergeable,"a",@progbits{{$}}
; CHECK: explicit_basic_8:
; CHECK: .section .explicit_initially_nonmergeable,"aM",@progbits,4,unique,6
; CHECK: explicit_basic_9:

;; Assign a mergeble symbol to a section that initially had a non-mergeable symbol explicitly assigned to it.
@explicit_basic_8 = constant [2 x i16] [i16 1, i16 1], section ".explicit_initially_nonmergeable"
@explicit_basic_9 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_initially_nonmergeable"

; CHECK: .section .explicit_initially_nonmergeable,"a",@progbits{{$}}
; CHECK: explicit_basic_10:
; CHECK: .section .explicit_initially_nonmergeable,"aM",@progbits,4,unique,6
; CHECK: explicit_basic_11:

;; Assign compatible globals to the previously created sections.
@explicit_basic_10 = constant [2 x i16] [i16 1, i16 1], section ".explicit_initially_nonmergeable"
@explicit_basic_11 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_initially_nonmergeable"

;; Check that mergeable symbols can be explicitly assigned to "default" sections.

; CHECK: .section .rodata.cst16,"a",@progbits,unique,7
; CHECK: explicit_default_1:

;; Assign an incompatible (non-mergeable) symbol to a "default" mergeable section.
@explicit_default_1 = constant [2 x i64] [i64 1, i64 1], section ".rodata.cst16"

; CHECK: .section .rodata.cst16,"aM",@progbits,16{{$}}
; CHECK: explicit_default_2:

;; Assign a compatible global to a "default" mergeable section.
@explicit_default_2 = unnamed_addr constant [2 x i64] [i64 1, i64 1], section ".rodata.cst16"

; CHECK: .section .debug_str,"aMS",@progbits,1,unique,[[#U:8]]
; CHECK: explicit_default_3:

;; Non-allocatable "default" sections can be re-emitted with allocatable flag and uniqued
@explicit_default_3 = unnamed_addr constant [2 x i8] [i8 1, i8 0], section ".debug_str"

; CHECK: .section .debug_str,"a",@progbits,unique,[[#U+1]]
; CHECK: explicit_default_4:

;; Non-allocatable "default" sections cannot have allocatable mergeable symbols with incompatible (non-mergeable) entry sizes assigned to them.
@explicit_default_4 = constant [2 x i16] [i16 1, i16 1], section ".debug_str"

;; Test implicit section assignment for globals with associated globals.
; CHECK: .section .rodata.cst4,"aMo",@progbits,4,implicit_rodata_cst4,unique,[[#U+2]]
; CHECK: implicit_rodata_cst4_assoc:
; CHECK: .section .rodata.cst8,"aMo",@progbits,8,implicit_rodata_cst4,unique,[[#U+3]]
; CHECK: implicit_rodata_cst8_assoc:

@implicit_rodata_cst4_assoc = unnamed_addr constant [2 x i16] [i16 1, i16 1], !associated !4
@implicit_rodata_cst8_assoc = unnamed_addr constant [2 x i32] [i32 1, i32 1], !associated !4

;; Check that globals with associated globals that are explicitly assigned
;; to a section have been placed into distinct sections with the same name, but
;; different entry sizes.
; CHECK: .section .explicit,"aMo",@progbits,4,implicit_rodata_cst4,unique,[[#U+4]]
; CHECK: explicit_assoc_1:
; CHECK: .section .explicit,"aMo",@progbits,4,implicit_rodata_cst4,unique,[[#U+5]]
; CHECK: explicit_assoc_2:
; CHECK: .section .explicit,"aMo",@progbits,8,implicit_rodata_cst4,unique,[[#U+6]]
; CHECK: explicit_assoc_3:

@explicit_assoc_1 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit", !associated !4
@explicit_assoc_2 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit", !associated !4
@explicit_assoc_3 = unnamed_addr constant [2 x i32] [i32 1, i32 1], section ".explicit", !associated !4

!4 = !{ptr @implicit_rodata_cst4}

;; Test implicit section assignment for globals in distinct comdat groups.
; CHECK: .section .rodata.cst4,"aGM",@progbits,4,f,comdat,unique,[[#U+7]]
; CHECK: implicit_rodata_cst4_comdat:
; CHECK: .section .rodata.cst8,"aGM",@progbits,8,g,comdat,unique,[[#U+8]]
; CHECK: implicit_rodata_cst8_comdat:

;; Check that globals in distinct comdat groups that are explicitly assigned
;; to a section have been placed into distinct sections with the same name, but
;; different entry sizes. Due to the way that MC currently works the unique ID
;; does not have any effect here, although it appears in the assembly. The unique ID's
;; appear incorrect as comdats are not taken into account when looking up the unique ID
;; for a mergeable section. However, as they have no effect it doesn't matter that they
;; are incorrect.
; CHECK: .section .explicit_comdat_distinct,"aM",@progbits,4,unique,[[#U+9]]
; CHECK: explicit_comdat_distinct_supply_uid:
; CHECK: .section .explicit_comdat_distinct,"aGM",@progbits,4,f,comdat,unique,[[#U+10]]
; CHECK: explicit_comdat_distinct1:
; CHECK: .section .explicit_comdat_distinct,"aGM",@progbits,4,g,comdat,unique,[[#U+10]]
; CHECK: explicit_comdat_distinct2:
; CHECK: .section .explicit_comdat_distinct,"aGM",@progbits,8,h,comdat,unique,[[#U+11]]
; CHECK: explicit_comdat_distinct3:

$f = comdat any
$g = comdat any
$h = comdat any

@implicit_rodata_cst4_comdat = unnamed_addr constant [2 x i16] [i16 1, i16 1], comdat($f)
@implicit_rodata_cst8_comdat = unnamed_addr constant [2 x i32] [i32 1, i32 1], comdat($g)

@explicit_comdat_distinct_supply_uid = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_distinct"
@explicit_comdat_distinct1 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_distinct", comdat($f)
@explicit_comdat_distinct2 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_distinct", comdat($g)
@explicit_comdat_distinct3 = unnamed_addr constant [2 x i32] [i32 1, i32 1], section ".explicit_comdat_distinct", comdat($h)

;; Test implicit section assignment for globals in the same comdat group.
; CHECK: .section .rodata.cst4,"aGM",@progbits,4,i,comdat,unique,[[#U+12]]
; CHECK: implicit_rodata_cst4_same_comdat:
; CHECK: .section .rodata.cst8,"aGM",@progbits,8,i,comdat,unique,[[#U+13]]
; CHECK: implicit_rodata_cst8_same_comdat:

;; Check that globals in the same comdat group that are explicitly assigned
;; to a section have been placed into distinct sections with the same name, but
;; different entry sizes. Due to the way that MC currently works the unique ID
;; does not have any effect here, although it appears in the assembly. The unique ID's
;; appear incorrect as comdats are not taken into account when looking up the unique ID
;; for a mergeable section. However, as they have no effect it doesn't matter that they
;; are incorrect.
; CHECK: .section .explicit_comdat_same,"aM",@progbits,4,unique,[[#U+14]]
; CHECK: explicit_comdat_same_supply_uid:
; CHECK: .section .explicit_comdat_same,"aGM",@progbits,4,i,comdat,unique,[[#U+15]]
; CHECK: explicit_comdat_same1:
; CHECK: explicit_comdat_same2:
; CHECK: .section .explicit_comdat_same,"aGM",@progbits,8,i,comdat,unique,[[#U+16]]
; CHECK: explicit_comdat_same3:

$i = comdat any

@implicit_rodata_cst4_same_comdat = unnamed_addr constant [2 x i16] [i16 1, i16 1], comdat($i)
@implicit_rodata_cst8_same_comdat = unnamed_addr constant [2 x i32] [i32 1, i32 1], comdat($i)

@explicit_comdat_same_supply_uid = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_same"
@explicit_comdat_same1 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_same", comdat($i)
@explicit_comdat_same2 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit_comdat_same", comdat($i)
@explicit_comdat_same3 = unnamed_addr constant [2 x i32] [i32 1, i32 1], section ".explicit_comdat_same", comdat($i)

;; Check interaction between symbols that are explicitly assigned
;; to a section and implicitly assigned symbols.

; CHECK: .section .rodata.str1.1,"aMS",@progbits,1{{$}}
; CHECK: implicit_rodata_str1_1:
; CHECK: explicit_implicit_1:

;; Assign a compatible global to an existing mergeable section created implicitly.
@implicit_rodata_str1_1 = unnamed_addr constant [2 x i8] [i8 1, i8 0]
@explicit_implicit_1 = unnamed_addr constant [2 x i8] [i8 1, i8 0], section ".rodata.str1.1"

; CHECK: .section .rodata.str1.1,"a",@progbits,unique,[[#U+17]]
; CHECK: explicit_implicit_2:

;; Assign an incompatible symbol (non-mergeable) to an existing mergeable section created implicitly.
@explicit_implicit_2 = constant [2 x i16] [i16 1, i16 1], section ".rodata.str1.1"

; CHECK: .section .rodata.str1.1,"aMS",@progbits,1{{$}}
; CHECK: explicit_implicit_3:
; CHECK: .section .rodata.str1.1,"a",@progbits,unique,[[#U+17]]
; CHECK: explicit_implicit_4:

;; Assign compatible globals to the previously created sections.
@explicit_implicit_3 = unnamed_addr constant [2 x i8] [i8 1, i8 0], section ".rodata.str1.1"
@explicit_implicit_4 = constant [2 x i16] [i16 1, i16 1], section ".rodata.str1.1"

; CHECK: .section .rodata.str2.2,"aMS",@progbits,2{{$}}
; CHECK: explicit_implicit_5:
; CHECK: implicit_rodata_str2_2:

;; Implicitly assign a compatible global to an existing mergeable section created explicitly.
@explicit_implicit_5 = unnamed_addr constant [2 x i16] [i16 1, i16 0], section ".rodata.str2.2"
@implicit_rodata_str2_2 = unnamed_addr constant [2 x i16] [i16 1, i16 0]

;; Check the interaction with inline asm.

; CHECK: .section .asm_mergeable1,"aMS",@progbits,2{{$}}
; CHECK: explicit_asm_1:
; CHECK: .section .asm_nonmergeable1,"a",@progbits{{$}}
; CHECK: explicit_asm_2:
; CHECK: .section .asm_mergeable1,"aM",@progbits,4,unique,[[#U+18]]
; CHECK: explicit_asm_3:
; CHECK: .section .asm_nonmergeable1,"aMS",@progbits,2,unique,[[#U+19]]
; CHECK: explicit_asm_4:
; CHECK: .section .asm_mergeable2,"aM",@progbits,4,unique,[[#U+20]]
; CHECK: explicit_asm_5:
; CHECK: .section .asm_nonmergeable2,"aMS",@progbits,2,unique,[[#U+21]]
; CHECK: explicit_asm_6:
; CHECK: .section .asm_mergeable2,"aMS",@progbits,2{{$}}
; CHECK: explicit_asm_7:
; CHECK: .section .asm_nonmergeable2,"a",@progbits{{$}}
; CHECK: explicit_asm_8:

module asm ".section .asm_mergeable1,\22aMS\22,@progbits,2"
module asm ".section .asm_nonmergeable1,\22a\22,@progbits"
module asm ".section .asm_mergeable2,\22aMS\22,@progbits,2"
module asm ".section .asm_nonmergeable2,\22a\22,@progbits"

;; Assign compatible symbols to sections created using inline asm.
@explicit_asm_1 = unnamed_addr constant [2 x i16] [i16 1, i16 0], section ".asm_mergeable1"
@explicit_asm_2 = constant [2 x i16] [i16 1, i16 0], section ".asm_nonmergeable1"
;; Assign incompatible globals to the same sections.
@explicit_asm_3 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".asm_mergeable1"
@explicit_asm_4 = unnamed_addr constant [2 x i16] [i16 1, i16 0], section ".asm_nonmergeable1"

;; Assign incompatible globals to sections created using inline asm.
@explicit_asm_5 = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".asm_mergeable2"
@explicit_asm_6 = unnamed_addr constant [2 x i16] [i16 1, i16 0], section ".asm_nonmergeable2"
;; Assign compatible globals to the same sections.
@explicit_asm_7 = unnamed_addr constant [2 x i16] [i16 1, i16 0], section ".asm_mergeable2"
@explicit_asm_8 = constant [2 x i16] [i16 1, i16 0], section ".asm_nonmergeable2"

;; A .note.GNU-stack section is created implicitly. We add a check for this as we want to use
;; --implicit-check-not to reduce the number of checks in this file.
; CHECK: .section ".note.GNU-stack","",@progbits{{$}}

;; --no-integrated-as avoids the use of ",unique," for compatibility with older binutils.

;; Error if an incompatible symbol is explicitly placed into a mergeable section.
; RUN: not llc < %s -mtriple=x86_64 --no-integrated-as -binutils-version=2.34 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NO-I-AS-ERR
; NO-I-AS-ERR: error: Symbol 'explicit_default_1' from module '<stdin>' required a section with entry-size=0 but was placed in section '.rodata.cst16' with entry-size=16: Explicit assignment by pragma or attribute of an incompatible symbol to this section?
; NO-I-AS-ERR: error: Symbol 'explicit_default_4' from module '<stdin>' required a section with entry-size=0 but was placed in section '.debug_str' with entry-size=1: Explicit assignment by pragma or attribute of an incompatible symbol to this section?
; NO-I-AS-ERR: error: Symbol 'explicit_implicit_2' from module '<stdin>' required a section with entry-size=0 but was placed in section '.rodata.str1.1' with entry-size=1: Explicit assignment by pragma or attribute of an incompatible symbol to this section?
; NO-I-AS-ERR: error: Symbol 'explicit_implicit_4' from module '<stdin>' required a section with entry-size=0 but was placed in section '.rodata.str1.1' with entry-size=1: Explicit assignment by pragma or attribute of an incompatible symbol to this section?

;; For GNU as before 2.35,
;; Don't create mergeable sections for globals with an explicit section name.
; RUN: echo '@explicit = unnamed_addr constant [2 x i16] [i16 1, i16 1], section ".explicit"' > %t.no_i_as.ll
; RUN: llc < %t.no_i_as.ll -mtriple=x86_64 --no-integrated-as -binutils-version=2.34 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NO-I-AS-OLD
; NO-I-AS-OLD: .section .explicit,"a",@progbits{{$}}
; RUN: llc < %t.no_i_as.ll -mtriple=x86_64 --no-integrated-as -binutils-version=2.35 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NO-I-AS-NEW
; RUN: llc < %t.no_i_as.ll -mtriple=x86_64 --no-integrated-as -binutils-version=none 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NO-I-AS-NEW
; NO-I-AS-NEW: .section .explicit,"aM",@progbits,4,unique,1
