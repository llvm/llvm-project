; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-32R2,TAILCALL-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-64R2,TAILCALL-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips32r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-32R6,TAILCALL-32R6

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips64r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-64R6,TAILCALL-64R6

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips32r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-32R2,TAILCALL-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips64r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-64R2,TAILCALL-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mattr=+micromips -mcpu=mips32r2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-MM,TAILCALL-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mattr=+micromips -mcpu=mips32r6 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 < %s | FileCheck %s -check-prefixes=ALL,JALR-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 < %s | FileCheck %s -check-prefixes=ALL,JALR-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips32r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-32R6

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips64r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-64R6

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips32r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips64r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mattr=+micromips -mcpu=mips32r2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mattr=+micromips -mcpu=mips32r6 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mips-jalr-reloc=false < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static -mips-tail-calls=1 \
; RUN:     -O2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O0 -mips-jalr-reloc=false < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static -mips-tail-calls=1 \
; RUN:     -O0 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mips-jalr-reloc=false < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips64-linux-gnu -mips-tail-calls=1 \
; RUN:     -O2 -relocation-model=static < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mips-jalr-reloc=false < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static \
; RUN:     -O0 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,NORELOC

define internal void @foo() noinline {
entry:
  ret void
}

define void @checkCall() {
entry:
; ALL-LABEL: checkCall:
  call void @foo()
;	JALR-32R2: 	.reloc ([[TMPLABEL:.*]]), R_MIPS_JALR, foo
; JALR-32R2-NEXT: [[TMPLABEL]]:
;	JALR-32R2-NEXT: 	jalr	$25

;	JALR-64R2: 	.reloc [[TMPLABEL:.*]], R_MIPS_JALR, foo
; JALR-64R2-NEXT: [[TMPLABEL]]:
;	JALR-64R2-NEXT: 	jalr	$25

;	JALR-MM: 	.reloc ([[TMPLABEL:.*]]), R_MICROMIPS_JALR, foo
; JALR-MM-NEXT: [[TMPLABEL]]:
;	JALR-MM-NEXT: 	jalr	$25

;	JALR-32R6: 	.reloc ([[TMPLABEL:.*]]), R_MIPS_JALR, foo
; JALR-32R6-NEXT: [[TMPLABEL]]:
;	JALR-32R6-NEXT: 	jalrc	$25

;	JALR-64R6: 	.reloc [[TMPLABEL:.*]], R_MIPS_JALR, foo
; JALR-64R6-NEXT: [[TMPLABEL]]:
;	JALR-64R6-NEXT: 	jalrc	$25

; NORELOC-NOT: R_MIPS_JALR
 ret void
}

define void @checkTailCall() {
entry:
; ALL-LABEL: checkTailCall:
  tail call void @foo()
;	TAILCALL-32R2: 	.reloc ([[TMPLABEL:.*]]), R_MIPS_JALR, foo
; TAILCALL-32R2-NEXT: [[TMPLABEL]]:
;	TAILCALL-32R2-NEXT: 	jr	$25

;	TAILCALL-64R2: 	.reloc [[TMPLABEL:.*]], R_MIPS_JALR, foo
; TAILCALL-64R2-NEXT: [[TMPLABEL]]:
;	TAILCALL-64R2-NEXT: 	jr	$25

;	TAILCALL-MM: 	.reloc ([[TMPLABEL:.*]]), R_MICROMIPS_JALR, foo
; TAILCALL-MM-NEXT: [[TMPLABEL]]:
;	TAILCALL-MM-NEXT: 	jrc	$25

;	TAILCALL-32R6: 	.reloc ([[TMPLABEL:.*]]), R_MIPS_JALR, foo
; TAILCALL-32R6-NEXT: [[TMPLABEL]]:
;	TAILCALL-32R6-NEXT: 	jrc	$25

;	TAILCALL-64R6: 	.reloc [[TMPLABEL:.*]], R_MIPS_JALR, foo
; TAILCALL-64R6-NEXT: [[TMPLABEL]]:
;	TAILCALL-64R6-NEXT: 	jrc	$25

; NORELOC-NOT: R_MIPS_JALR
  ret void
}
