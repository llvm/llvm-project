; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R2,TAILCALL-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R2,TAILCALL-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips32r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R6,TAILCALL-32R6

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips64r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R6,TAILCALL-64R6

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips32r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R2,TAILCALL-32R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mcpu=mips64r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R2,TAILCALL-64R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mattr=+micromips -mcpu=mips32r2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-MM,TAILCALL-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic -mips-tail-calls=1 \
; RUN:     -O2 -mattr=+micromips -mcpu=mips32r6 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-MM,TAILCALL-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 < %s | FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R2,PIC-NOTAILCALL-R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 < %s | FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R2,PIC-NOTAILCALL-R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips32r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R6,PIC-NOTAILCALL-R6

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips64r6 -mips-compact-branches=always < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R6,PIC-NOTAILCALL-R6

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips32r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-32,JALR-32R2,PIC-NOTAILCALL-R2

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mcpu=mips64r6 -mips-compact-branches=never < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-64,JALR-64R2,PIC-NOTAILCALL-R2

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mattr=+micromips -mcpu=mips32r2 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-MM,PIC-NOTAILCALL-MM

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=pic \
; RUN:     -O0 -mattr=+micromips -mcpu=mips32r6 < %s | \
; RUN:     FileCheck %s -check-prefixes=ALL,JALR-ALL,JALR-MM,PIC-NOTAILCALL-MM

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
; ALL-NOT: MIPS_JALR
  call void @foo()
; JALR-32:       .reloc ([[TMPLABEL:\$.+]]), R_MIPS_JALR, foo
; JALR-64:       .reloc [[TMPLABEL:\..+]], R_MIPS_JALR, foo
; JALR-MM:       .reloc ([[TMPLABEL:\$.+]]), R_MICROMIPS_JALR, foo
; NORELOC-NOT:   .reloc
; JALR-ALL-NEXT: [[TMPLABEL]]:
; JALR-32R2-NEXT: 	jalr	$25
; JALR-64R2-NEXT: 	jalr	$25
; JALR-32R6-NEXT: 	jalrc	$25
; JALR-64R6-NEXT: 	jalrc	$25
; JALR-MM-NEXT: 	jalr	$25
; ALL-NOT: MIPS_JALR
 ret void
}

define void @checkTailCall() {
entry:
; ALL-LABEL: checkTailCall:
; ALL-NOT: MIPS_JALR
  tail call void @foo()
; JALR-32:       .reloc ([[TMPLABEL:\$.+]]), R_MIPS_JALR, foo
; JALR-64:       .reloc [[TMPLABEL:\..+]], R_MIPS_JALR, foo
; JALR-MM:       .reloc ([[TMPLABEL:\$.+]]), R_MICROMIPS_JALR, foo
; JALR-ALL-NEXT: [[TMPLABEL]]:
; NORELOC-NOT:   .reloc
; TAILCALL-32R2-NEXT: 	jr	$25
; TAILCALL-64R2-NEXT: 	jr	$25
; TAILCALL-MM-NEXT: 	jrc	$25
; TAILCALL-32R6-NEXT: 	jrc	$25
; TAILCALL-64R6-NEXT: 	jrc	$25
; PIC-NOTAILCALL-R2-NEXT: 	jalr	$25
; PIC-NOTAILCALL-R6-NEXT: 	jalrc	$25
; PIC-NOTAILCALL-MM-NEXT: 	jalr	$25
; ALL-NOT: MIPS_JALR
  ret void
}

; Check that we don't emit R_MIPS_JALR relocations against function pointers.
; This resulted in run-time crashes until lld was modified to ignore
; R_MIPS_JALR relocations against data symbols (commit 5bab291b7b).
; However, the better approach is to not emit these relocations in the first
; place so check that we no longer emit them.
; Previously we were adding them for local dynamic TLS function pointers and
; function pointers with internal linkage.

@fnptr_internal = internal global ptr @checkFunctionPointerCall
@fnptr_internal_const = internal constant ptr @checkFunctionPointerCall
@fnptr_const = constant ptr @checkFunctionPointerCall
@fnptr_global = global ptr @checkFunctionPointerCall

define void @checkFunctionPointerCall() {
entry:
; ALL-LABEL: checkFunctionPointerCall:
; ALL-NOT: MIPS_JALR
  %func_internal = load ptr, ptr @fnptr_internal
  call void %func_internal()
  %func_internal_const = load ptr, ptr @fnptr_internal_const
  call void %func_internal_const()
  %func_const = load ptr, ptr @fnptr_const
  call void %func_const()
  %func_global = load ptr, ptr @fnptr_global
  call void %func_global()
  ret void
}

@tls_fnptr_gd = thread_local global ptr @checkTlsFunctionPointerCall
@tls_fnptr_ld = thread_local(localdynamic) global ptr @checkTlsFunctionPointerCall
@tls_fnptr_ie = thread_local(initialexec) global ptr @checkTlsFunctionPointerCall
@tls_fnptr_le = thread_local(localexec) global ptr @checkTlsFunctionPointerCall

define void @checkTlsFunctionPointerCall() {
entry:
; There should not be any *JALR relocations in this function other than the
; calls to __tls_get_addr:
; ALL-LABEL: checkTlsFunctionPointerCall:
; ALL-NOT: MIPS_JALR
; JALR-ALL: .reloc {{.+}}MIPS_JALR, __tls_get_addr
; ALL-NOT: MIPS_JALR
; JALR-ALL: .reloc {{.+}}MIPS_JALR, __tls_get_addr
; NORELOC-NOT:   .reloc
; ALL-NOT: _MIPS_JALR
  %func_gd = load ptr, ptr @tls_fnptr_gd
  call void %func_gd()
  %func_ld = load ptr, ptr @tls_fnptr_ld
  call void %func_ld()
  %func_ie = load ptr, ptr @tls_fnptr_ie
  call void %func_ie()
  %func_le = load ptr, ptr @tls_fnptr_le
  call void %func_le()
  ret void
}
