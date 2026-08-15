; RUN: llc -mtriple=aarch64-apple-macosx13.0.0 -filetype=obj %s -o %t.o
; RUN: llvm-nm -m %t.o | FileCheck --check-prefix=NM %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefix=SYMBOL %s

; llvm.used gives foo_internal N_NO_DEAD_STRIP.
; Under the name-flag rule, aliases must NOT inherit it.
@llvm.used = appending global [1 x ptr] [ptr @foo_internal], section "llvm.metadata"

define internal void @foo_internal() {
  ret void
}

; Weak alias to internal target: bar sets N_WEAK_DEF, foo has N_NO_DEAD_STRIP.
; Result: N_WEAK_DEF only (0x80), no N_NO_DEAD_STRIP.
@foo_default = weak_odr alias void (), ptr @foo_internal

; Weak hidden alias: bar sets N_WEAK_DEF + hidden.
@foo_hidden = weak_odr hidden alias void (), ptr @foo_internal

; Baseline: ordinary weak hidden definition.
define weak_odr hidden void @foo_defined() {
  ret void
}

; Weak aliasee target.
define weak_odr void @foo_weak_aliasee() {
  ret void
}

; Strong alias to weak target: bar has no weak, foo has N_WEAK_DEF.
; Result: no weak flags (0x0).
@foo_strong_alias = alias void (), ptr @foo_weak_aliasee

; Auto-hide weak: linkonce_odr unnamed_addr -> emit .weak_def_can_be_hidden.
; Result: N_WEAK_DEF | N_WEAK_REF (0xC0).
@foo_auto = linkonce_odr unnamed_addr alias void (), ptr @foo_internal

; NM-DAG: weak external _foo_default
; NM-DAG: weak private external _foo_hidden
; NM-DAG: weak private external _foo_defined
; NM-DAG: {{^[0-9a-fA-F]+ \(__TEXT,__text\) external _foo_strong_alias$}}
; NM-DAG: weak external automatically hidden _foo_auto
; NM-DAG: non-external [no dead strip] _foo_internal

; _foo_internal: N_NO_DEAD_STRIP from llvm.used (0x20).
; SYMBOL:      Name: _foo_internal
; SYMBOL-NEXT: Type: Section (0xE)
; SYMBOL-NEXT: Section: __text
; SYMBOL-NEXT: RefType: UndefinedNonLazy (0x0)
; SYMBOL-NEXT: Flags [ (0x20)
; SYMBOL-NEXT:   NoDeadStrip (0x20)
; SYMBOL-NEXT: ]

; _foo_auto: 0xC0 = N_WEAK_DEF | N_WEAK_REF.
; SYMBOL:      Name: _foo_auto
; SYMBOL-NEXT: Extern
; SYMBOL-NEXT: Type: Section (0xE)
; SYMBOL-NEXT: Section: __text
; SYMBOL-NEXT: RefType: UndefinedNonLazy (0x0)
; SYMBOL-NEXT: Flags [ (0xC0)
; SYMBOL-NEXT:   WeakDef (0x80)
; SYMBOL-NEXT:   WeakRef (0x40)
; SYMBOL-NEXT: ]

; _foo_default: 0x80 = N_WEAK_DEF, no N_NO_DEAD_STRIP inherited.
; SYMBOL:      Name: _foo_default
; SYMBOL-NEXT: Extern
; SYMBOL-NEXT: Type: Section (0xE)
; SYMBOL-NEXT: Section: __text
; SYMBOL-NEXT: RefType: UndefinedNonLazy (0x0)
; SYMBOL-NEXT: Flags [ (0x80)
; SYMBOL-NEXT:   WeakDef (0x80)
; SYMBOL-NEXT: ]

; _foo_strong_alias: 0x0, no weak inherited from foo.
; SYMBOL:      Name: _foo_strong_alias
; SYMBOL-NEXT: Extern
; SYMBOL-NEXT: Type: Section (0xE)
; SYMBOL-NEXT: Section: __text
; SYMBOL-NEXT: RefType: UndefinedNonLazy (0x0)
; SYMBOL-NEXT: Flags [ (0x0)
; SYMBOL-NEXT: ]