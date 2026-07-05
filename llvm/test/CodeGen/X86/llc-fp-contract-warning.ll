; RUN: llc < %s -mtriple=x86_64-unknown-unknown --fp-contract=fast 2>&1 | grep "X86 backend ignores --fp-contract"

; RUN: llc < %s -mtriple=x86_64-unknown-unknown --fp-contract=off 2>&1 | grep "X86 backend ignores --fp-contract"

; on, as a default setting that's passed to backend when no --fp-contract option is specified, is not diagnosed.
; RUN: llc < %s -mtriple=x86_64-unknown-unknown --fp-contract=on 2>&1 | grep -v "X86 backend ignores --fp-contract"

define float @foo(float %f) {
  %res = fadd float %f, %f
  ret float %res
}

