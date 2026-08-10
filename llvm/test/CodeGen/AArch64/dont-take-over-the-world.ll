; RUN: not llc -mtriple=x86-64 2>&1 | FileCheck %s

; To support "arm64" as a -march option, we need to register a second AArch64
; target, but we have to be careful how we do that so that it doesn't become the
; target of last resort when the specified triple is completely wrong.

; CHECK: unable to get target for 'x86-64', see --version and --triple.
