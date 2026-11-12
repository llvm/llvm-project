;; Ensure we invoke the preinliner when feeding back a memprof profile.

;; The opt invocation will fail as the profdata file is empty, which is fine
;; since we are simply testing the pass pipeline below.
; RUN: not opt -debug-pass-manager -passes='default<O2>' -memory-profile-file=/dev/null %s 2>&1 | FileCheck %s

; CHECK: Running pass: InlinerPass on (foo)
; CHECK: Running pass: MemProfUsePass

define void @foo() {
  ret void
}
