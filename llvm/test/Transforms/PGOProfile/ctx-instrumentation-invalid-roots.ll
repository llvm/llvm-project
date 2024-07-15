; RUN: not opt -passes=pgo-instr-gen,ctx-instr-lower -profile-context-root=good \
; RUN:   -profile-context-root=bad \
; RUN:   -S < %s 2>&1 | FileCheck %s

declare void @foo()

define void @good() {
  call void @foo()
  ret void
}

define void @bad() {
  musttail call void @foo()
  ret void
}

; CHECK: error: The function bad was indicated as a context root, but it features musttail calls, which is not supported.
