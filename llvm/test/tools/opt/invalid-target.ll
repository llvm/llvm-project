;; Check that invalid triples are handled correctly by opt.

;; No diagnostics should be printed for an explicitly/implicitly empty triple
; RUN: opt -S -passes=no-op-module -o /dev/null < %s 2>&1 | FileCheck %s --allow-empty --check-prefix=EMPTY
; RUN: opt '-mtriple=' -S -passes=no-op-module -o /dev/null < %s 2>&1 | FileCheck %s --allow-empty --check-prefix=EMPTY
; EMPTY-NOT: {{.+}}

;; Using "unknown" as the architecture is explicitly allowed (but warns)
; RUN: opt -mtriple=unknown -S -passes=no-op-module -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; UNKNOWN: warning: failed to infer data layout: unable to get target for 'unknown', see --version and --triple.

;; However, any other invalid target triple should cause the tool to fail:
; RUN: not opt -mtriple=invalid -S -passes=no-op-module -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=INVALID
; INVALID: warning: failed to infer data layout: unable to get target for 'invalid', see --version and --triple.
; INVALID-NEXT: unrecognized architecture 'invalid' provided.
; INVALID-EMPTY:
