; RUN: opt %loadPolly -polly-detect-full-functions -polly-scop-inliner \
; RUN: -polly-scops -analyze < %s

; Check that we do not crash if there are declares. We should skip function
; declarations and not try to query for domtree.

declare void @foo()

