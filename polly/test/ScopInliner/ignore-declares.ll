; RUN: opt %loadNPMPolly -polly-detect-full-functions '-passes=cgscc(polly-inline),function(print<polly-function-scops>)' -disable-output < %s

; Check that we do not crash if there are declares. We should skip function
; declarations and not try to query for domtree.

declare void @foo()

