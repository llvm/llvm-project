; Disabling output means we'll just skip the summary entries, which is the code
; path we're trying to test. There's no output to check against, so we have no
; CHECKs.
;
; RUN: opt %s -disable-output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

^0 = module: (path: "thinlto-vtable-skip.ll", hash: (0, 0, 0, 0, 0))
^1 = typeidCompatibleVTable: (name: "_ZTS1A", summary: ((offset: 16, ^0)))
