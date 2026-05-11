; RUN: opt %s -disable-output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

^0 = module: (path: "test.ll", hash: (0, 0, 0, 0, 0))
^1 = typeidCompatibleVTable: (name: "_ZTS1A", summary: ((offset: 16, ^0)))
