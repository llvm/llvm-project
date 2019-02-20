target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$comdat.any = comdat any
@g = global i32 42, comdat($comdat.any)
