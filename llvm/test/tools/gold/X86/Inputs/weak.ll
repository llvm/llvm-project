target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@a = weak global i32 41
@c = global i32* @a
