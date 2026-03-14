target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

define void @Callee(i8* %p) #0 {
entry:
  ret void
}

define void @Callee2(i32 %x, i8* %p) #0 {
entry:
  ret void
}

