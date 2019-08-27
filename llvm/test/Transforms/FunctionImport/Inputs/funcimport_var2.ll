target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@link = internal global i32 0, align 4

; Function Attrs: norecurse nounwind readnone uwtable
define nonnull i32* @get_link() local_unnamed_addr {
  ret i32* @link
}

