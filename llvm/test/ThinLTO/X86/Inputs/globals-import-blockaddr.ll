target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@label_addr = internal constant [1 x i8*] [i8* blockaddress(@foo, %lb)], align 8

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local [1 x i8*]* @foo() {
  br label %lb

lb:
  ret [1 x i8*]* @label_addr
}
