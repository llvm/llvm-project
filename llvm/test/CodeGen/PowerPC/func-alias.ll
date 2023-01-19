; RUN: llc -opaque-pointers=0 -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefix=P9
; RUN: llc -opaque-pointers=0 -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefix=P10

@newname = dso_local alias i32 (...), bitcast (i32 ()* @oldname to i32 (...)*)

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @oldname() #0 {
entry:
  ret i32 55
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @caller() #0 {
; #P9-LABEL: caller
; #P9:       bl newname
; #P9-NOT:   nop
; #P9:       blr
; #P10-LABEL: caller
; #P10:       bl newname@notoc
; #P10-NOT:   nop
; #P10:       blr
entry:
  %call = call signext i32 bitcast (i32 (...)* @newname to i32 ()*)()
  ret i32 %call
}

; Function Attrs: noinline nounwind optnone -pcrelative-memops
; This caller does not use PC Relative memops
define dso_local signext i32 @caller_nopcrel() #1 {
; #P9-LABEL: caller_nopcrel
; #P9:       bl newname
; #P9-NOT:   nop
; #P9:       blr
; #P10-LABEL: caller_nopcrel
; #P10:       bl newname
; #P10-NEXT:  nop
; #P10:       blr
entry:
  %call = call signext i32 bitcast (i32 (...)* @newname to i32 ()*)()
  ret i32 %call
}

attributes #0 = { noinline nounwind optnone }
attributes #1 = { noinline nounwind optnone "target-features"="-pcrelative-memops" }
