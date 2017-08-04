; RUN: opt < %s -inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:     -pass-remarks-analysis=inline -pass-remarks-with-hotness -S 2>&1 \
; RUN:     | FileCheck %s

; CHECK: foo should always be inlined (cost=always) (hotness: 30)
; CHECK: foo inlined into bar (hotness: 30)
; CHECK: foz not inlined into bar because it should never be inlined (cost=never) (hotness: 30)

; Function Attrs: alwaysinline nounwind uwtable
define i32 @foo() #0 !prof !1 {
entry:
  ret i32 4
}

; Function Attrs: noinline nounwind uwtable
define i32 @foz() #1 !prof !2 {
entry:
  ret i32 2
}

; Function Attrs: nounwind uwtable
define i32 @bar() !prof !3 {
entry:
  %call = call i32 @foo()
  %call2 = call i32 @foz()
  %mul = mul i32 %call, %call2
  ret i32 %mul
}

attributes #0 = { alwaysinline }
attributes #1 = { noinline }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}
!1 = !{!"function_entry_count", i64 10}
!2 = !{!"function_entry_count", i64 20}
!3 = !{!"function_entry_count", i64 30}
