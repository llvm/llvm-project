; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=CHECK1 %s

target triple = "x86_64-unknown-linux-gnu"

define hidden void @Func() !type !0 {
  ret void
}

; CHECK1: !aliases = !{![[A1:[0-9]+]], ![[A2:[0-9]+]], ![[A3:[0-9]+]]}

; CHECK1: ![[A1]] = !{!"Alias", !"Func", i8 1, i8 0}
; CHECK1: ![[A2]] = !{!"Hidden_Alias", !"Func", i8 1, i8 0}
; CHECK1: ![[A3]] = !{!"Weak_Alias", !"Func", i8 0, i8 1}
@Alias = hidden alias void (), void ()* @Func
@Hidden_Alias = hidden alias void (), void ()* @Func
@Weak_Alias = weak alias void (), void ()* @Func

@Variable = global i32 0

; Only generate summary alias information for aliases to functions
; CHECK1-NOT: Variable_Alias
@Variable_Alias = alias i32, i32* @Variable

!0 = !{i64 0, !"_ZTSFvvE"}
