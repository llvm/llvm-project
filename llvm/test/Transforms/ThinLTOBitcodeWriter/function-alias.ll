; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=CHECK1 %s

target triple = "x86_64-unknown-linux-gnu"

define hidden void @Func() !type !0 {
  ret void
}

; CHECK1: !cfi.functions = !{![[F1:[0-9]+]], ![[F2:[0-9]+]], ![[F3:[0-9]+]], ![[F4:[0-9]+]]}
; CHECK1: !aliases = !{![[A:[0-9]+]]}

; CHECK1: ![[F1]] = !{!"Func", i8 0, ![[T:[0-9]+]]}
; CHECK1: ![[T]] = !{i64 0, !"_ZTSFvvE"}
; CHECK1: ![[F2]] = !{!"Alias", i8 0, ![[T]]}
; CHECK1: ![[F3]] = !{!"Hidden_Alias", i8 0, ![[T]]}
; CHECK1: ![[F4]] = !{!"Weak_Alias", i8 0, ![[T]]}
; 
; CHECK1: ![[A]] = !{!"Func", !"Alias", !"Hidden_Alias", !"Weak_Alias"}
@Alias = hidden alias void (), ptr @Func
@Hidden_Alias = hidden alias void (), ptr @Func
@Weak_Alias = weak alias void (), ptr @Func

@Variable = global i32 0

; Only generate summary alias information for aliases to functions
; CHECK1-NOT: Variable_Alias
@Variable_Alias = alias i32, ptr @Variable

!0 = !{i64 0, !"_ZTSFvvE"}
