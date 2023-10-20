; RUN: opt -S -passes=globaldce %s | FileCheck %s

@conditional_gv = internal unnamed_addr constant i64 42
@non_conditional_gv = internal unnamed_addr constant i64 42

declare external void @some_externally_defined_symbol()

@llvm.used = appending global [2 x i8*] [
    i8* bitcast (i64* @conditional_gv to i8*),
    i8* bitcast (i64* @non_conditional_gv to i8*)
], section "llvm.metadata"

!1 = !{i64* @conditional_gv, i32 0, !{void()* @some_externally_defined_symbol}}
!llvm.used.conditional = !{!1}

; CHECK-DAG: @conditional_gv
; CHECK-DAG: @llvm.used = appending global [2 x ptr] [ptr @conditional_gv, ptr @non_conditional_gv], section "llvm.metadata"

