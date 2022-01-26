; ModuleID = 'test.cpp.o'
source_filename = "test.cpp"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @a()

!llvm.module.flags = !{!9, !39}

!9 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!39 = !{i32 5, !"CG Profile", !40}
!40 = !{!41}
!41 = distinct !{null, i32 ()* bitcast (void ()* @a to i32 ()*), i64 2594092}
