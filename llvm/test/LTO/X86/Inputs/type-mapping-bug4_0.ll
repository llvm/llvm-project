target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.CWBD = type { float }
%"class.std::_Unique_ptr_base" = type { %class.CWBD* }

%class.CB = type opaque

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
