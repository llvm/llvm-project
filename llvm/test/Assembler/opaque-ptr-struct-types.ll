; RUN: opt -S -opaque-pointers < %s | opt -S -opaque-pointers | FileCheck %s

; CHECK: %T1 = type { i8 }
; CHECK: %T2 = type { i8 }
; CHECK: %T3 = type { i8 }
; CHECK: %T4 = type { i8 }
; CHECK: %T5 = type { i8 }
; CHECK: %T6 = type { i8 }
; CHECK: %T7 = type { i8 }

%T1 = type { i8 }
%T2 = type { i8 }
%T3 = type { i8 }
%T4 = type { i8 }
%T5 = type { i8 }
%T6 = type { i8 }
%T7 = type { i8 }

@g = external global %T1

define %T2 @f(ptr %p) {
  alloca %T3
  getelementptr %T4, ptr %p, i64 1
  call void @f(ptr sret(%T5) %p)
  store ptr getelementptr (%T6, ptr @g, i64 1), ptr %p
  unreachable
}

declare void @f2(ptr sret(%T7))
