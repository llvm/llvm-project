; RUN: opt -S < %s | opt -S | FileCheck %s

; CHECK-DAG: %T1 = type { i8 }
; CHECK-DAG: %T2 = type { i8 }
; CHECK-DAG: %T3 = type { i8 }
; CHECK-DAG: %T4 = type { i8 }
; CHECK-DAG: %T5 = type { i8 }
; CHECK-DAG: %T6 = type { i8 }
; CHECK-DAG: %T7 = type { i8 }
; CHECK-DAG: %T8 = type { i8 }

%T1 = type { i8 }
%T2 = type { i8 }
%T3 = type { i8 }
%T4 = type { i8 }
%T5 = type { i8 }
%T6 = type { i8 }
%T7 = type { i8 }
%T8 = type { i8 }

@g = external global %T1

@g.ifunc = ifunc %T8 (), ptr @f.resolver

define %T2 @f(ptr %p) {
  alloca %T3
  getelementptr %T4, ptr %p, i64 1
  call void @f(ptr sret(%T5) %p)
  store ptr getelementptr (%T6, ptr @g, i64 1), ptr %p
  unreachable
}

define ptr @f.resolver(ptr %p) {
  alloca %T3
  getelementptr %T4, ptr %p, i64 1
  call void @f(ptr sret(%T5) %p)
  store ptr getelementptr (%T6, ptr @g, i64 1), ptr %p
  unreachable
}

declare void @f2(ptr sret(%T7))
