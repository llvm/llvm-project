; RUN: opt -passes='default<O2>' -S %s | FileCheck %s
;
; Stores to aggregate globals that match the initializer slice at a constant
; offset should be treated as InitializerStored, allowing GlobalOpt to mark
; the global constant and fold dead control flow.

%struct.i8042_port = type { ptr, i32, i8, i8, i8 }

@i8042_ports = internal unnamed_addr global [6 x %struct.i8042_port] zeroinitializer

define void @i8042_remove() local_unnamed_addr {
  br label %1

1:
  %2 = phi i64 [ 0, %0 ], [ %8, %7 ]
  %3 = getelementptr [6 x %struct.i8042_port], ptr @i8042_ports, i64 0, i64 %2
  %4 = load ptr, ptr %3, align 16
  %5 = icmp eq ptr %4, null
  br i1 %5, label %7, label %6

6:
  store ptr null, ptr @i8042_ports, align 16
  br label %7

7:
  %8 = add nuw nsw i64 %2, 1
  %9 = icmp eq i64 %8, 6
  br i1 %9, label %10, label %1

10:
  ret void
}

; CHECK-LABEL: define void @i8042_remove
; CHECK-NOT: br label
; CHECK: ret void
