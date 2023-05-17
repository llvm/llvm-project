; RUN: llc -march=hexagon  < %s | FileCheck %s

; This test checks that S2_tstbit_i instruction is generated
; and it does not assert.

; CHECK: p{{[0-9]+}} = tstbit


target triple = "hexagon-unknown-unknown-elf"

%struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192 = type { ptr, ptr }

@.str.8 = external dso_local unnamed_addr constant [5 x i8], align 1

declare dso_local void @panic(ptr, ...) local_unnamed_addr

define dso_local fastcc void @elv_rqhash_find() unnamed_addr {
entry:
  %cmd_flags = getelementptr inbounds %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192, ptr null, i32 -5
  %0 = load i64, ptr %cmd_flags, align 8
  %1 = and i64 %0, 4294967296
  %tobool10 = icmp eq i64 %1, 0
  br i1 %tobool10, label %do.body11, label %do.end14

do.body11:                                        ; preds = %entry
  tail call void (ptr, ...) @panic(ptr @.str.8) #1
  unreachable

do.end14:                                         ; preds = %entry
  %and.i = and i64 %0, -4294967297
  store i64 %and.i, ptr %cmd_flags, align 8
  ret void
}
