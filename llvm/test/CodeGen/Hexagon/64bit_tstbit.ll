; RUN: llc -march=hexagon  < %s | FileCheck %s

; This test checks that S2_tstbit_i instruction is generated
; and it does not assert.

; CHECK: p{{[0-9]+}} = tstbit


target triple = "hexagon-unknown-unknown-elf"

%struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192 = type { %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192*, %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192** }

@.str.8 = external dso_local unnamed_addr constant [5 x i8], align 1

declare dso_local void @panic(i8*, ...) local_unnamed_addr

define dso_local fastcc void @elv_rqhash_find() unnamed_addr {
entry:
  %cmd_flags = getelementptr inbounds %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192, %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192* null, i32 -5
  %0 = bitcast %struct.hlist_node.45.966.3115.3729.4036.4650.4957.6492.6799.7413.7720.9562.10790.11097.11404.11711.14474.17192* %cmd_flags to i64*
  %1 = load i64, i64* %0, align 8
  %2 = and i64 %1, 4294967296
  %tobool10 = icmp eq i64 %2, 0
  br i1 %tobool10, label %do.body11, label %do.end14

do.body11:                                        ; preds = %entry
  tail call void (i8*, ...) @panic(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.8, i32 0, i32 0)) #1
  unreachable

do.end14:                                         ; preds = %entry
  %and.i = and i64 %1, -4294967297
  store i64 %and.i, i64* %0, align 8
  ret void
}
