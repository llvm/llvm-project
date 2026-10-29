; RUN: opt < %s -passes=instrprof -instrprof-atomic-counter-update-all -do-counter-promotion -S | FileCheck %s

; Test that promoting a counter into an enclosing loop that was not previously
; in LoopToCandidates does not invalidate references when LoopToCandidates
; grows (at the 48th entry for the initial 64-bucket DenseMap).

@__profn_foo = private constant [3 x i8] c"foo"

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

; CHECK-LABEL: define void @foo(
define void @foo(i1 %c) {
entry:
  br label %L0

L0:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 0)
  br i1 %c, label %L0, label %L1
L1:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 1)
  br i1 %c, label %L1, label %L2
L2:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 2)
  br i1 %c, label %L2, label %L3
L3:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 3)
  br i1 %c, label %L3, label %L4
L4:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 4)
  br i1 %c, label %L4, label %L5
L5:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 5)
  br i1 %c, label %L5, label %L6
L6:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 6)
  br i1 %c, label %L6, label %L7
L7:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 7)
  br i1 %c, label %L7, label %L8
L8:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 8)
  br i1 %c, label %L8, label %L9
L9:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 9)
  br i1 %c, label %L9, label %L10
L10:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 10)
  br i1 %c, label %L10, label %L11
L11:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 11)
  br i1 %c, label %L11, label %L12
L12:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 12)
  br i1 %c, label %L12, label %L13
L13:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 13)
  br i1 %c, label %L13, label %L14
L14:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 14)
  br i1 %c, label %L14, label %L15
L15:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 15)
  br i1 %c, label %L15, label %L16
L16:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 16)
  br i1 %c, label %L16, label %L17
L17:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 17)
  br i1 %c, label %L17, label %L18
L18:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 18)
  br i1 %c, label %L18, label %L19
L19:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 19)
  br i1 %c, label %L19, label %L20
L20:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 20)
  br i1 %c, label %L20, label %L21
L21:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 21)
  br i1 %c, label %L21, label %L22
L22:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 22)
  br i1 %c, label %L22, label %L23
L23:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 23)
  br i1 %c, label %L23, label %L24
L24:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 24)
  br i1 %c, label %L24, label %L25
L25:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 25)
  br i1 %c, label %L25, label %L26
L26:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 26)
  br i1 %c, label %L26, label %L27
L27:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 27)
  br i1 %c, label %L27, label %L28
L28:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 28)
  br i1 %c, label %L28, label %L29
L29:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 29)
  br i1 %c, label %L29, label %L30
L30:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 30)
  br i1 %c, label %L30, label %L31
L31:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 31)
  br i1 %c, label %L31, label %L32
L32:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 32)
  br i1 %c, label %L32, label %L33
L33:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 33)
  br i1 %c, label %L33, label %L34
L34:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 34)
  br i1 %c, label %L34, label %L35
L35:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 35)
  br i1 %c, label %L35, label %L36
L36:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 36)
  br i1 %c, label %L36, label %L37
L37:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 37)
  br i1 %c, label %L37, label %L38
L38:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 38)
  br i1 %c, label %L38, label %L39
L39:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 39)
  br i1 %c, label %L39, label %L40
L40:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 40)
  br i1 %c, label %L40, label %L41
L41:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 41)
  br i1 %c, label %L41, label %L42
L42:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 42)
  br i1 %c, label %L42, label %L43
L43:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 43)
  br i1 %c, label %L43, label %L44
L44:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 44)
  br i1 %c, label %L44, label %L45
L45:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 45)
  br i1 %c, label %L45, label %outer.ph

outer.ph:
  br label %outer.header

outer.header:
  br label %inner.body

inner.body:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 47, i32 46)
  br i1 %c, label %inner.body, label %outer.latch

outer.latch:
  br i1 %c, label %outer.header, label %exit

exit:
  ret void
}
