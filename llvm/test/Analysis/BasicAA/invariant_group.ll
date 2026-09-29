; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn -S | FileCheck -check-prefix=CHECK-GVN %s

; The input *.ll had been adapted from bug 37458:
;
; struct A { virtual void f(); int n; };
;
; int h() {
;     A a;
;     a.n = 42;
;     return __builtin_launder(&a)->n;
; }

%struct.A = type <{ ptr, i8 }>

; CHECK: testLaunderInvariantGroupIsNotEscapeSource
; CHECK-GVN: testLaunderInvariantGroupIsNotEscapeSource
define i8 @testLaunderInvariantGroupIsNotEscapeSource() {
; CHECK-DAG: MustAlias:	i8* %a, %struct.A* %a
; CHECK-DAG: PartialAlias (off 8):	%struct.A* %a, i8* %n
; CHECK-DAG: NoAlias:	i8* %a, i8* %n
; CHECK-DAG: MustAlias:	%struct.A* %a, i8* %a.laundered
; CHECK-DAG: MustAlias:	i8* %a, i8* %a.laundered
; CHECK-DAG: NoAlias:	i8* %a.laundered, i8* %n
; CHECK-DAG: PartialAlias (off 8):	%struct.A* %a, i8* %n.laundered
; CHECK-DAG: NoAlias:	i8* %a, i8* %n.laundered
; CHECK-DAG: MustAlias:	i8* %n, i8* %n.laundered
; CHECK-DAG: NoAlias:	i8* %a.laundered, i8* %n.laundered
; CHECK-DAG: NoModRef: Ptr: %struct.A* %a <-> %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)
; CHECK-DAG: NoModRef: Ptr: i8* %a <-> %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)
; CHECK-DAG: NoModRef: Ptr: i8* %n <-> %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)
; CHECK-DAG: NoModRef: Ptr: i8* %a.laundered <-> %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)
; CHECK-DAG: NoModRef: Ptr: i8* %n.laundered <-> %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)

entry:
  %a = alloca %struct.A, align 8
  load %struct.A, ptr %a
  load i8, ptr %a
  %n = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 1
  store i8 42, ptr %n
  %a.laundered = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %a)
  load i8, ptr %a.laundered
  %n.laundered = getelementptr inbounds i8, ptr %a.laundered, i64 8
  %v = load i8, ptr %n.laundered
; make sure that the load from %n.laundered to %v aliases the store of 42 to %n
; CHECK-GVN: ret i8 42
  ret i8 %v
}

declare ptr @llvm.launder.invariant.group.p0(ptr)
