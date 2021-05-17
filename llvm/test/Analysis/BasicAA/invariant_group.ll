; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -basic-aa -gvn -S | FileCheck -check-prefix=CHECK-GVN %s

; The input *.ll had been adapted from bug 37458:
;
; struct A { virtual void f(); int n; };
;
; int h() {
;     A a;
;     a.n = 42;
;     return __builtin_launder(&a)->n;
; }

%struct.A = type <{ i8*, i8 }>

; CHECK: testLaunderInvariantGroupIsNotEscapeSource
; CHECK-GVN: testLaunderInvariantGroupIsNotEscapeSource
define i8 @testLaunderInvariantGroupIsNotEscapeSource() {
; CHECK-DAG: MustAlias: %struct.A* %a, i8* %a.bitcast
; CHECK-DAG: PartialAlias (off {{[0-9]+}}): %struct.A* %a, i8* %n
; CHECK-DAG: NoAlias: i8* %a.bitcast, i8* %n
; CHECK-DAG: MustAlias: %struct.A* %a, i8* %a.laundered
; CHECK-DAG: MustAlias: i8* %a.bitcast, i8* %a.laundered
; CHECK-DAG: NoAlias: i8* %a.laundered, i8* %n
; CHECK-DAG: PartialAlias (off {{[0-9]+}}): %struct.A* %a, i8* %n.laundered
; CHECK-DAG: NoAlias: i8* %a.bitcast, i8* %n.laundered
; CHECK-DAG: MustAlias: i8* %n, i8* %n.laundered
; CHECK-DAG: NoAlias: i8* %a.laundered, i8* %n.laundered
; CHECK-DAG: NoModRef: Ptr: %struct.A* %a <-> %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)
; CHECK-DAG: NoModRef: Ptr: i8* %a.bitcast <-> %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)
; CHECK-DAG: NoModRef: Ptr: i8* %n <-> %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)
; CHECK-DAG: NoModRef: Ptr: i8* %a.laundered <-> %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)
; CHECK-DAG: NoModRef: Ptr: i8* %n.laundered <-> %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)

entry:
  %a = alloca %struct.A, align 8
  %a.bitcast = bitcast %struct.A* %a to i8*
  %n = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 1
  store i8 42, i8* %n
  %a.laundered = call i8* @llvm.launder.invariant.group.p0i8(i8* nonnull %a.bitcast)
  %n.laundered = getelementptr inbounds i8, i8* %a.laundered, i64 8
  %v = load i8, i8* %n.laundered
; make sure that the load from %n.laundered to %v aliases the store of 42 to %n
; CHECK-GVN: ret i8 42
  ret i8 %v
}

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
