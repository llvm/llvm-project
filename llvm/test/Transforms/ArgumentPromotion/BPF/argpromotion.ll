; RUN: opt -passes=argpromotion -mtriple=bpf-pc-linux -S %s | FileCheck %s
; Source:
;  struct t {
;    int a, b, c, d, e, f, g;
;  };
;  __attribute__((noinline)) static int foo1(struct t *p1, struct t *p2, struct t *p3) {
;    return p1->a + p1->b + p2->c + p2->e + p3->f + p3->g;
;  }
;  __attribute__((noinline)) static int foo2(struct t *p1, struct t *p2, struct t *p3) {
;    return p1->a + p1->b + p2->c + p2->e + p3->f;
;  }
;  void init(void *);
;  int bar(void) {
;    struct t v1, v2, v3;
;    init(&v1); init(&v2); init(&v3);
;    return foo1(&v1, &v2, &v3) + foo2(&v1, &v2, &v3);
;  }
; Compilation flag:
;   clang -target bpf -O2 -S t.c -mllvm -print-before=argpromotion -mllvm -print-module-scope
;   and then do some manual tailoring to remove some attributes/metadata which is not used
;   by argpromotion pass.

%struct.t = type { i32, i32, i32, i32, i32, i32, i32 }

define i32 @bar() {
entry:
  %v1 = alloca %struct.t, align 4
  %v2 = alloca %struct.t, align 4
  %v3 = alloca %struct.t, align 4
  call void @init(ptr noundef nonnull %v1)
  call void @init(ptr noundef nonnull %v2)
  call void @init(ptr noundef nonnull %v3)
  %call = call fastcc i32 @foo1(ptr noundef nonnull %v1, ptr noundef nonnull %v2, ptr noundef nonnull %v3)
  %call1 = call fastcc i32 @foo2(ptr noundef nonnull %v1, ptr noundef nonnull %v2, ptr noundef nonnull %v3)
  %add = add nsw i32 %call, %call1
  ret i32 %add
}

declare void @init(ptr noundef)

define internal i32 @foo1(ptr nocapture noundef readonly %p1, ptr nocapture noundef readonly %p2, ptr nocapture noundef readonly %p3) {
entry:
  %0 = load i32, ptr %p1, align 4
  %b = getelementptr inbounds %struct.t, ptr %p1, i64 0, i32 1
  %1 = load i32, ptr %b, align 4
  %add = add nsw i32 %1, %0
  %c = getelementptr inbounds %struct.t, ptr %p2, i64 0, i32 2
  %2 = load i32, ptr %c, align 4
  %add1 = add nsw i32 %add, %2
  %e = getelementptr inbounds %struct.t, ptr %p2, i64 0, i32 4
  %3 = load i32, ptr %e, align 4
  %add2 = add nsw i32 %add1, %3
  %f = getelementptr inbounds %struct.t, ptr %p3, i64 0, i32 5
  %4 = load i32, ptr %f, align 4
  %add3 = add nsw i32 %add2, %4
  %g = getelementptr inbounds %struct.t, ptr %p3, i64 0, i32 6
  %5 = load i32, ptr %g, align 4
  %add4 = add nsw i32 %add3, %5
  ret i32 %add4
}

; Without number-of-argument constraint, argpromotion will create a function signature with 6 arguments. Since
; bpf target only supports maximum 5 arguments, so no argpromotion here.
;
; CHECK:  i32 @foo1(ptr noundef readonly captures(none) %p1, ptr noundef readonly captures(none) %p2, ptr noundef readonly captures(none) %p3)

define internal i32 @foo2(ptr noundef %p1, ptr noundef %p2, ptr noundef %p3) {
entry:
  %0 = load i32, ptr %p1, align 4
  %b = getelementptr inbounds %struct.t, ptr %p1, i64 0, i32 1
  %1 = load i32, ptr %b, align 4
  %add = add nsw i32 %0, %1
  %c = getelementptr inbounds %struct.t, ptr %p2, i64 0, i32 2
  %2 = load i32, ptr %c, align 4
  %add1 = add nsw i32 %add, %2
  %e = getelementptr inbounds %struct.t, ptr %p2, i64 0, i32 4
  %3 = load i32, ptr %e, align 4
  %add2 = add nsw i32 %add1, %3
  %f = getelementptr inbounds %struct.t, ptr %p3, i64 0, i32 5
  %4 = load i32, ptr %f, align 4
  %add3 = add nsw i32 %add2, %4
  ret i32 %add3
}

; Without number-of-argument constraint, argpromotion will create a function signature with 5 arguments, which equals
; the maximum number of argument permitted by bpf backend, so argpromotion result code does work.
;
; CHECK:  i32 @foo2(i32 %p1.0.val, i32 %p1.4.val, i32 %p2.8.val, i32 %p2.16.val, i32 %p3.20.val)
