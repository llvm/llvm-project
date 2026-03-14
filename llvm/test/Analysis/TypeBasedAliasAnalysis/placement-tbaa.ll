; RUN: opt < %s -aa-pipeline=tbaa,basic-aa -passes=aa-eval -evaluate-aa-metadata -print-no-aliases -print-may-aliases -disable-output 2>&1 | FileCheck %s

; Generated with "clang -cc1 -disable-llvm-optzns -O1 -emit-llvm"
; #include <new>
; struct Foo { long i; };
; struct Bar { void *p; };
; long foo(int n) {
;   Foo *f = new Foo;
;   f->i = 1;
;   for (int i=0; i<n; ++i) {
;     Bar *b = new (f) Bar;
;     b->p = 0;
;     f = new (f) Foo;
;     f->i = i;
;   }
;   return f->i;
; }

; Basic AA says MayAlias, TBAA says NoAlias
; CHECK: MayAlias: ptr* %5, i64* %9
; CHECK: NoAlias: store i64 %conv, ptr %9, align 8, !tbaa !6 <->   store ptr null, ptr %5, align 8, !tbaa !9

%struct.Foo = type { i64 }
%struct.Bar = type { ptr }

define i64 @_Z3fooi(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %f = alloca ptr, align 8
  %i1 = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4, !tbaa !0
  %call = call noalias ptr @_Znwm(i64 8)
  store ptr %call, ptr %f, align 8, !tbaa !4
  %0 = load ptr, ptr %f, align 8, !tbaa !4
  store i64 1, ptr %0, align 8, !tbaa !6
  store i32 0, ptr %i1, align 4, !tbaa !0
  br label %for.cond

for.cond:
  %1 = load i32, ptr %i1, align 4, !tbaa !0
  %2 = load i32, ptr %n.addr, align 4, !tbaa !0
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %3 = load ptr, ptr %f, align 8, !tbaa !4
  %new.isnull = icmp eq ptr %3, null
  br i1 %new.isnull, label %new.cont, label %new.notnull

new.notnull:
  br label %new.cont

new.cont:
  %4 = phi ptr [ %3, %new.notnull ], [ null, %for.body ]
  store ptr %4, ptr %b, align 8, !tbaa !4
  %5 = load ptr, ptr %b, align 8, !tbaa !4
  store ptr null, ptr %5, align 8, !tbaa !9
  %6 = load ptr, ptr %f, align 8, !tbaa !4
  %new.isnull2 = icmp eq ptr %6, null
  br i1 %new.isnull2, label %new.cont4, label %new.notnull3

new.notnull3:
  br label %new.cont4

new.cont4:
  %7 = phi ptr [ %6, %new.notnull3 ], [ null, %new.cont ]
  store ptr %7, ptr %f, align 8, !tbaa !4
  %8 = load i32, ptr %i1, align 4, !tbaa !0
  %conv = sext i32 %8 to i64
  %9 = load ptr, ptr %f, align 8, !tbaa !4
  store i64 %conv, ptr %9, align 8, !tbaa !6
  br label %for.inc

for.inc:
  %10 = load i32, ptr %i1, align 4, !tbaa !0
  %inc = add nsw i32 %10, 1
  store i32 %inc, ptr %i1, align 4, !tbaa !0
  br label %for.cond

for.end:
  %11 = load ptr, ptr %f, align 8, !tbaa !4
  %12 = load i64, ptr %11, align 8, !tbaa !6
  ret i64 %12
}

declare noalias ptr @_Znwm(i64)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !2, i64 0}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS3Foo", !8, i64 0}
!8 = !{!"long", !2, i64 0}
!9 = !{!10, !5, i64 0}
!10 = !{!"_ZTS3Bar", !5, i64 0}
