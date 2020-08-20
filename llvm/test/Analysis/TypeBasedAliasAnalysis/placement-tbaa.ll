; RUN: opt < %s -tbaa -basic-aa -aa-eval -evaluate-aa-metadata -print-no-aliases -print-may-aliases -disable-output 2>&1 | FileCheck %s
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
; CHECK: MayAlias: i64* %i5, i8** %p
; CHECK: NoAlias: store i64 %conv, i64* %i5, align 8, !tbaa !6 <->   store i8* null, i8** %p, align 8, !tbaa !9

%struct.Foo = type { i64 }
%struct.Bar = type { i8* }

define i64 @_Z3fooi(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %f = alloca %struct.Foo*, align 8
  %i1 = alloca i32, align 4
  %b = alloca %struct.Bar*, align 8
  store i32 %n, i32* %n.addr, align 4, !tbaa !0
  %call = call noalias i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %struct.Foo*
  store %struct.Foo* %0, %struct.Foo** %f, align 8, !tbaa !4
  %1 = load %struct.Foo*, %struct.Foo** %f, align 8, !tbaa !4
  %i = getelementptr inbounds %struct.Foo, %struct.Foo* %1, i32 0, i32 0
  store i64 1, i64* %i, align 8, !tbaa !6
  store i32 0, i32* %i1, align 4, !tbaa !0
  br label %for.cond

for.cond:
  %2 = load i32, i32* %i1, align 4, !tbaa !0
  %3 = load i32, i32* %n.addr, align 4, !tbaa !0
  %cmp = icmp slt i32 %2, %3
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %4 = load %struct.Foo*, %struct.Foo** %f, align 8, !tbaa !4
  %5 = bitcast %struct.Foo* %4 to i8*
  %new.isnull = icmp eq i8* %5, null
  br i1 %new.isnull, label %new.cont, label %new.notnull

new.notnull:
  %6 = bitcast i8* %5 to %struct.Bar*
  br label %new.cont

new.cont:
  %7 = phi %struct.Bar* [ %6, %new.notnull ], [ null, %for.body ]
  store %struct.Bar* %7, %struct.Bar** %b, align 8, !tbaa !4
  %8 = load %struct.Bar*, %struct.Bar** %b, align 8, !tbaa !4
  %p = getelementptr inbounds %struct.Bar, %struct.Bar* %8, i32 0, i32 0
  store i8* null, i8** %p, align 8, !tbaa !9
  %9 = load %struct.Foo*, %struct.Foo** %f, align 8, !tbaa !4
  %10 = bitcast %struct.Foo* %9 to i8*
  %new.isnull2 = icmp eq i8* %10, null
  br i1 %new.isnull2, label %new.cont4, label %new.notnull3

new.notnull3:
  %11 = bitcast i8* %10 to %struct.Foo*
  br label %new.cont4

new.cont4:
  %12 = phi %struct.Foo* [ %11, %new.notnull3 ], [ null, %new.cont ]
  store %struct.Foo* %12, %struct.Foo** %f, align 8, !tbaa !4
  %13 = load i32, i32* %i1, align 4, !tbaa !0
  %conv = sext i32 %13 to i64
  %14 = load %struct.Foo*, %struct.Foo** %f, align 8, !tbaa !4
  %i5 = getelementptr inbounds %struct.Foo, %struct.Foo* %14, i32 0, i32 0
  store i64 %conv, i64* %i5, align 8, !tbaa !6
  br label %for.inc

for.inc:
  %15 = load i32, i32* %i1, align 4, !tbaa !0
  %inc = add nsw i32 %15, 1
  store i32 %inc, i32* %i1, align 4, !tbaa !0
  br label %for.cond

for.end:
  %16 = load %struct.Foo*, %struct.Foo** %f, align 8, !tbaa !4
  %i6 = getelementptr inbounds %struct.Foo, %struct.Foo* %16, i32 0, i32 0
  %17 = load i64, i64* %i6, align 8, !tbaa !6
  ret i64 %17
}

declare noalias i8* @_Znwm(i64)

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
