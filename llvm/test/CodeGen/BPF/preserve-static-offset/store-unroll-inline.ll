; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check position of bpf-preserve-static-offset pass in the pipeline:
; - preserve.static.offset call is preserved if address is passed as
;   a parameter to an inline-able function;
; - second bpf-preserve-static-offset pass (after inlining) should introduce
;   getelementptr.and.store call using the preserved marker after loops
;   unrolling;
; - memory(argmem: readwrite) and tbaa attributes should allow
;   removing one getelementptr.and.store call.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a;
;      int b[4];
;    } __ctx;
;    
;    static inline void bar(int * restrict p, unsigned long i) {
;      p[0] = i;
;    }
;    
;    void quux(struct foo *p){
;      unsigned long i = 0;
;    #pragma clang loop unroll(full)
;      while (i < 2) {
;        bar(p->b, i);
;        ++i;
;      }
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { i32, [4 x i32] }

; Function Attrs: nounwind
define dso_local void @quux(ptr noundef %p) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %while.body ]
  %cmp = icmp ult i64 %i.0, 2
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 1
  %arraydecay = getelementptr inbounds [4 x i32], ptr %b, i64 0, i64 0
  call void @bar(ptr noundef %arraydecay, i64 noundef %i.0)
  %inc = add i64 %i.0, 1
  br label %while.cond, !llvm.loop !2

while.end:                                        ; preds = %while.cond
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: inlinehint nounwind
define internal void @bar(ptr noalias noundef %p, i64 noundef %i) #2 {
entry:
  %conv = trunc i64 %i to i32
  %arrayidx = getelementptr inbounds i32, ptr %p, i64 0
  store i32 %conv, ptr %arrayidx, align 4, !tbaa !5
  ret void
}

; CHECK:      define dso_local void @quux(ptr nocapture noundef writeonly %[[p:.*]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void (i32, ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.store.i32
; CHECK-SAME:       (i32 1,
; CHECK-SAME:        ptr writeonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 4)
; CHECK-NEXT:   ret void

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { inlinehint nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.mustprogress"}
!4 = !{!"llvm.loop.unroll.full"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
