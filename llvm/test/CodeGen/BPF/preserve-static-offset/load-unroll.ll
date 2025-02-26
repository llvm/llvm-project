; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check position of bpf-preserve-static-offset pass in the pipeline:
; preserve.static.offset call should be preserved long enough to allow
; introduction of getelementptr.and.load after loops unrolling.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a;
;      int b[4];
;    } __ctx;
;    
;    extern void consume(int);
;    
;    void bar(struct foo *p){
;      unsigned long i = 0;
;    #pragma clang loop unroll(full)
;      while (i < 2)
;        consume(p->b[i++]);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { i32, [4 x i32] }

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %p) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %while.body ]
  %cmp = icmp ult i64 %i.0, 2
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 1
  %inc = add i64 %i.0, 1
  %arrayidx = getelementptr inbounds [4 x i32], ptr %b, i64 0, i64 %i.0
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !2
  call void @consume(i32 noundef %1)
  br label %while.cond, !llvm.loop !6

while.end:                                        ; preds = %while.cond
  ret void
}

; CHECK:      define dso_local void @bar(ptr noundef readonly captures(none) %[[p:.*]])
; CHECK:        %[[v1:.*]] = tail call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 4)
; CHECK-SAME:      #[[attrs:.*]], !tbaa
; CHECK-NEXT:   tail call void @consume(i32 noundef %[[v1]])
; CHECK-NEXT:   %[[v2:.*]] = tail call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 8)
; CHECK-SAME:      #[[attrs]], !tbaa
; CHECK-NEXT:   tail call void @consume(i32 noundef %[[v2]])
; CHECK:      attributes #[[attrs]] = { memory(argmem: read) }

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare void @consume(i32 noundef) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.unroll.full"}
