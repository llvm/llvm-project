; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check that bpf-preserve-static-offset keeps track of 'inbounds' flags while
; folding chain of GEP instructions.
;
; Source (IR modified by hand):
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a[2];
;    };
;    
;    struct bar {
;      int a;
;      struct foo b;
;    } __ctx;
;    
;    extern void consume(int);
;    
;    void buz(struct bar *p) {
;      consume(p->b.a[1]);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -
;
; Modified to remove one of the 'inbounds' from one of the GEP instructions.

%struct.bar = type { i32, %struct.foo }
%struct.foo = type { [2 x i32] }

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.bar, ptr %0, i32 0, i32 1
  %a = getelementptr %struct.foo, ptr %b, i32 0, i32 0
  %arrayidx = getelementptr inbounds [2 x i32], ptr %a, i64 0, i64 1
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !2
  call void @consume(i32 noundef %1)
  ret void
}

; CHECK:      %[[v1:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:    @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:      (ptr readonly elementtype(%struct.bar) %{{[^,]+}},
; CHECK-SAME:       i1 false, i8 0, i8 1, i8 2, i1 false,
;                                               ^^^^^^^^
;                                             not inbounds
; CHECK-SAME:       i32 immarg 0, i32 immarg 1, i32 immarg 0, i64 immarg 1)
;                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
;                                         folded gep chain
; CHECK-NEXT: call void @consume(i32 noundef %[[v1]])

declare void @consume(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
