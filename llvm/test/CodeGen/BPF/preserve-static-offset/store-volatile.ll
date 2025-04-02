; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check handling of a volatile store instruction by bpf-preserve-static-offset.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a;
;      volatile int b;
;    } __ctx;
;    
;    void bar(struct foo *p) {
;      p->b = 42;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 1
  store volatile i32 42, ptr %b, align 4, !tbaa !2
  ret void
}

; CHECK:      define dso_local void @bar(ptr noundef captures(none) %[[p:.*]])
; CHECK:        tail call void (i32, ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.store.i32
; CHECK-SAME:       (i32 42,
; CHECK-SAME:        ptr elementtype(%struct.foo) %[[p]],
; CHECK-SAME:        i1 true, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 1),
; CHECK-NOT:       #{{[0-9]+}}
; CHECK-SAME:      !tbaa

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 4}
!3 = !{!"foo", !4, i64 0, !4, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
