; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check that stores from zero offset are not modified by bpf-preserve-static-offset.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a;
;    } __ctx;
;    
;    void bar(struct foo *p) {
;      p->a = 0;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { i32 }

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %a = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 0
  store i32 0, ptr %a, align 4, !tbaa !2
  ret void
}

; CHECK:      define dso_local void @bar(ptr nocapture noundef writeonly %[[p:.*]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   store i32 0, ptr %[[p]], align 4, !tbaa
; CHECK-NEXT:   ret void

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 0}
!3 = !{!"foo", !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
