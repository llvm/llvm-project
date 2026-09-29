; RUN: opt --bpf-check-and-opt-ir -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check unroll of getelementptr.and.store when direct memory offset is
; used instead of field indexes.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      char aa;
;      char bb;
;    };
;    
;    struct bar {
;      char a;
;      struct foo b;
;    } __ctx;
;    
;    void buz(struct bar *p) {
;      ((struct foo *)(((char*)&p->b) + 1))->bb = 42;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=sroa,bpf-preserve-static-offset -S -o -

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %p) #0 {
entry:
  call void (i8, ptr, i1, i8, i8, i8, i1, ...)
    @llvm.bpf.getelementptr.and.store.i8
      (i8 42,
       ptr writeonly elementtype(i8) %p,
       i1 false, i8 0, i8 1, i8 0, i1 true, i64 immarg 3)
    #3, !tbaa !2
  ret void
}

; CHECK: define dso_local void @buz(ptr noundef %[[p:.*]])
; CHECK:   %[[v2:.*]] = getelementptr inbounds i8, ptr %[[p]], i64 3
; CHECK:   store i8 42, ptr %[[v2]], align 1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.bpf.getelementptr.and.store.i8(i8, ptr nocapture, i1 immarg, i8 immarg, i8 immarg, i8 immarg, i1 immarg, ...) #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn }
attributes #3 = { memory(argmem: write) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 1}
!3 = !{!"foo", !4, i64 0, !4, i64 1}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
