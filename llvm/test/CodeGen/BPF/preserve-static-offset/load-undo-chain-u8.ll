; RUN: opt --bpf-check-and-opt-ir -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check unroll of getelementptr.and.load when direct memory offset is
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
;    extern void consume(char);
;    
;    void buz(struct bar *p) {
;      consume(((struct foo *)(((char*)&p->b) + 1))->bb);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=sroa,bpf-preserve-static-offset -S -o -

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %p) #0 {
entry:
  %bb1 = call i8 (ptr, i1, i8, i8, i8, i1, ...)
    @llvm.bpf.getelementptr.and.load.i8
      (ptr readonly elementtype(i8) %p,
       i1 false, i8 0, i8 1, i8 0, i1 true, i64 immarg 3)
    #4, !tbaa !2
  call void @consume(i8 noundef signext %bb1)
  ret void
}

; CHECK: define dso_local void @buz(ptr noundef %[[p:.*]])
; CHECK:   %[[bb11:.*]] = getelementptr inbounds i8, ptr %[[p]], i64 3
; CHECK:   %[[v2:.*]] = load i8, ptr %[[bb11]], align 1
; CHECK:   call void @consume(i8 noundef signext %[[v2]])

declare void @consume(i8 noundef signext) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #2

; Function Attrs: nocallback nofree nounwind willreturn
declare i8 @llvm.bpf.getelementptr.and.load.i8(ptr nocapture, i1 immarg, i8 immarg, i8 immarg, i8 immarg, i1 immarg, ...) #3

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nounwind willreturn }
attributes #4 = { memory(argmem: read) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 1}
!3 = !{!"foo", !4, i64 0, !4, i64 1}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
