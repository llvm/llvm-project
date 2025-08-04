; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check position of bpf-preserve-static-offset pass in the pipeline:
; - preserve.static.offset call is preserved if address is passed as
;   a parameter to an inline-able function;
; - second bpf-preserve-static-offset pass (after inlining) should introduce
;   getelementptr.and.load call using the preserved marker.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct bar {
;      int aa;
;      int bb;
;    };
;    
;    struct foo {
;      int a;
;      struct bar b;
;    } __ctx;
;    
;    extern void consume(int);
;    
;    static inline void bar(struct bar *p){
;      consume(p->bb);
;    }
;    
;    void quux(struct foo *p) {
;      bar(&p->b);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { i32, %struct.bar }
%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @quux(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 1
  call void @bar(ptr noundef %b)
  ret void
}

; Function Attrs: inlinehint nounwind
define internal void @bar(ptr noundef %p) #1 {
entry:
  %bb = getelementptr inbounds %struct.bar, ptr %p, i32 0, i32 1
  %0 = load i32, ptr %bb, align 4, !tbaa !2
  call void @consume(i32 noundef %0)
  ret void
}

; CHECK:      define dso_local void @quux(ptr noundef readonly captures(none) %[[p:.*]])
; CHECK:        %[[bb_i1:.*]] = tail call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 8)
; CHECK-SAME:      #[[v2:.*]], !tbaa
; CHECK-NEXT:   tail call void @consume(i32 noundef %[[bb_i1]])
; CHECK:      attributes #[[v2]] = { memory(argmem: read) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #2

declare void @consume(i32 noundef) #3

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { inlinehint nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 4}
!3 = !{!"bar", !4, i64 0, !4, i64 4}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
