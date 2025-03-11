; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check that bpf-preserve-static-offset folds chain of GEP instructions.
; The GEP chain in this example has type mismatch and thus is
; folded as i8 access.
;
; Source (modified by hand):
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
;       | opt -passes=function(sroa) -S -o -
;
; Modified to remove one of the 'inbounds' from one of the getelementptr.

%struct.bar = type { i8, %struct.foo }
%struct.foo = type { i8, i8 }

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.bar, ptr %0, i32 0, i32 1
  %add.ptr = getelementptr i8, ptr %b, i64 1
  %bb = getelementptr inbounds %struct.foo, ptr %add.ptr, i32 0, i32 1
  store i8 42, ptr %bb, align 1, !tbaa !2
  ret void
}

; CHECK:      define dso_local void @buz(ptr noundef writeonly captures(none) %[[p:.*]])
; CHECK:        tail call void (i8, ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.store.i8
; CHECK-SAME:       (i8 42,
; CHECK-SAME:        ptr writeonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 0, i1 false, i64 immarg 3)
; CHECK-SAME:      #[[v2:.*]], !tbaa ![[v3:.*]]
; CHECK:      attributes #[[v2]] = { memory(argmem: write) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 1}
!3 = !{!"foo", !4, i64 0, !4, i64 1}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
