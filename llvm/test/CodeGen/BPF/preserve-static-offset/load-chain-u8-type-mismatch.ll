; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Check that bpf-preserve-static-offset folds chain of GEP instructions.
; The GEP chain in this example has unexpected shape and thus is
; folded as i8 access.
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
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { i8, %struct.foo }
%struct.foo = type { i8, i8 }

; Function Attrs: nounwind
define dso_local void @buz(ptr noundef %p) #0 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p)
  %b = getelementptr inbounds %struct.bar, ptr %0, i32 0, i32 1
  %add.ptr = getelementptr inbounds i8, ptr %b, i64 1
;                                   ~~
;         these types do not match, thus GEP chain is folded as an offset
;                              ~~~~~~~~~~~
  %bb = getelementptr inbounds %struct.foo, ptr %add.ptr, i32 0, i32 1
  %1 = load i8, ptr %bb, align 1, !tbaa !2
  call void @consume(i8 noundef signext %1)
  ret void
}

; CHECK:      %[[bb1:.*]] = call i8 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:   @llvm.bpf.getelementptr.and.load.i8
; CHECK-SAME:     (ptr readonly elementtype(i8) %{{[^,]+}},
; CHECK-SAME:      i1 false, i8 0, i8 1, i8 0, i1 true, i64 immarg 3)
;                                                       ^^^^^^^^^^^^
;                                      offset from 'struct bar' start
; CHECK-NEXT: call void @consume(i8 noundef signext %[[bb1]])

declare void @consume(i8 noundef signext) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
!2 = !{!3, !4, i64 1}
!3 = !{!"foo", !4, i64 0, !4, i64 1}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
