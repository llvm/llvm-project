; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   extern void foo(void) __attribute__((nomerge));
;
;   void bar(long i) {
;     if (i)
;       foo();
;     else
;       foo();
;   }
;
; Compilation flag:
;   clang -target bpf -S -O2 -emit-llvm t.c -o t.ll

; The goal of the test is to check that 'nomerge' attribute
; preserves two calls to 'foo' from merging.

; CHECK:     call foo
; CHECK:     call foo

; Function Attrs: nounwind
define dso_local void @bar(i64 noundef %i) local_unnamed_addr #0 {
entry:
  %tobool.not = icmp eq i64 %i, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @foo() #2
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void @foo() #2
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare dso_local void @foo() local_unnamed_addr #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nomerge nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang bd66f4b1da304af8e5a890b3205ce6f3d76667ee)"}
