; RUN: not llc -mtriple=bpfel -mcpu=v1 -filetype=asm < %s
;
; Source:
; $ cat xaddd_v1.c
; long test_fetch_add_64_ret(long *i) {
;   return __sync_fetch_and_add(i, 10);
; }

target triple = "bpf"

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define dso_local i64 @test_fetch_add_64_ret(ptr nocapture noundef %i) local_unnamed_addr #0 {
entry:
  %0 = atomicrmw add ptr %i, i64 10 seq_cst, align 8
  ret i64 %0
}

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v1" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git 6f71e34e194dab5a52cb2211af575c6067e9e504)"}
