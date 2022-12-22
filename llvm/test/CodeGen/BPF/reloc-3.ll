; RUN: llc -march=bpfel -filetype=obj -o %t.el < %s
; RUN: llvm-readelf -r %t.el | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o %t.eb < %s
; RUN: llvm-readelf -r %t.eb | FileCheck %s

; source code:
;   int g() { return 0; }
;   struct t { ptr p; } gbl = { g };
; compilation flag:
;   clang -target bpf -O2 -emit-llvm -S test.c

%struct.t = type { ptr }

@gbl = dso_local local_unnamed_addr global %struct.t { ptr @g }, align 8

; CHECK: '.rel.data'
; CHECK: 0000000000000000  0000000200000002 R_BPF_64_ABS64         0000000000000000 g

; Function Attrs: nofree norecurse nosync nounwind readnone willreturn mustprogress
define dso_local i32 @g() #0 {
entry:
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind readnone willreturn mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
