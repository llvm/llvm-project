; RUN: llc -mtriple=bpfel -mcpu=v1 -filetype=obj -o %t.el < %s
; RUN: llvm-objdump -r %t.el | FileCheck --check-prefix=RELOC %s
; RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.el | FileCheck --check-prefix=DUMP %s
; RUN: llc -mtriple=bpfeb -mcpu=v1 -filetype=obj -o %t.eb < %s
; RUN: llvm-objdump -r %t.eb | FileCheck --check-prefix=RELOC %s
; RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.eb | FileCheck --check-prefix=DUMP %s

; source code:
;   static __attribute__((noinline)) __attribute__((section("sec1")))
;   int add(int a, int b) {
;     return a + b;
;   }
;   static __attribute__((noinline))
;   int sub(int a, int b) {
;     return a - b;
;   }
;   int test(int a, int b) {
;     return add(a, b) + sub(a, b);
;   }
; compilation flag:
;   clang -target bpf -O2 -emit-llvm -S test.c

define dso_local i32 @test(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %call = tail call fastcc i32 @add(i32 %a, i32 %b)
  %call1 = tail call fastcc i32 @sub(i32 %a, i32 %b)
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define internal fastcc i32 @add(i32 %a, i32 %b) unnamed_addr #1 section "sec1" {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

; Function Attrs: nofree noinline norecurse nosync nounwind readnone willreturn mustprogress
define internal fastcc i32 @sub(i32 %a, i32 %b) unnamed_addr #1 {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

; DUMP:       .text:
; DUMP-EMPTY:
; DUMP-NEXT:  <test>
; DUMP-NEXT:  r[[#]] = r[[#]]
; DUMP-NEXT:  r[[#]] = r[[#]]
; DUMP-NEXT:  call -1

; DUMP:       sec1:
; DUMP-EMPTY:
; DUMP-NEXT:  <add>

; RELOC:      RELOCATION RECORDS FOR [.text]:
; RELOC:      R_BPF_64_32            sec1
; RELOC-NOT:  R_BPF_64_32

attributes #0 = { nofree norecurse nosync nounwind readnone willreturn mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nofree noinline norecurse nosync nounwind readnone willreturn mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
