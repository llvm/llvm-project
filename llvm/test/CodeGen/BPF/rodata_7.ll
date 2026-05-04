; RUN: llc -mtriple=bpf -mcpu=v1 < %s | FileCheck %s
;
; Source code:
;   struct t1 { int a; };
;   volatile const struct t1 data = { .a = 3 };
;   int foo(void) {
;     return data.a + 20;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm test.c

%struct.t1 = type { i32 }

@data = dso_local constant %struct.t1 { i32 3 }, align 4

; Function Attrs: nofree norecurse nounwind
define dso_local i32 @foo() local_unnamed_addr {
entry:
  %0 = load volatile i32, ptr @data, align 4
  %add = add nsw i32 %0, 20
; CHECK:   [[REG1:r[0-9]+]] = data ll
; CHECK:   r0 = *(u32 *)([[REG1]] + 0)
; CHECK:   r0 += 20
  ret i32 %add
}
