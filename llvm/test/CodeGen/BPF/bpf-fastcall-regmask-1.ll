; RUN: llc -O2 --mtriple=bpfel \
; RUN:   -print-after=stack-slot-coloring %s \
; RUN:   -o /dev/null 2>&1 | FileCheck %s

; Generated from the following C code:
;
;   #define __bpf_fastcall __attribute__((bpf_fastcall))
;
;   void bar1(void) __bpf_fastcall;
;   void buz1(long i, long j, long k);
;   void foo1(long i, long j, long k) {
;     bar1();
;     buz1(i, j, k);
;   }
;
;   long bar2(void) __bpf_fastcall;
;   void buz2(long i, long j, long k);
;   void foo2(long i, long j, long k) {
;     bar2();
;     buz2(i, j, k);
;   }
;
;   void bar3(long) __bpf_fastcall;
;   void buz3(long i, long j, long k);
;   void foo3(long i, long j, long k) {
;     bar3(i);
;     buz3(i, j, k);
;   }
;
;   long bar4(long, long) __bpf_fastcall;
;   void buz4(long i, long j, long k);
;   void foo4(long i, long j, long k) {
;     bar4(i, j);
;     buz4(i, j, k);
;   }
;
; Using the following command:
;
;   clang --target=bpf -emit-llvm -O2 -S -o - t.c
;
; (unnecessary attrs removed maually)

; Check regmask for calls to functions marked with bpf_fastcall:
; - void function w/o parameters
; - non-void function w/o parameters
; - void function with parameters
; - non-void function with parameters

declare dso_local void @bar1() #0
declare dso_local void @buz1(i64 noundef, i64 noundef, i64 noundef)
define dso_local void @foo1(i64 noundef %i, i64 noundef %j, i64 noundef %k) {
entry:
  tail call void @bar1() #1
  tail call void @buz1(i64 noundef %i, i64 noundef %j, i64 noundef %k)
  ret void
}

; CHECK:      JAL @bar1, <regmask $r0 $r1 $r2 $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10
; CHECK-SAME:                     $w0 $w1 $w2 $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit-def $r11
; CHECK:      JAL @buz1, <regmask $r6 $r7 $r8 $r9 $r10 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit $r2, implicit $r3, implicit-def $r11

declare dso_local i64 @bar2() #0
declare dso_local void @buz2(i64 noundef, i64 noundef, i64 noundef)
define dso_local void @foo2(i64 noundef %i, i64 noundef %j, i64 noundef %k) {
entry:
  tail call i64 @bar2() #1
  tail call void @buz2(i64 noundef %i, i64 noundef %j, i64 noundef %k)
  ret void
}

; CHECK:      JAL @bar2, <regmask $r1 $r2 $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10
; CHECK-SAME:                     $w1 $w2 $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit-def $r11, implicit-def dead $r0
; CHECK:      JAL @buz2, <regmask $r6 $r7 $r8 $r9 $r10 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit $r2, implicit $r3, implicit-def $r11

declare dso_local void @bar3(i64) #0
declare dso_local void @buz3(i64 noundef, i64 noundef, i64 noundef)
define dso_local void @foo3(i64 noundef %i, i64 noundef %j, i64 noundef %k) {
entry:
  tail call void @bar3(i64 noundef %i) #1
  tail call void @buz3(i64 noundef %i, i64 noundef %j, i64 noundef %k)
  ret void
}

; CHECK:      JAL @bar3, <regmask $r0 $r2 $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10
; CHECK-SAME:                     $w0 $w2 $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit-def $r11
; CHECK:      JAL @buz3, <regmask $r6 $r7 $r8 $r9 $r10 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit $r2, implicit $r3, implicit-def $r11

declare dso_local i64 @bar4(i64 noundef, i64 noundef) #0
declare dso_local void @buz4(i64 noundef, i64 noundef, i64 noundef)
define dso_local void @foo4(i64 noundef %i, i64 noundef %j, i64 noundef %k) {
entry:
  tail call i64 @bar4(i64 noundef %i, i64 noundef %j) #1
  tail call void @buz4(i64 noundef %i, i64 noundef %j, i64 noundef %k)
  ret void
}

; CHECK:      JAL @bar4, <regmask $r3 $r4 $r5 $r6 $r7 $r8 $r9 $r10
; CHECK-SAME:                     $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit $r2, implicit-def $r11, implicit-def dead $r0
; CHECK:      JAL @buz4, <regmask $r6 $r7 $r8 $r9 $r10 $w6 $w7 $w8 $w9 $w10>
; CHECK-SAME:          , implicit $r11, implicit $r1, implicit $r2, implicit $r3, implicit-def $r11

attributes #0 = { "bpf_fastcall" }
attributes #1 = { nounwind "bpf_fastcall" }
