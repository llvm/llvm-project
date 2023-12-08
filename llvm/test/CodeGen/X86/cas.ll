; RUN: llc -mtriple=x86_64-pc-linux-gnu %s -o - -no-integrated-as | FileCheck %s

; C code this came from
;bool cas(float volatile *p, ptr expected, float desired) {
;  bool success;
;  __asm__ __volatile__("lock; cmpxchg %[desired], %[mem]; "
;                       "mov %[expected], %[expected_out]; "
;                       "sete %[success]"
;                       : [success] "=a" (success),
;                         [expected_out] "=rm" (*expected)
;                       : [expected] "a" (*expected),
;                         [desired] "q" (desired),
;                         [mem] "m" (*p)
;                       : "memory", "cc");
;  return success;
;}

define zeroext i1 @cas(ptr %p, ptr %expected, float %desired) nounwind {
entry:
  %p.addr = alloca ptr, align 8
  %expected.addr = alloca ptr, align 8
  %desired.addr = alloca float, align 4
  %success = alloca i8, align 1
  store ptr %p, ptr %p.addr, align 8
  store ptr %expected, ptr %expected.addr, align 8
  store float %desired, ptr %desired.addr, align 4
  %0 = load ptr, ptr %expected.addr, align 8
  %1 = load ptr, ptr %expected.addr, align 8
  %2 = load float, ptr %1, align 4
  %3 = load float, ptr %desired.addr, align 4
  %4 = load ptr, ptr %p.addr, align 8
  %5 = call i8 asm sideeffect "lock; cmpxchg $3, $4; mov $2, $1; sete $0", "={ax},=*rm,{ax},q,*m,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(float) %0, float %2, float %3, ptr elementtype(float) %4) nounwind
  store i8 %5, ptr %success, align 1
  %6 = load i8, ptr %success, align 1
  %tobool = trunc i8 %6 to i1
  ret i1 %tobool
}

; CHECK: @cas
; Make sure we're emitting a move from eax.
; CHECK: #APP
; CHECK-NEXT: lock;{{.*}}mov %eax,{{.*}}
; CHECK-NEXT: #NO_APP

define zeroext i1 @cas2(ptr %p, ptr %expected, i1 zeroext %desired) nounwind {
entry:
  %p.addr = alloca ptr, align 8
  %expected.addr = alloca ptr, align 8
  %desired.addr = alloca i8, align 1
  %success = alloca i8, align 1
  store ptr %p, ptr %p.addr, align 8
  store ptr %expected, ptr %expected.addr, align 8
  %frombool = zext i1 %desired to i8
  store i8 %frombool, ptr %desired.addr, align 1
  %0 = load ptr, ptr %expected.addr, align 8
  %1 = load ptr, ptr %expected.addr, align 8
  %2 = load i8, ptr %1, align 1
  %tobool = trunc i8 %2 to i1
  %3 = load i8, ptr %desired.addr, align 1
  %tobool1 = trunc i8 %3 to i1
  %4 = load ptr, ptr %p.addr, align 8
  %5 = call i8 asm sideeffect "lock; cmpxchg $3, $4; mov $2, $1; sete $0", "={ax},=*rm,{ax},q,*m,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i8) %0, i1 %tobool, i1 %tobool1, ptr elementtype(i8) %4) nounwind
  store i8 %5, ptr %success, align 1
  %6 = load i8, ptr %success, align 1
  %tobool2 = trunc i8 %6 to i1
  ret i1 %tobool2
}

; CHECK: @cas2
; Make sure we're emitting a move from %al here.
; CHECK: #APP
; CHECK-NEXT: lock;{{.*}}mov %al,{{.*}}
; CHECK-NEXT: #NO_APP
