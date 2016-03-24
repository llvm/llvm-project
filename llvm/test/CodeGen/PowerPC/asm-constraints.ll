; RUN: llc < %s -mcpu=pwr8 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nounwind
; Check that we accept 'U' and 'X' constraints.
; Generated from following C code:
;
; void foo (int result, char *addr) {
;   __asm__ __volatile__ (
;     "ld%U1%X1 %0,%1\n"
;     "cmpw %0,%0\n"
;     "bne- 1f\n"
;     "1: isync\n"
;     : "=r" (result)
;     : "m"(*addr) : "memory", "cr0");
; }

define void @foo(i32 signext %result, i8* %addr) #0 {

; CHECK-LABEL: @foo
; CHECK: ld [[REG:[0-9]+]], 0(4)
; CHECK: cmpw [[REG]], [[REG]]
; CHECK: bne- 0, .Ltmp[[TMP:[0-9]+]]
; CHECK: .Ltmp[[TMP]]:
; CHECK: isync

entry:
  %result.addr = alloca i32, align 4
  %addr.addr = alloca i8*, align 8
  store i32 %result, i32* %result.addr, align 4
  store i8* %addr, i8** %addr.addr, align 8
  %0 = load i8*, i8** %addr.addr, align 8
  %1 = call i32 asm sideeffect "ld${1:U}${1:X} $0,$1\0Acmpw $0,$0\0Abne- 1f\0A1: isync\0A", "=r,*m,~{memory},~{cr0}"(i8* %0) #1, !srcloc !0
  store i32 %1, i32* %result.addr, align 4
  ret void
}

; Function Attrs: nounwind
; Check that we accept the 'd' constraint.
; Generated from the following C code:
; int foo(double x) {
;   int64_t result;
;   __asm__ __volatile__("fctid %0, %1"
;                        : "=d"(result)
;                        : "d"(x)
;                        : /* No clobbers */);
;   return result;
; }
define signext i32 @bar(double %x) #0 {

; CHECK-LABEL: @bar
; CHECK: fctid 0, 1
entry:
  %x.addr = alloca double, align 8
  %result = alloca i64, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double, double* %x.addr, align 8
  %1 = call i64 asm sideeffect "fctid $0, $1", "=d,d"(double %0) #1, !srcloc !1
  store i64 %1, i64* %result, align 8
  %2 = load i64, i64* %result, align 8
  %conv = trunc i64 %2 to i32
  ret i32 %conv
}


attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }

attributes #1 = { nounwind }

!0 = !{i32 67, i32 91, i32 110, i32 126}
!1 = !{i32 84}
