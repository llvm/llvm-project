; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx | FileCheck %s

; These tests were generated from simplified libm C code.
; When compiled for the x86_64-linux-android target,
; long double is mapped to f128 type that should be passed
; in SSE registers. When the f128 type calling convention
; problem was fixed, old llvm code failed to handle f128 values
; in several f128/i128 type operations. These unit tests hopefully
; will catch regression in any future change in this area.
; To modified or enhance these test cases, please consult libm
; code pattern and compile with -target x86_64-linux-android
; to generate IL. The __float128 keyword if not accepted by
; clang, just define it to "long double".
;

; typedef long double __float128;
; union IEEEl2bits {
;   __float128 e;
;   struct {
;     unsigned long manl :64;
;     unsigned long manh :48;
;     unsigned int exp :15;
;     unsigned int sign :1;
;   } bits;
;   struct {
;     unsigned long manl :64;
;     unsigned long manh :48;
;     unsigned int expsign :16;
;   } xbits;
; };

; C code:
; void foo(__float128 x);
; void TestUnionLD1(__float128 s, unsigned long n) {
;      union IEEEl2bits u;
;      __float128 w;
;      u.e = s;
;      u.bits.manh = n;
;      w = u.e;
;      foo(w);
; }
define void @TestUnionLD1(fp128 %s, i64 %n) #0 {
entry:
  %0 = bitcast fp128 %s to i128
  %1 = zext i64 %n to i128
  %bf.value = shl nuw i128 %1, 64
  %bf.shl = and i128 %bf.value, 5192296858534809181786422619668480
  %bf.clear = and i128 %0, -5192296858534809181786422619668481
  %bf.set = or i128 %bf.shl, %bf.clear
  %2 = bitcast i128 %bf.set to fp128
  tail call void @foo(fp128 %2) #2
  ret void
; CHECK-LABEL: TestUnionLD1:
; CHECK:       movaps %xmm0, -24(%rsp)
; CHECK-NEXT:  movq -24(%rsp), %rax
; CHECK-NEXT:  movabsq $281474976710655, %rcx
; CHECK-NEXT:  andq %rdi, %rcx
; CHECK-NEXT:  movabsq $-281474976710656, %rdx
; CHECK-NEXT:  andq -16(%rsp), %rdx
; CHECK-NEXT:  movq %rax, -40(%rsp)
; CHECK-NEXT:  orq %rcx, %rdx
; CHECK-NEXT:  movq %rdx, -32(%rsp)
; CHECK-NEXT:  movaps -40(%rsp), %xmm0
; CHECK-NEXT:  jmp foo
}

; C code:
; __float128 TestUnionLD2(__float128 s) {
;      union IEEEl2bits u;
;      __float128 w;
;      u.e = s;
;      u.bits.manl = 0;
;      w = u.e;
;      return w;
; }
define fp128 @TestUnionLD2(fp128 %s) #0 {
entry:
  %0 = bitcast fp128 %s to i128
  %bf.clear = and i128 %0, -18446744073709551616
  %1 = bitcast i128 %bf.clear to fp128
  ret fp128 %1
; CHECK-LABEL: TestUnionLD2:
; CHECK:       movaps %xmm0, -24(%rsp)
; CHECK-NEXT:  movq -16(%rsp), %rax
; CHECK-NEXT:  movq %rax, -32(%rsp)
; CHECK-NEXT:  movq $0, -40(%rsp)
; CHECK-NEXT:  movaps -40(%rsp), %xmm0
; CHECK-NEXT:  retq
}

; C code:
; __float128 TestI128_1(__float128 x)
; {
;  union IEEEl2bits z;
;  z.e = x;
;  z.bits.sign = 0;
;  return (z.e < 0.1L) ? 1.0L : 2.0L;
; }
define fp128 @TestI128_1(fp128 %x) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %bf.clear = and i128 %0, 170141183460469231731687303715884105727
  %1 = bitcast i128 %bf.clear to fp128
  %cmp = fcmp olt fp128 %1, 0xL999999999999999A3FFB999999999999
  %cond = select i1 %cmp, fp128 0xL00000000000000003FFF000000000000, fp128 0xL00000000000000004000000000000000
  ret fp128 %cond
; CHECK-LABEL: TestI128_1:
; CHECK:       movaps %xmm0,
; CHECK:       movabsq $9223372036854775807,
; CHECK:       callq __lttf2
; CHECK:       testl %eax, %eax
; CHECK:       movaps {{.*}}, %xmm0
; CHECK:       retq
}

; C code:
; __float128 TestI128_2(__float128 x, __float128 y)
; {
;  unsigned short hx;
;  union IEEEl2bits ge_u;
;  ge_u.e = x;
;  hx = ge_u.xbits.expsign;
;  return (hx & 0x8000) == 0 ? x : y;
; }
define fp128 @TestI128_2(fp128 %x, fp128 %y) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %cmp = icmp sgt i128 %0, -1
  %cond = select i1 %cmp, fp128 %x, fp128 %y
  ret fp128 %cond
; CHECK-LABEL: TestI128_2:
; CHECK:       movaps %xmm0, -24(%rsp)
; CHECK-NEXT:  cmpq $0, -16(%rsp)
; CHECK-NEXT:  jns
; CHECK:       movaps %xmm1, %xmm0
; CHECK:       retq
}

; C code:
; __float128 TestI128_3(__float128 x, int *ex)
; {
;  union IEEEl2bits u;
;  u.e = x;
;  if (u.bits.exp == 0) {
;    u.e *= 0x1.0p514;
;    u.bits.exp = 0x3ffe;
;  }
;  return (u.e);
; }
define fp128 @TestI128_3(fp128 %x, i32* nocapture readnone %ex) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %bf.cast = and i128 %0, 170135991163610696904058773219554885632
  %cmp = icmp eq i128 %bf.cast, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %mul = fmul fp128 %x, 0xL00000000000000004201000000000000
  %1 = bitcast fp128 %mul to i128
  %bf.clear4 = and i128 %1, -170135991163610696904058773219554885633
  %bf.set = or i128 %bf.clear4, 85060207136517546210586590865283612672
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %u.sroa.0.0 = phi i128 [ %bf.set, %if.then ], [ %0, %entry ]
  %2 = bitcast i128 %u.sroa.0.0 to fp128
  ret fp128 %2
; CHECK-LABEL: TestI128_3:
; CHECK:       movaps %xmm0,
; CHECK:       movabsq $9223090561878065152,
; CHECK:       testq
; CHECK:       callq __multf3
; CHECK-NEXT:  movaps %xmm0
; CHECK:       movabsq $-9223090561878065153,
; CHECK:       movabsq $4611123068473966592,
; CHECK:       retq
}

; C code:
; __float128 TestI128_4(__float128 x)
; {
;  union IEEEl2bits u;
;  __float128 df;
;  u.e = x;
;  u.xbits.manl = 0;
;  df = u.e;
;  return x + df;
; }
define fp128 @TestI128_4(fp128 %x) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %bf.clear = and i128 %0, -18446744073709551616
  %1 = bitcast i128 %bf.clear to fp128
  %add = fadd fp128 %1, %x
  ret fp128 %add
; CHECK-LABEL: TestI128_4:
; CHECK:       movaps %xmm0, %xmm1
; CHECK-NEXT:  movaps %xmm1, 16(%rsp)
; CHECK-NEXT:  movq 24(%rsp), %rax
; CHECK-NEXT:  movq %rax, 8(%rsp)
; CHECK-NEXT:  movq $0, (%rsp)
; CHECK-NEXT:  movaps (%rsp), %xmm0
; CHECK-NEXT:  callq __addtf3
; CHECK:       retq
}

@v128 = common global i128 0, align 16
@v128_2 = common global i128 0, align 16

; C code:
; unsigned __int128 v128, v128_2;
; void TestShift128_2() {
;   v128 = ((v128 << 96) | v128_2);
; }
define void @TestShift128_2() #2 {
entry:
  %0 = load i128, i128* @v128, align 16
  %shl = shl i128 %0, 96
  %1 = load i128, i128* @v128_2, align 16
  %or = or i128 %shl, %1
  store i128 %or, i128* @v128, align 16
  ret void
; CHECK-LABEL: TestShift128_2:
; CHECK:       movq v128(%rip), %rax
; CHECK-NEXT:  shlq $32, %rax
; CHECK-NEXT:  movq v128_2(%rip), %rcx
; CHECK-NEXT:  orq v128_2+8(%rip), %rax
; CHECK-NEXT:  movq %rcx, v128(%rip)
; CHECK-NEXT:  movq %rax, v128+8(%rip)
; CHECK-NEXT:  retq
}

define fp128 @acosl(fp128 %x) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %bf.clear = and i128 %0, -18446744073709551616
  %1 = bitcast i128 %bf.clear to fp128
  %add = fadd fp128 %1, %x
  ret fp128 %add
; CHECK-LABEL: acosl:
; CHECK:       movaps %xmm0, %xmm1
; CHECK-NEXT:  movaps %xmm1, 16(%rsp)
; CHECK-NEXT:  movq 24(%rsp), %rax
; CHECK-NEXT:  movq %rax, 8(%rsp)
; CHECK-NEXT:  movq $0, (%rsp)
; CHECK-NEXT:  movaps (%rsp), %xmm0
; CHECK-NEXT:  callq __addtf3
; CHECK:       retq
}

; Compare i128 values and check i128 constants.
define fp128 @TestComp(fp128 %x, fp128 %y) #0 {
entry:
  %0 = bitcast fp128 %x to i128
  %cmp = icmp sgt i128 %0, -1
  %cond = select i1 %cmp, fp128 %x, fp128 %y
  ret fp128 %cond
; CHECK-LABEL: TestComp:
; CHECK:       movaps %xmm0, -24(%rsp)
; CHECK-NEXT:  cmpq $0, -16(%rsp)
; CHECK-NEXT:  jns
; CHECK:       movaps %xmm1, %xmm0
; CHECK:       retq
}

declare void @foo(fp128) #1

; Test logical operations on fp128 values.
define fp128 @TestFABS_LD(fp128 %x) #0 {
entry:
  %call = tail call fp128 @fabsl(fp128 %x) #2
  ret fp128 %call
; CHECK-LABEL: TestFABS_LD
; CHECK:       andps {{.*}}, %xmm0
; CHECK-NEXT:  retq
}

declare fp128 @fabsl(fp128) #1

declare fp128 @copysignl(fp128, fp128) #1

; Test more complicated logical operations generated from copysignl.
define void @TestCopySign({ fp128, fp128 }* noalias nocapture sret %agg.result, { fp128, fp128 }* byval nocapture readonly align 16 %z) #0 {
entry:
  %z.realp = getelementptr inbounds { fp128, fp128 }, { fp128, fp128 }* %z, i64 0, i32 0
  %z.real = load fp128, fp128* %z.realp, align 16
  %z.imagp = getelementptr inbounds { fp128, fp128 }, { fp128, fp128 }* %z, i64 0, i32 1
  %z.imag4 = load fp128, fp128* %z.imagp, align 16
  %cmp = fcmp ogt fp128 %z.real, %z.imag4
  %sub = fsub fp128 %z.imag4, %z.imag4
  br i1 %cmp, label %if.then, label %cleanup

if.then:                                          ; preds = %entry
  %call = tail call fp128 @fabsl(fp128 %sub) #2
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.then
  %z.real.sink = phi fp128 [ %z.real, %if.then ], [ %sub, %entry ]
  %call.sink = phi fp128 [ %call, %if.then ], [ %z.real, %entry ]
  %call5 = tail call fp128 @copysignl(fp128 %z.real.sink, fp128 %z.imag4) #2
  %0 = getelementptr inbounds { fp128, fp128 }, { fp128, fp128 }* %agg.result, i64 0, i32 0
  %1 = getelementptr inbounds { fp128, fp128 }, { fp128, fp128 }* %agg.result, i64 0, i32 1
  store fp128 %call.sink, fp128* %0, align 16
  store fp128 %call5, fp128* %1, align 16
  ret void
; CHECK-LABEL: TestCopySign
; CHECK-NOT:   call
; CHECK:       callq __subtf3
; CHECK-NOT:   call
; CHECK:       callq __gttf2
; CHECK-NOT:   call
; CHECK:       andps {{.*}}, %xmm0
; CHECK:       retq
}


attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+ssse3,+sse3,+popcnt,+sse,+sse2,+sse4.1,+sse4.2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+ssse3,+sse3,+popcnt,+sse,+sse2,+sse4.1,+sse4.2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
