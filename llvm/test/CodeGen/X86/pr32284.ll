; RUN: llc -O0 -mtriple=x86_64-unknown -mcpu=skx -o - %s | FileCheck %s --check-prefixes=CHECK,X64
; RUN: llc     -mtriple=x86_64-unknown -mcpu=skx -o - %s | FileCheck %s --check-prefixes=CHECK,X64
; RUN: llc -O0 -mtriple=i686-unknown   -mcpu=skx -o - %s | FileCheck %s --check-prefixes=CHECK,686
; RUN: llc     -mtriple=i686-unknown   -mcpu=skx -o - %s | FileCheck %s --check-prefixes=CHECK,686
; REQUIRES: asserts

@c = external constant i8, align 1

define void @foo() {
; CHECK-LABEL: foo:
; CHECK:    # BB#0: # %entry
; CHECK-DAG:    setne
; CHECK-DAG:    setle
; CHECK:    ret
entry:
  %a = alloca i8, align 1
  %b = alloca i32, align 4
  %0 = load i8, i8* @c, align 1
  %conv = zext i8 %0 to i32
  %sub = sub nsw i32 0, %conv
  %conv1 = sext i32 %sub to i64
  %sub2 = sub nsw i64 0, %conv1
  %conv3 = trunc i64 %sub2 to i8
  %tobool = icmp ne i8 %conv3, 0
  %frombool = zext i1 %tobool to i8
  store i8 %frombool, i8* %a, align 1
  %1 = load i8, i8* @c, align 1
  %tobool4 = icmp ne i8 %1, 0
  %lnot = xor i1 %tobool4, true
  %lnot5 = xor i1 %lnot, true
  %conv6 = zext i1 %lnot5 to i32
  %2 = load i8, i8* @c, align 1
  %conv7 = zext i8 %2 to i32
  %cmp = icmp sle i32 %conv6, %conv7
  %conv8 = zext i1 %cmp to i32
  store i32 %conv8, i32* %b, align 4
  ret void
}

@var_5 = external global i32, align 4
@var_57 = external global i64, align 8
@_ZN8struct_210member_2_0E = external global i64, align 8

define void @f1() {
; CHECK-LABEL: f1:
; CHECK:       # BB#0: # %entry
; CHECK:    sete
; X64:      addq $7093, {{.*}}
; 686:      addl $7093, {{.*}}
; CHECK:    ret
entry:
  %a = alloca i8, align 1
  %0 = load i32, i32* @var_5, align 4
  %conv = sext i32 %0 to i64
  %add = add nsw i64 %conv, 8381627093
  %tobool = icmp ne i64 %add, 0
  %frombool = zext i1 %tobool to i8
  store i8 %frombool, i8* %a, align 1
  %1 = load i32, i32* @var_5, align 4
  %neg = xor i32 %1, -1
  %tobool1 = icmp ne i32 %neg, 0
  %lnot = xor i1 %tobool1, true
  %conv2 = zext i1 %lnot to i64
  %2 = load i32, i32* @var_5, align 4
  %conv3 = sext i32 %2 to i64
  %add4 = add nsw i64 %conv3, 7093
  %cmp = icmp sgt i64 %conv2, %add4
  %conv5 = zext i1 %cmp to i64
  store i64 %conv5, i64* @var_57, align 8
  %3 = load i32, i32* @var_5, align 4
  %neg6 = xor i32 %3, -1
  %tobool7 = icmp ne i32 %neg6, 0
  %lnot8 = xor i1 %tobool7, true
  %conv9 = zext i1 %lnot8 to i64
  store i64 %conv9, i64* @_ZN8struct_210member_2_0E, align 8
  ret void
}


@var_7 = external global i8, align 1

define void @f2() {
; CHECK-LABEL: f2:
; CHECK:       # BB#0: # %entry
; X64:    movzbl {{.*}}(%rip), %[[R:[a-z]*]]
; 686:    movzbl {{.*}}, %[[R:[a-z]*]]
; CHECK:    test{{[qlwb]}} %[[R]], %[[R]]
; CHECK:    sete {{.*}}
; CHECK:    ret
entry:
  %a = alloca i16, align 2
  %0 = load i8, i8* @var_7, align 1
  %conv = zext i8 %0 to i32
  %1 = load i8, i8* @var_7, align 1
  %tobool = icmp ne i8 %1, 0
  %lnot = xor i1 %tobool, true
  %conv1 = zext i1 %lnot to i32
  %xor = xor i32 %conv, %conv1
  %conv2 = trunc i32 %xor to i16
  store i16 %conv2, i16* %a, align 2
  %2 = load i8, i8* @var_7, align 1
  %conv3 = zext i8 %2 to i16
  %tobool4 = icmp ne i16 %conv3, 0
  %lnot5 = xor i1 %tobool4, true
  %conv6 = zext i1 %lnot5 to i32
  %3 = load i8, i8* @var_7, align 1
  %conv7 = zext i8 %3 to i32
  %cmp = icmp eq i32 %conv6, %conv7
  %conv8 = zext i1 %cmp to i32
  %conv9 = trunc i32 %conv8 to i16
  store i16 %conv9, i16* undef, align 2
  ret void
}


@var_13 = external global i32, align 4
@var_16 = external global i32, align 4
@var_46 = external global i32, align 4

define void @f3() #0 {
; CHECK-LABEL: f3:
; X64-DAG: movl    var_13(%rip), {{.*}}
; X64-DAG: movl    var_16(%rip), {{.*}}
; X64-DAG: movl   {{.*}},{{.*}}var_46{{.*}}
; X64: retq
; 686-DAG: movl    var_13, {{.*}}
; 686-DAG: movl    var_16, {{.*}}
; 686-DAG: movl   {{.*}},{{.*}}var_46{{.*}}
; 686: retl
entry:
  %a = alloca i64, align 8
  %0 = load i32, i32* @var_13, align 4
  %neg = xor i32 %0, -1
  %conv = zext i32 %neg to i64
  %1 = load i32, i32* @var_13, align 4
  %tobool = icmp ne i32 %1, 0
  %lnot = xor i1 %tobool, true
  %conv1 = zext i1 %lnot to i64
  %2 = load i32, i32* @var_13, align 4
  %neg2 = xor i32 %2, -1
  %3 = load i32, i32* @var_16, align 4
  %xor = xor i32 %neg2, %3
  %conv3 = zext i32 %xor to i64
  %and = and i64 %conv1, %conv3
  %or = or i64 %conv, %and
  store i64 %or, i64* %a, align 8
  %4 = load i32, i32* @var_13, align 4
  %neg4 = xor i32 %4, -1
  %conv5 = zext i32 %neg4 to i64
  %5 = load i32, i32* @var_13, align 4
  %tobool6 = icmp ne i32 %5, 0
  %lnot7 = xor i1 %tobool6, true
  %conv8 = zext i1 %lnot7 to i64
  %and9 = and i64 %conv8, 0
  %or10 = or i64 %conv5, %and9
  %conv11 = trunc i64 %or10 to i32
  store i32 %conv11, i32* @var_46, align 4
  ret void
}

