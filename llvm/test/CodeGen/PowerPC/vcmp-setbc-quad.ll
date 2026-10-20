; RUN: llc -mcpu=pwr10 -mtriple=powerpc64-ibm-aix -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mcpu=pwr10 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s

define range(i64 -2147483648, 2147483648) i64 @cmpgt(<1 x i128> noundef %a, <1 x i128> noundef %b) local_unnamed_addr {
; CHECK: vcmpgtuq. v2, v3, v2
; CHECK: setbc r3, 4*cr6+lt
entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 2, <1 x i128> %b, <1 x i128> %a)
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

declare i32 @llvm.ppc.altivec.vcmpgtuq.p(i32, <1 x i128>, <1 x i128>)
