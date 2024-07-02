; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -ppc-asm-full-reg-names \
; RUN:      -mtriple powerpc64-ibm-aix-xcoff -mattr=+aix-small-local-exec-tls < %s \
; RUN:      | FileCheck %s --check-prefixes=COMMONCM,SMALL-LOCAL-EXEC-SMALLCM64
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -ppc-asm-full-reg-names \
; RUN:      -mtriple powerpc64-ibm-aix-xcoff --code-model=large \
; RUN:      -mattr=+aix-small-local-exec-tls < %s | FileCheck %s \
; RUN:      --check-prefixes=COMMONCM,SMALL-LOCAL-EXEC-LARGECM64

@mySmallTLS = thread_local(localexec) global [7800 x i64] zeroinitializer, align 8 #0
@mySmallTLS2 = thread_local(localexec) global [3000 x i64] zeroinitializer, align 8 #0
@mySmallTLS3 = thread_local(localexec) global [3000 x i64] zeroinitializer, align 8
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

; Although some global variables are annotated with 'aix-small-tls', because the
; aix-small-local-exec-tls target attribute is turned on, all accesses will use
; a "faster" local-exec sequence directly off the thread pointer.
define i64 @StoreLargeAccess1() {
; COMMONCM-LABEL:    StoreLargeAccess1:
; COMMONCM-NEXT:     # %bb.0: # %entry
; SMALL-LOCAL-EXEC-SMALLCM64:         ld r3, L..C0(r2) # target-flags(ppc-tprel) @mySmallTLS
; SMALL-LOCAL-EXEC-SMALLCM64-NEXT:    li r4, 0
; SMALL-LOCAL-EXEC-SMALLCM64-NEXT:    li r5, 23
; SMALL-LOCAL-EXEC-LARGECM64:         addis r3, L..C0@u(r2)
; SMALL-LOCAL-EXEC-LARGECM64-NEXT:    li r4, 0
; SMALL-LOCAL-EXEC-LARGECM64-NEXT:    li r5, 23
; SMALL-LOCAL-EXEC-LARGECM64-NEXT:    ld r3, L..C0@l(r3)
; COMMONCM:                           ori r4, r4, 53328
; COMMONCM-NEXT:                      add r3, r13, r3
; COMMONCM-NEXT:                      stdx r5, r3, r4
; COMMONCM-NEXT:                      li r3, 55
; COMMONCM-NEXT:                      li r4, 64
; COMMONCM-NEXT:                      std r3, (mySmallTLS2[TL]@le+696)-65536(r13)
; COMMONCM-NEXT:                      li r3, 142
; COMMONCM-NEXT:                      std r4, (mySmallTLS3[TL]@le+20000)-131072(r13)
; COMMONCM-NEXT:                      blr
entry:
  %tls0 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @mySmallTLS)
  %arrayidx = getelementptr inbounds i8, ptr %tls0, i32 53328
  store i64 23, ptr %arrayidx, align 8
  %tls1 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @mySmallTLS2)
  %arrayidx1 = getelementptr inbounds i8, ptr %tls1, i32 696
  store i64 55, ptr %arrayidx1, align 8
  %tls2 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @mySmallTLS3)
  %arrayidx2 = getelementptr inbounds i8, ptr %tls2, i32 20000
  store i64 64, ptr %arrayidx2, align 8
  %load1 = load i64, ptr %arrayidx, align 8
  %load2 = load i64, ptr %arrayidx1, align 8
  %add1 = add i64 %load1, 64
  %add2 = add i64 %add1, %load2
  ret i64 %add2
}

attributes #0 = { "aix-small-tls" }
