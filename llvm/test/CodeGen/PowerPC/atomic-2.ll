; RUN: llc < %s -ppc-asm-full-reg-names -march=ppc64 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-BE
; RUN: llc < %s -ppc-asm-full-reg-names -march=ppc64le -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-LE
; RUN: llc < %s -ppc-asm-full-reg-names -march=ppc64 -mcpu=pwr7 | FileCheck %s
; RUN: llc < %s -ppc-asm-full-reg-names -march=ppc64 -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-P8U

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange_and_add:
; CHECK: ldarx
  %tmp = atomicrmw add i64* %mem, i64 %val monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange_and_add8(i8* %mem, i8 %val) nounwind {
; CHECK-LABEL: exchange_and_add8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmp = atomicrmw add i8* %mem, i8 %val monotonic
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange_and_add16(i16* %mem, i16 %val) nounwind {
; CHECK-LABEL: exchange_and_add16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmp = atomicrmw add i16* %mem, i16 %val monotonic
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp:
; CHECK: ldarx
  %tmppair = cmpxchg i64* %mem, i64 0, i64 1 monotonic monotonic
  %tmp = extractvalue { i64, i1 } %tmppair, 0
; CHECK: stdcx.
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange_and_cmp8(i8* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmppair = cmpxchg i8* %mem, i8 0, i8 1 monotonic monotonic
  %tmp = extractvalue { i8, i1 } %tmppair, 0
; CHECK-P8U: stbcx.
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange_and_cmp16(i16* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmppair = cmpxchg i16* %mem, i16 0, i16 1 monotonic monotonic
  %tmp = extractvalue { i16, i1 } %tmppair, 0
; CHECK-P8U: sthcx.
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange:
; CHECK: ldarx
  %tmp = atomicrmw xchg i64* %mem, i64 1 monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i8 @exchange8(i8* %mem, i8 %val) nounwind {
; CHECK-LABEL: exchange8:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lbarx
  %tmp = atomicrmw xchg i8* %mem, i8 1 monotonic
; CHECK-P8U: stbcx.
  ret i8 %tmp
}

define i16 @exchange16(i16* %mem, i16 %val) nounwind {
; CHECK-LABEL: exchange16:
; CHECK-BE: xori
; CHECK-LE-NOT: xori
; CHECK-P8U: lharx
  %tmp = atomicrmw xchg i16* %mem, i16 1 monotonic
; CHECK-P8U: sthcx.
  ret i16 %tmp
}

define void @atomic_store(i64* %mem, i64 %val) nounwind {
entry:
; CHECK: @atomic_store
  store atomic i64 %val, i64* %mem release, align 64
; CHECK: lwsync
; CHECK-NOT: stdcx
; CHECK: std
  ret void
}

define i64 @atomic_load(i64* %mem) nounwind {
entry:
; CHECK: @atomic_load
  %tmp = load atomic i64, i64* %mem acquire, align 64
; CHECK-NOT: ldarx
; CHECK: ld [[VAL:r[0-9]+]]
; CHECK: cmpd [[CR:cr[0-9]+]], [[VAL]], [[VAL]]
; CHECK: bne- [[CR]], .+4
; CHECK: isync
  ret i64 %tmp
}

