; Test for the conservative assembly handling mode used by KMSAN.
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0                    \
; RUN: -msan-handle-asm-conservative=0 -S -passes=msan 2>&1 | FileCheck        \
; RUN: "-check-prefix=CHECK" %s
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0                    \
; RUN: -msan-handle-asm-conservative=1 -S -passes=msan 2>&1 | FileCheck        \
; RUN: "-check-prefixes=CHECK,CHECK-CONS" %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The IR below was generated from the following source:
;  int main() {
;    bool bit;
;    unsigned long value = 2;
;    long nr = 0;
;    unsigned long *addr = &value;
;    asm("btsq %2, %1; setc %0" : "=qm" (bit), "=m" (addr): "Ir" (nr));
;    if (bit)
;      return 0;
;    else
;      return 1;
;  }
;
; In the regular instrumentation mode MSan is unable to understand that |bit|
; is initialized by the asm() call, and therefore reports a false positive on
; the if-statement.
; The conservative assembly handling mode initializes every memory location
; passed by pointer into an asm() call. This prevents false positive reports,
; but may introduce false negatives.
;
; This test makes sure that the conservative mode unpoisons the shadow of |bit|
; by writing 0 to it.

define dso_local i32 @main() sanitize_memory {
entry:
  %retval = alloca i32, align 4
  %bit = alloca i8, align 1
  %value = alloca i64, align 8
  %nr = alloca i64, align 8
  %addr = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store i64 2, ptr %value, align 8
  store i64 0, ptr %nr, align 8
  store ptr %value, ptr %addr, align 8
  %0 = load i64, ptr %nr, align 8
  call void asm "btsq $2, $1; setc $0", "=*qm,=*m,Ir,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i8) %bit, ptr elementtype(ptr) %addr, i64 %0)
  %1 = load i8, ptr %bit, align 1
  %tobool = trunc i8 %1 to i1
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ret i32 0

if.else:                                          ; preds = %entry
  ret i32 1
}

; %nr is first poisoned, then unpoisoned (written to). Need to optimize this in the future.
; CHECK: call void @__msan_poison_alloca(ptr %nr{{.*}})
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %nr)

; Hooks for inputs usually go before the assembly statement. But here we have none,
; because %nr is passed by value. However we check %nr for being initialized.
; CHECK-CONS: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %nr)

; In the conservative mode, call the store hooks for %bit and %addr:
; CHECK-CONS: call void @__msan_instrument_asm_store(ptr %bit, i64 1)
; CHECK-CONS: call void @__msan_instrument_asm_store(ptr %addr, i64 8)

; Landing pad for the %nr check above.
; CHECK-CONS: call void @__msan_warning

; CHECK: call void asm "btsq $2, $1; setc $0"

; Calculating the shadow offset of %bit.
; CHECKz: [[PTR:%.*]] = ptrtoint {{.*}} %bit to i64
; CHECKz: [[SH_NUM:%.*]] = xor i64 [[PTR]]
; CHECKz: [[SHADOW:%.*]] = inttoptr i64 [[SH_NUM]] {{.*}}

; CHECK: [[META:%.*]] = call {{.*}} @__msan_metadata_ptr_for_load_1(ptr %bit)
; CHECK: [[SHADOW:%.*]] = extractvalue { ptr, ptr } [[META]], 0

; Now load the shadow value for the boolean.
; CHECK: [[MSLD:%.*]] = load {{.*}} [[SHADOW]]
; CHECK: [[MSPROP:%.*]] = trunc i8 [[MSLD]] to i1

; Is the shadow poisoned?
; CHECK: br i1 [[MSPROP]], label %[[IFTRUE:.*]], label {{.*}}

; If yes, raise a warning.
; CHECK: [[IFTRUE]]:
; CHECK: call void @__msan_warning

