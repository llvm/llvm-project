; NOTE: Scaled memory forms should be selected at all llc optimization levels.
; RUN: llc -O0 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O1 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O2 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O3 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O2 -mtriple=loongarch64 -mcpu=la464 -mtune=la664 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O2 -mtriple=loongarch64 -mcpu=la664 -mtune=la464 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA64
; RUN: llc -O2 -mtriple=loongarch32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LA32

; Verify load-extension semantics and store widths, shifts 1-4, and typed GEP.
; LA32 must not select the LA64-only ALSL.D pattern.

define i64 @scaled_zext_i8_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_zext_i8_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.bu {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_zext_i8_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i8, ptr %addr, align 1
  %ext = zext i8 %value to i64
  ret i64 %ext
}

define i64 @scaled_sext_i8_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_sext_i8_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.b {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_sext_i8_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i8, ptr %addr, align 1
  %ext = sext i8 %value to i64
  ret i64 %ext
}

define i64 @scaled_zext_i16_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_zext_i16_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.hu {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_zext_i16_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i16, ptr %addr, align 2
  %ext = zext i16 %value to i64
  ret i64 %ext
}

define i64 @scaled_sext_i16_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_sext_i16_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.h {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_sext_i16_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i16, ptr %addr, align 2
  %ext = sext i16 %value to i64
  ret i64 %ext
}

define i64 @scaled_zext_i32_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_zext_i32_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.wu {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_zext_i32_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 2
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i32, ptr %addr, align 4
  %ext = zext i32 %value to i64
  ret i64 %ext
}

define i64 @scaled_sext_i32_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_sext_i32_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.w {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_sext_i32_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 2
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i32, ptr %addr, align 4
  %ext = sext i32 %value to i64
  ret i64 %ext
}

define i64 @scaled_i64_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_i64_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.d {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_i64_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 3
  %addr = getelementptr i8, ptr %base, i64 %scaled
  %value = load i64, ptr %addr, align 8
  ret i64 %value
}

define void @scaled_i8_store(ptr %base, i64 %idx, i8 %value) {
; LA64-LABEL: scaled_i8_store:
; LA64:       alsl.d
; LA64-NEXT:  st.b {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_i8_store:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  store i8 %value, ptr %addr, align 1
  ret void
}

define void @scaled_i16_store(ptr %base, i64 %idx, i16 %value) {
; LA64-LABEL: scaled_i16_store:
; LA64:       alsl.d
; LA64-NEXT:  st.h {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_i16_store:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 1
  %addr = getelementptr i8, ptr %base, i64 %scaled
  store i16 %value, ptr %addr, align 2
  ret void
}

define void @scaled_i32_store(ptr %base, i64 %idx, i32 %value) {
; LA64-LABEL: scaled_i32_store:
; LA64:       alsl.d
; LA64-NEXT:  st.w {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_i32_store:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 2
  %addr = getelementptr i8, ptr %base, i64 %scaled
  store i32 %value, ptr %addr, align 4
  ret void
}

define void @scaled_i64_store(ptr %base, i64 %idx, i64 %value) {
; LA64-LABEL: scaled_i64_store:
; LA64:       alsl.d
; LA64-NEXT:  st.d {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_i64_store:
; LA32-NOT:   alsl.d
; LA32:       ret
  %scaled = shl i64 %idx, 3
  %addr = getelementptr i8, ptr %base, i64 %scaled
  store i64 %value, ptr %addr, align 8
  ret void
}

; The immediate field uimm2_plus1 permits a scale of 4.
define i64 @scaled_shift4_load(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_shift4_load:
; LA64:       alsl.d
; LA64-NEXT:  ld.d {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_shift4_load:
; LA32-NOT:   alsl.d
; LA32:       ret
  %off = shl i64 %idx, 4
  %addr = getelementptr i8, ptr %base, i64 %off
  %v = load i64, ptr %addr, align 8
  ret i64 %v
}

; Typed GEP should become a byte-scaled address.
define i64 @scaled_gep_i32_zext(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_gep_i32_zext:
; LA64:       alsl.d
; LA64-NEXT:  ld.wu {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_gep_i32_zext:
; LA32-NOT:   alsl.d
; LA32:       ret
  %addr = getelementptr inbounds i32, ptr %base, i64 %idx
  %v = load i32, ptr %addr, align 4
  %ext = zext i32 %v to i64
  ret i64 %ext
}

; Swapped add operand order must not require another pattern.
define i64 @scaled_commuted_add_i64(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_commuted_add_i64:
; LA64:       alsl.d
; LA64-NEXT:  ld.d {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_commuted_add_i64:
; LA32-NOT:   alsl.d
; LA32:       ret
  %off = shl i64 %idx, 3
  %baseint = ptrtoint ptr %base to i64
  %addrint = add i64 %off, %baseint
  %addr = inttoptr i64 %addrint to ptr
  %v = load i64, ptr %addr, align 8
  ret i64 %v
}

; Volatile memory operations must retain their memory semantics.
define i64 @scaled_volatile_zext_i32(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_volatile_zext_i32:
; LA64:       alsl.d
; LA64-NEXT:  ld.wu {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_volatile_zext_i32:
; LA32-NOT:   alsl.d
; LA32:       ret
  %off = shl i64 %idx, 2
  %addr = getelementptr i8, ptr %base, i64 %off
  %v = load volatile i32, ptr %addr, align 4
  %ext = zext i32 %v to i64
  ret i64 %ext
}

define void @scaled_volatile_i32_store(ptr %base, i64 %idx, i32 %value) {
; LA64-LABEL: scaled_volatile_i32_store:
; LA64:       alsl.d
; LA64-NEXT:  st.w {{.*}}
; LA64:       ret
; LA32-LABEL: scaled_volatile_i32_store:
; LA32-NOT:   alsl.d
; LA32:       ret
  %off = shl i64 %idx, 2
  %addr = getelementptr i8, ptr %base, i64 %off
  store volatile i32 %value, ptr %addr, align 4
  ret void
}

; Shift amounts outside uimm2_plus1 must not be selected as a single ALSL.D.
define i64 @scaled_shift0_negative(ptr %base, i64 %idx) {
; LA64-LABEL: scaled_shift0_negative:
; LA64-NOT:   alsl.d
; LA64:       ret
; LA32-LABEL: scaled_shift0_negative:
; LA32-NOT:   alsl.d
; LA32:       ret
  %addr = getelementptr i8, ptr %base, i64 %idx
  %v = load i64, ptr %addr, align 1
  ret i64 %v
}
