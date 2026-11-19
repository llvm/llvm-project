; REQUIRES: aarch64-registered-target
; Strip blank lines and inline attribute comments so the declaration
; stream is contiguous and can be checked with -NEXT: no unexpected call
; can sneak in between two checked declarations.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux-gnu < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=LINUX %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-apple-macosx < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=DARWIN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-pc-windows-msvc < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=WINMSVC %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-w64-windows-gnu < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=MINGW %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux-android < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=ANDROID %s

; LINUX: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; LINUX-NEXT: @__stack_chk_guard = external global ptr
; LINUX-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; LINUX-NEXT: declare void @_Unwind_Resume(ptr)
; LINUX-NEXT: declare i128 @__aarch64_cas16_acq(i128, i128, ptr)
; LINUX-NEXT: declare i128 @__aarch64_cas16_acq_rel(i128, i128, ptr)
; LINUX-NEXT: declare i128 @__aarch64_cas16_rel(i128, i128, ptr)
; LINUX-NEXT: declare i128 @__aarch64_cas16_relax(i128, i128, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_cas1_acq(i8 zeroext, i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_cas1_acq_rel(i8 zeroext, i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_cas1_rel(i8 zeroext, i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_cas1_relax(i8 zeroext, i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_cas2_acq(i16 zeroext, i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_cas2_acq_rel(i16 zeroext, i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_cas2_rel(i16 zeroext, i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_cas2_relax(i16 zeroext, i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_cas4_acq(i32, i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_cas4_acq_rel(i32, i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_cas4_rel(i32, i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_cas4_relax(i32, i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_cas8_acq(i64, i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_cas8_acq_rel(i64, i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_cas8_rel(i64, i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_cas8_relax(i64, i64, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldadd1_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldadd1_relax(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldadd2_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldadd2_relax(i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldadd4_acq(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldadd4_acq_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldadd4_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldadd4_relax(i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldadd8_acq(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldadd8_acq_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldadd8_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldadd8_relax(i64, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldclr1_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldclr1_relax(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldclr2_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldclr2_relax(i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldclr4_acq(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldclr4_acq_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldclr4_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldclr4_relax(i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldclr8_acq(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldclr8_acq_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldclr8_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldclr8_relax(i64, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldeor1_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldeor1_relax(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldeor2_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldeor2_relax(i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldeor4_acq(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldeor4_acq_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldeor4_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldeor4_relax(i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldeor8_acq(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldeor8_acq_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldeor8_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldeor8_relax(i64, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldset1_acq(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldset1_acq_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldset1_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_ldset1_relax(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldset2_acq(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldset2_acq_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldset2_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_ldset2_relax(i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldset4_acq(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldset4_acq_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldset4_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_ldset4_relax(i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldset8_acq(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldset8_acq_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldset8_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_ldset8_relax(i64, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_swp1_acq(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_swp1_acq_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_swp1_rel(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i8 @__aarch64_swp1_relax(i8 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_swp2_acq(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_swp2_acq_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_swp2_rel(i16 zeroext, ptr)
; LINUX-NEXT: declare zeroext i16 @__aarch64_swp2_relax(i16 zeroext, ptr)
; LINUX-NEXT: declare i32 @__aarch64_swp4_acq(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_swp4_acq_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_swp4_rel(i32, ptr)
; LINUX-NEXT: declare i32 @__aarch64_swp4_relax(i32, ptr)
; LINUX-NEXT: declare i64 @__aarch64_swp8_acq(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_swp8_acq_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_swp8_rel(i64, ptr)
; LINUX-NEXT: declare i64 @__aarch64_swp8_relax(i64, ptr)
; LINUX-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_get_current_vg()
; LINUX-NEXT: declare ptr @__arm_sc_memchr(ptr, i32, i64)
; LINUX-NEXT: declare ptr @__arm_sc_memcpy(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @__arm_sc_memmove(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @__arm_sc_memset(ptr, i32, i64)
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_restore(ptr)
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_save(ptr)
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x2 { i64, i64 } @__arm_sme_state()
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_sme_state_size()
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_restore(ptr)
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
; LINUX-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_za_disable()
; LINUX-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; LINUX-NEXT: declare i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; LINUX-NEXT: declare i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; LINUX-NEXT: declare i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; LINUX-NEXT: declare i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; LINUX-NEXT: declare i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; LINUX-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; LINUX-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; LINUX-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; LINUX-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; LINUX-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; LINUX-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; LINUX-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; LINUX-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; LINUX-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; LINUX-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; LINUX-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; LINUX-NEXT: declare void @__clear_cache(ptr, ptr)
; LINUX-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare void @__cxa_end_cleanup()
; LINUX-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__exp2l_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__expl_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare void @__llvm_deoptimize(...)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; LINUX-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__log10l_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__log2l_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__logl_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; LINUX-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; LINUX-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; LINUX-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__powl_finite(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; LINUX-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; LINUX-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; LINUX-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; LINUX-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; LINUX-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; LINUX-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; LINUX-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @acosl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @asinl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @atan2l(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @atanl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @cabs(...)
; LINUX-NEXT: declare void @cabsf(...)
; LINUX-NEXT: declare void @cabsl(...)
; LINUX-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @cbrtl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @ceill(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @copysignl(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @coshl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @cosl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @exp10l(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @exp2l(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @expl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @fdiml(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @fegetenv(ptr)
; LINUX-NEXT: declare i32 @fegetmode(ptr)
; LINUX-NEXT: declare i32 @fesetenv(ptr)
; LINUX-NEXT: declare i32 @fesetmode(ptr)
; LINUX-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @floorl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @fmal(fp128, fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaximum_numl(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaximuml(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaxl(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminimum_numl(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminimuml(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminl(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @fmodl(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare fp128 @frexpl(fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @ldexpl(fp128, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @log10l(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @log2l(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @logl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrintl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lroundl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @memmove(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @memset(ptr, i32, i64)
; LINUX-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare fp128 @modfl(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) fp128 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; LINUX-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @nearbyintl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @nexttoward(double, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @nexttowardf(float, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @nexttowardl(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @powl(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @remainderl(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare fp128 @remquol(fp128, fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @rintl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @roundevenl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @roundl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @scalblnl(fp128, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @scalbnl(fp128, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosl(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16), ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @sinhl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @sinl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) fp128 @sqrtl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @tanhl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @tanl(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @tgammal(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @truncl(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare ptr @wmemset(ptr, i32, i64)
; LINUX-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; LINUX-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; LINUX-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; LINUX-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; LINUX-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; LINUX-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; DARWIN: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; DARWIN-NEXT: @__stack_chk_guard = external global ptr
; DARWIN-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; DARWIN-NEXT: declare void @_Unwind_Resume(ptr)
; DARWIN-NEXT: declare i128 @__aarch64_cas16_acq(i128, i128, ptr)
; DARWIN-NEXT: declare i128 @__aarch64_cas16_acq_rel(i128, i128, ptr)
; DARWIN-NEXT: declare i128 @__aarch64_cas16_rel(i128, i128, ptr)
; DARWIN-NEXT: declare i128 @__aarch64_cas16_relax(i128, i128, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_cas1_acq(i8 zeroext, i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_cas1_acq_rel(i8 zeroext, i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_cas1_rel(i8 zeroext, i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_cas1_relax(i8 zeroext, i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_cas2_acq(i16 zeroext, i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_cas2_acq_rel(i16 zeroext, i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_cas2_rel(i16 zeroext, i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_cas2_relax(i16 zeroext, i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_cas4_acq(i32, i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_cas4_acq_rel(i32, i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_cas4_rel(i32, i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_cas4_relax(i32, i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_cas8_acq(i64, i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_cas8_acq_rel(i64, i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_cas8_rel(i64, i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_cas8_relax(i64, i64, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldadd1_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldadd1_relax(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldadd2_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldadd2_relax(i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldadd4_acq(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldadd4_acq_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldadd4_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldadd4_relax(i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldadd8_acq(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldadd8_acq_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldadd8_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldadd8_relax(i64, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldclr1_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldclr1_relax(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldclr2_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldclr2_relax(i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldclr4_acq(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldclr4_acq_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldclr4_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldclr4_relax(i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldclr8_acq(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldclr8_acq_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldclr8_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldclr8_relax(i64, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldeor1_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldeor1_relax(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldeor2_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldeor2_relax(i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldeor4_acq(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldeor4_acq_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldeor4_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldeor4_relax(i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldeor8_acq(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldeor8_acq_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldeor8_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldeor8_relax(i64, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldset1_acq(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldset1_acq_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldset1_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_ldset1_relax(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldset2_acq(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldset2_acq_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldset2_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_ldset2_relax(i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldset4_acq(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldset4_acq_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldset4_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_ldset4_relax(i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldset8_acq(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldset8_acq_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldset8_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_ldset8_relax(i64, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_swp1_acq(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_swp1_acq_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_swp1_rel(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i8 @__aarch64_swp1_relax(i8 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_swp2_acq(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_swp2_acq_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_swp2_rel(i16 zeroext, ptr)
; DARWIN-NEXT: declare zeroext i16 @__aarch64_swp2_relax(i16 zeroext, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_swp4_acq(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_swp4_acq_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_swp4_rel(i32, ptr)
; DARWIN-NEXT: declare i32 @__aarch64_swp4_relax(i32, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_swp8_acq(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_swp8_acq_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_swp8_rel(i64, ptr)
; DARWIN-NEXT: declare i64 @__aarch64_swp8_relax(i64, ptr)
; DARWIN-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_get_current_vg()
; DARWIN-NEXT: declare ptr @__arm_sc_memchr(ptr, i32, i64)
; DARWIN-NEXT: declare ptr @__arm_sc_memcpy(ptr, ptr, i64)
; DARWIN-NEXT: declare ptr @__arm_sc_memmove(ptr, ptr, i64)
; DARWIN-NEXT: declare ptr @__arm_sc_memset(ptr, i32, i64)
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_restore(ptr)
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_save(ptr)
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x2 { i64, i64 } @__arm_sme_state()
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_sme_state_size()
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_restore(ptr)
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
; DARWIN-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_za_disable()
; DARWIN-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; DARWIN-NEXT: declare i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; DARWIN-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; DARWIN-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; DARWIN-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; DARWIN-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; DARWIN-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; DARWIN-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; DARWIN-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; DARWIN-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; DARWIN-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; DARWIN-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; DARWIN-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; DARWIN-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; DARWIN-NEXT: declare void @__clear_cache(ptr, ptr)
; DARWIN-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare void @__cxa_end_cleanup()
; DARWIN-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare void @__llvm_deoptimize(...)
; DARWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; DARWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; DARWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; DARWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; DARWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; DARWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; DARWIN-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; DARWIN-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; DARWIN-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; DARWIN-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; DARWIN-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; DARWIN-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; DARWIN-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; DARWIN-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; DARWIN-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; DARWIN-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; DARWIN-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare void @bzero(ptr writeonly captures(none), i64) [[ARGMEM_WRITE_ATTRS]]
; DARWIN-NEXT: declare void @cabs(...)
; DARWIN-NEXT: declare void @cabsf(...)
; DARWIN-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i32 @fegetenv(ptr)
; DARWIN-NEXT: declare i32 @fegetmode(ptr)
; DARWIN-NEXT: declare i32 @fesetenv(ptr)
; DARWIN-NEXT: declare i32 @fesetmode(ptr)
; DARWIN-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; DARWIN-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; DARWIN-NEXT: declare i32 @getc_unlocked(ptr)
; DARWIN-NEXT: declare i32 @getchar_unlocked()
; DARWIN-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; DARWIN-NEXT: declare ptr @memmove(ptr, ptr, i64)
; DARWIN-NEXT: declare ptr @memset(ptr, i32, i64)
; DARWIN-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; DARWIN-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; DARWIN-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; DARWIN-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare i32 @putc_unlocked(i32, ptr)
; DARWIN-NEXT: declare i32 @putchar_unlocked(i32)
; DARWIN-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; DARWIN-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; DARWIN-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; DARWIN-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; DARWIN-NEXT: declare ptr @wmemset(ptr, i32, i64)
; DARWIN-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; DARWIN-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; DARWIN-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; DARWIN-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; DARWIN-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; DARWIN-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; DARWIN-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; DARWIN-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; DARWIN-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; WINMSVC: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; WINMSVC-NEXT: @__security_cookie = external global ptr
; WINMSVC-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; WINMSVC-NEXT: declare void @_Unwind_Resume(ptr)
; WINMSVC-NEXT: declare i128 @__aarch64_cas16_acq(i128, i128, ptr)
; WINMSVC-NEXT: declare i128 @__aarch64_cas16_acq_rel(i128, i128, ptr)
; WINMSVC-NEXT: declare i128 @__aarch64_cas16_rel(i128, i128, ptr)
; WINMSVC-NEXT: declare i128 @__aarch64_cas16_relax(i128, i128, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_cas1_acq(i8 zeroext, i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_cas1_acq_rel(i8 zeroext, i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_cas1_rel(i8 zeroext, i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_cas1_relax(i8 zeroext, i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_cas2_acq(i16 zeroext, i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_cas2_acq_rel(i16 zeroext, i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_cas2_rel(i16 zeroext, i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_cas2_relax(i16 zeroext, i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_cas4_acq(i32, i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_cas4_acq_rel(i32, i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_cas4_rel(i32, i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_cas4_relax(i32, i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_cas8_acq(i64, i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_cas8_acq_rel(i64, i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_cas8_rel(i64, i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_cas8_relax(i64, i64, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldadd1_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldadd1_relax(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldadd2_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldadd2_relax(i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldadd4_acq(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldadd4_acq_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldadd4_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldadd4_relax(i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldadd8_acq(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldadd8_acq_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldadd8_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldadd8_relax(i64, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldclr1_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldclr1_relax(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldclr2_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldclr2_relax(i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldclr4_acq(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldclr4_acq_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldclr4_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldclr4_relax(i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldclr8_acq(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldclr8_acq_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldclr8_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldclr8_relax(i64, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldeor1_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldeor1_relax(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldeor2_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldeor2_relax(i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldeor4_acq(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldeor4_acq_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldeor4_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldeor4_relax(i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldeor8_acq(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldeor8_acq_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldeor8_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldeor8_relax(i64, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldset1_acq(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldset1_acq_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldset1_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_ldset1_relax(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldset2_acq(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldset2_acq_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldset2_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_ldset2_relax(i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldset4_acq(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldset4_acq_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldset4_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_ldset4_relax(i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldset8_acq(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldset8_acq_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldset8_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_ldset8_relax(i64, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_swp1_acq(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_swp1_acq_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_swp1_rel(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i8 @__aarch64_swp1_relax(i8 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_swp2_acq(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_swp2_acq_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_swp2_rel(i16 zeroext, ptr)
; WINMSVC-NEXT: declare zeroext i16 @__aarch64_swp2_relax(i16 zeroext, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_swp4_acq(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_swp4_acq_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_swp4_rel(i32, ptr)
; WINMSVC-NEXT: declare i32 @__aarch64_swp4_relax(i32, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_swp8_acq(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_swp8_acq_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_swp8_rel(i64, ptr)
; WINMSVC-NEXT: declare i64 @__aarch64_swp8_relax(i64, ptr)
; WINMSVC-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_get_current_vg()
; WINMSVC-NEXT: declare ptr @__arm_sc_memchr(ptr, i32, i64)
; WINMSVC-NEXT: declare ptr @__arm_sc_memcpy(ptr, ptr, i64)
; WINMSVC-NEXT: declare ptr @__arm_sc_memmove(ptr, ptr, i64)
; WINMSVC-NEXT: declare ptr @__arm_sc_memset(ptr, i32, i64)
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_restore(ptr)
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_save(ptr)
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x2 { i64, i64 } @__arm_sme_state()
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_sme_state_size()
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_restore(ptr)
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
; WINMSVC-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_za_disable()
; WINMSVC-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; WINMSVC-NEXT: declare i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; WINMSVC-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; WINMSVC-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; WINMSVC-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; WINMSVC-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; WINMSVC-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; WINMSVC-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; WINMSVC-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; WINMSVC-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; WINMSVC-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; WINMSVC-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; WINMSVC-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; WINMSVC-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; WINMSVC-NEXT: declare void @__chkstk()
; WINMSVC-NEXT: declare void @__clear_cache(ptr, ptr)
; WINMSVC-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare void @__cxa_end_cleanup()
; WINMSVC-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare void @__llvm_deoptimize(...)
; WINMSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; WINMSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; WINMSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; WINMSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; WINMSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; WINMSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; WINMSVC-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; WINMSVC-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; WINMSVC-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; WINMSVC-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare void @__security_check_cookie(ptr)
; WINMSVC-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; WINMSVC-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; WINMSVC-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; WINMSVC-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; WINMSVC-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; WINMSVC-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; WINMSVC-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; WINMSVC-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare void @cabs(...)
; WINMSVC-NEXT: declare void @cabsf(...)
; WINMSVC-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i32 @fegetenv(ptr)
; WINMSVC-NEXT: declare i32 @fegetmode(ptr)
; WINMSVC-NEXT: declare i32 @fesetenv(ptr)
; WINMSVC-NEXT: declare i32 @fesetmode(ptr)
; WINMSVC-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; WINMSVC-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; WINMSVC-NEXT: declare ptr @memmove(ptr, ptr, i64)
; WINMSVC-NEXT: declare ptr @memset(ptr, i32, i64)
; WINMSVC-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; WINMSVC-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; WINMSVC-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; WINMSVC-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; WINMSVC-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; WINMSVC-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; WINMSVC-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; WINMSVC-NEXT: declare ptr @wmemset(ptr, i16, i64)
; WINMSVC-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; WINMSVC-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; WINMSVC-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; WINMSVC-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; WINMSVC-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; WINMSVC-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; WINMSVC-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; WINMSVC-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; MINGW: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; MINGW-NEXT: @__stack_chk_guard = external global ptr
; MINGW-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; MINGW-NEXT: declare void @_Unwind_Resume(ptr)
; MINGW-NEXT: declare i128 @__aarch64_cas16_acq(i128, i128, ptr)
; MINGW-NEXT: declare i128 @__aarch64_cas16_acq_rel(i128, i128, ptr)
; MINGW-NEXT: declare i128 @__aarch64_cas16_rel(i128, i128, ptr)
; MINGW-NEXT: declare i128 @__aarch64_cas16_relax(i128, i128, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_cas1_acq(i8 zeroext, i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_cas1_acq_rel(i8 zeroext, i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_cas1_rel(i8 zeroext, i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_cas1_relax(i8 zeroext, i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_cas2_acq(i16 zeroext, i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_cas2_acq_rel(i16 zeroext, i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_cas2_rel(i16 zeroext, i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_cas2_relax(i16 zeroext, i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_cas4_acq(i32, i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_cas4_acq_rel(i32, i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_cas4_rel(i32, i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_cas4_relax(i32, i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_cas8_acq(i64, i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_cas8_acq_rel(i64, i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_cas8_rel(i64, i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_cas8_relax(i64, i64, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldadd1_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldadd1_relax(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldadd2_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldadd2_relax(i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldadd4_acq(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldadd4_acq_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldadd4_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldadd4_relax(i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldadd8_acq(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldadd8_acq_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldadd8_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldadd8_relax(i64, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldclr1_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldclr1_relax(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldclr2_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldclr2_relax(i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldclr4_acq(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldclr4_acq_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldclr4_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldclr4_relax(i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldclr8_acq(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldclr8_acq_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldclr8_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldclr8_relax(i64, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldeor1_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldeor1_relax(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldeor2_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldeor2_relax(i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldeor4_acq(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldeor4_acq_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldeor4_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldeor4_relax(i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldeor8_acq(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldeor8_acq_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldeor8_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldeor8_relax(i64, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldset1_acq(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldset1_acq_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldset1_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_ldset1_relax(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldset2_acq(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldset2_acq_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldset2_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_ldset2_relax(i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldset4_acq(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldset4_acq_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldset4_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_ldset4_relax(i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldset8_acq(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldset8_acq_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldset8_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_ldset8_relax(i64, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_swp1_acq(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_swp1_acq_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_swp1_rel(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i8 @__aarch64_swp1_relax(i8 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_swp2_acq(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_swp2_acq_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_swp2_rel(i16 zeroext, ptr)
; MINGW-NEXT: declare zeroext i16 @__aarch64_swp2_relax(i16 zeroext, ptr)
; MINGW-NEXT: declare i32 @__aarch64_swp4_acq(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_swp4_acq_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_swp4_rel(i32, ptr)
; MINGW-NEXT: declare i32 @__aarch64_swp4_relax(i32, ptr)
; MINGW-NEXT: declare i64 @__aarch64_swp8_acq(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_swp8_acq_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_swp8_rel(i64, ptr)
; MINGW-NEXT: declare i64 @__aarch64_swp8_relax(i64, ptr)
; MINGW-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_get_current_vg()
; MINGW-NEXT: declare ptr @__arm_sc_memchr(ptr, i32, i64)
; MINGW-NEXT: declare ptr @__arm_sc_memcpy(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @__arm_sc_memmove(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @__arm_sc_memset(ptr, i32, i64)
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_restore(ptr)
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_save(ptr)
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x2 { i64, i64 } @__arm_sme_state()
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_sme_state_size()
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_restore(ptr)
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
; MINGW-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_za_disable()
; MINGW-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; MINGW-NEXT: declare i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; MINGW-NEXT: declare i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; MINGW-NEXT: declare i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; MINGW-NEXT: declare i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; MINGW-NEXT: declare i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; MINGW-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; MINGW-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; MINGW-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; MINGW-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; MINGW-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; MINGW-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; MINGW-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; MINGW-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; MINGW-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; MINGW-NEXT: declare void @__chkstk()
; MINGW-NEXT: declare void @__clear_cache(ptr, ptr)
; MINGW-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare void @__cxa_end_cleanup()
; MINGW-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare void @__llvm_deoptimize(...)
; MINGW-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; MINGW-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; MINGW-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; MINGW-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; MINGW-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; MINGW-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; MINGW-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; MINGW-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; MINGW-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; MINGW-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; MINGW-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; MINGW-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; MINGW-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @cabs(...)
; MINGW-NEXT: declare void @cabsf(...)
; MINGW-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @fegetenv(ptr)
; MINGW-NEXT: declare i32 @fegetmode(ptr)
; MINGW-NEXT: declare i32 @fesetenv(ptr)
; MINGW-NEXT: declare i32 @fesetmode(ptr)
; MINGW-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @memmove(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @memset(ptr, i32, i64)
; MINGW-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; MINGW-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare ptr @wmemset(ptr, i16, i64)
; MINGW-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; MINGW-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; MINGW-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; MINGW-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; MINGW-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; MINGW-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; MINGW-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; MINGW-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; MINGW-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; ANDROID: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; ANDROID-NEXT: @__stack_chk_guard = external global ptr
; ANDROID-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; ANDROID-NEXT: declare void @_Unwind_Resume(ptr)
; ANDROID-NEXT: declare i128 @__aarch64_cas16_acq(i128, i128, ptr)
; ANDROID-NEXT: declare i128 @__aarch64_cas16_acq_rel(i128, i128, ptr)
; ANDROID-NEXT: declare i128 @__aarch64_cas16_rel(i128, i128, ptr)
; ANDROID-NEXT: declare i128 @__aarch64_cas16_relax(i128, i128, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_cas1_acq(i8 zeroext, i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_cas1_acq_rel(i8 zeroext, i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_cas1_rel(i8 zeroext, i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_cas1_relax(i8 zeroext, i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_cas2_acq(i16 zeroext, i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_cas2_acq_rel(i16 zeroext, i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_cas2_rel(i16 zeroext, i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_cas2_relax(i16 zeroext, i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_cas4_acq(i32, i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_cas4_acq_rel(i32, i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_cas4_rel(i32, i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_cas4_relax(i32, i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_cas8_acq(i64, i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_cas8_acq_rel(i64, i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_cas8_rel(i64, i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_cas8_relax(i64, i64, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldadd1_acq_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldadd1_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldadd1_relax(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldadd2_acq_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldadd2_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldadd2_relax(i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldadd4_acq(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldadd4_acq_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldadd4_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldadd4_relax(i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldadd8_acq(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldadd8_acq_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldadd8_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldadd8_relax(i64, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldclr1_acq_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldclr1_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldclr1_relax(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldclr2_acq_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldclr2_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldclr2_relax(i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldclr4_acq(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldclr4_acq_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldclr4_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldclr4_relax(i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldclr8_acq(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldclr8_acq_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldclr8_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldclr8_relax(i64, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldeor1_acq_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldeor1_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldeor1_relax(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldeor2_acq_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldeor2_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldeor2_relax(i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldeor4_acq(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldeor4_acq_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldeor4_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldeor4_relax(i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldeor8_acq(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldeor8_acq_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldeor8_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldeor8_relax(i64, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldset1_acq(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldset1_acq_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldset1_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_ldset1_relax(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldset2_acq(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldset2_acq_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldset2_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_ldset2_relax(i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldset4_acq(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldset4_acq_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldset4_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_ldset4_relax(i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldset8_acq(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldset8_acq_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldset8_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_ldset8_relax(i64, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_swp1_acq(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_swp1_acq_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_swp1_rel(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i8 @__aarch64_swp1_relax(i8 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_swp2_acq(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_swp2_acq_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_swp2_rel(i16 zeroext, ptr)
; ANDROID-NEXT: declare zeroext i16 @__aarch64_swp2_relax(i16 zeroext, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_swp4_acq(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_swp4_acq_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_swp4_rel(i32, ptr)
; ANDROID-NEXT: declare i32 @__aarch64_swp4_relax(i32, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_swp8_acq(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_swp8_acq_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_swp8_rel(i64, ptr)
; ANDROID-NEXT: declare i64 @__aarch64_swp8_relax(i64, ptr)
; ANDROID-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_get_current_vg()
; ANDROID-NEXT: declare ptr @__arm_sc_memchr(ptr, i32, i64)
; ANDROID-NEXT: declare ptr @__arm_sc_memcpy(ptr, ptr, i64)
; ANDROID-NEXT: declare ptr @__arm_sc_memmove(ptr, ptr, i64)
; ANDROID-NEXT: declare ptr @__arm_sc_memset(ptr, i32, i64)
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_restore(ptr)
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x1 void @__arm_sme_save(ptr)
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x2 { i64, i64 } @__arm_sme_state()
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x1 i64 @__arm_sme_state_size()
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_restore(ptr)
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_tpidr2_save()
; ANDROID-NEXT: declare aarch64_sme_preservemost_from_x0 void @__arm_za_disable()
; ANDROID-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; ANDROID-NEXT: declare i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; ANDROID-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; ANDROID-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; ANDROID-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; ANDROID-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; ANDROID-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; ANDROID-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; ANDROID-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; ANDROID-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; ANDROID-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; ANDROID-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; ANDROID-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; ANDROID-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; ANDROID-NEXT: declare void @__clear_cache(ptr, ptr)
; ANDROID-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare void @__cxa_end_cleanup()
; ANDROID-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @__exp2l_finite(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @__expl_finite(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare void @__llvm_deoptimize(...)
; ANDROID-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; ANDROID-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; ANDROID-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; ANDROID-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; ANDROID-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; ANDROID-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; ANDROID-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @__log10l_finite(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @__log2l_finite(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @__logl_finite(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; ANDROID-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; ANDROID-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; ANDROID-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__powl_finite(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare ptr @__safestack_pointer_address()
; ANDROID-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; ANDROID-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; ANDROID-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; ANDROID-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; ANDROID-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; ANDROID-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; ANDROID-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; ANDROID-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @acosl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @asinl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @atan2l(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @atanl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare void @cabs(...)
; ANDROID-NEXT: declare void @cabsf(...)
; ANDROID-NEXT: declare void @cabsl(...)
; ANDROID-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @cbrtl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @ceill(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @copysignl(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @coshl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @cosl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @exp10l(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @exp2l(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @expl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @fdiml(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i32 @fegetenv(ptr)
; ANDROID-NEXT: declare i32 @fegetmode(ptr)
; ANDROID-NEXT: declare i32 @fesetenv(ptr)
; ANDROID-NEXT: declare i32 @fesetmode(ptr)
; ANDROID-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @floorl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @fmal(fp128, fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fmaximum_numl(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fmaximuml(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fmaxl(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fminimum_numl(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fminimuml(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @fminl(fp128, fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @fmodl(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare fp128 @frexpl(fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @ldexpl(fp128, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llrintl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @llroundl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @log10l(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @log2l(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @logl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lrintl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare i64 @lroundl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; ANDROID-NEXT: declare ptr @memmove(ptr, ptr, i64)
; ANDROID-NEXT: declare ptr @memset(ptr, i32, i64)
; ANDROID-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare fp128 @modfl(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_WRITE_ATTRS]]
; ANDROID-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; ANDROID-NEXT: declare nofpclass(snan inf zero sub norm) fp128 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; ANDROID-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @nearbyintl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @nexttoward(double, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @nexttowardf(float, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @nexttowardl(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @powl(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @remainderl(fp128, fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; ANDROID-NEXT: declare fp128 @remquol(fp128, fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; ANDROID-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @rintl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @roundevenl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @roundl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @scalblnl(fp128, i64) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @scalbnl(fp128, i32) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; ANDROID-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; ANDROID-NEXT: declare void @sincosl(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16), ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; ANDROID-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @sinhl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @sinl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; ANDROID-NEXT: declare nofpclass(ninf nsub nnorm) fp128 @sqrtl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @tanhl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @tanl(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare fp128 @tgammal(fp128) [[SQRT_ATTRS]]
; ANDROID-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare fp128 @truncl(fp128) [[NOMEM_ATTRS]]
; ANDROID-NEXT: declare ptr @wmemset(ptr, i32, i64)
; ANDROID-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; ANDROID-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; ANDROID-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; ANDROID-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; ANDROID-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; ANDROID-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; ANDROID-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; ANDROID-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; ANDROID-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }
