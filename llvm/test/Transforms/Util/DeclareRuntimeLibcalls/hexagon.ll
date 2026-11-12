; REQUIRES: hexagon-registered-target
; Strip blank lines and inline attribute comments so the declaration
; stream is contiguous and can be checked with -NEXT: no unexpected call
; can sneak in between two checked declarations.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=hexagon-unknown-elf < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=CHECK %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=hexagon-unknown-linux-gnu < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=LINUX %s

; CHECK: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; CHECK-NEXT: @__stack_chk_guard = external global ptr
; CHECK-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; CHECK-NEXT: declare void @_Unwind_Resume(ptr)
; CHECK-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange(i32, ptr, ptr, ptr, i32, i32)
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; CHECK-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; CHECK-NEXT: declare void @__atomic_exchange(i32, ptr, ptr, ptr, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; CHECK-NEXT: declare void @__atomic_load(i32, ptr, ptr, i32)
; CHECK-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; CHECK-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; CHECK-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; CHECK-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; CHECK-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; CHECK-NEXT: declare void @__atomic_store(i32, ptr, ptr, i32)
; CHECK-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; CHECK-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; CHECK-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; CHECK-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; CHECK-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; CHECK-NEXT: declare void @__clear_cache(ptr, ptr)
; CHECK-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare void @__cxa_end_cleanup()
; CHECK-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__divdi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__gedf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__gesf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_adddf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_divdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__hexagon_divdi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__hexagon_divsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__hexagon_divsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_fast2_sqrtdf2(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__hexagon_fast2_sqrtf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_fast_adddf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_fast_divdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__hexagon_fast_divsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_fast_muldf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_fast_subdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare void @__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(ptr, ptr, i32)
; CHECK-NEXT: declare i64 @__hexagon_moddi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__hexagon_modsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__hexagon_muldf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare nofpclass(ninf nsub nnorm) float @__hexagon_sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare double @__hexagon_subdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__hexagon_udivdi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__hexagon_udivsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__hexagon_umoddi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__hexagon_umodsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare void @__llvm_deoptimize(...)
; CHECK-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i32)
; CHECK-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i32)
; CHECK-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i32)
; CHECK-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i32)
; CHECK-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i32)
; CHECK-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i32)
; CHECK-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i32, i32)
; CHECK-NEXT: declare ptr @__memmove_chk(ptr, ptr, i32, i32)
; CHECK-NEXT: declare ptr @__memset_chk(ptr, i32, i32, i32)
; CHECK-NEXT: declare i64 @__moddi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__muldi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__nedf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__nesf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; CHECK-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; CHECK-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; CHECK-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; CHECK-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; CHECK-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; CHECK-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; CHECK-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__udivdi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i64 @__umoddi3(i64, i64) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare i32 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare void @cabs(...)
; CHECK-NEXT: declare void @cabsf(...)
; CHECK-NEXT: declare noalias noundef ptr @calloc(i32 noundef, i32 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i32 @fegetenv(ptr)
; CHECK-NEXT: declare i32 @fegetmode(ptr)
; CHECK-NEXT: declare i32 @fesetenv(ptr)
; CHECK-NEXT: declare i32 @fesetmode(ptr)
; CHECK-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CHECK-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CHECK-NEXT: declare void @hexagon_memcpy_forward_vp4cp4n2(ptr, ptr, i32)
; CHECK-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare noalias noundef ptr @malloc(i32 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare ptr @memcpy(ptr, ptr, i32)
; CHECK-NEXT: declare ptr @memmove(ptr, ptr, i32)
; CHECK-NEXT: declare ptr @memset(ptr, i32, i32)
; CHECK-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; CHECK-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CHECK-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; CHECK-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; CHECK-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; CHECK-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; CHECK-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; CHECK-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; CHECK-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; CHECK-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; CHECK-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; CHECK-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; CHECK-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CHECK-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CHECK-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; CHECK-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; LINUX: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; LINUX-NEXT: @__stack_chk_guard = external global ptr
; LINUX-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; LINUX-NEXT: declare void @_Unwind_Resume(ptr)
; LINUX-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange(i32, ptr, ptr, ptr, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; LINUX-NEXT: declare void @__atomic_exchange(i32, ptr, ptr, ptr, i32)
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
; LINUX-NEXT: declare void @__atomic_load(i32, ptr, ptr, i32)
; LINUX-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; LINUX-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; LINUX-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; LINUX-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; LINUX-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; LINUX-NEXT: declare void @__atomic_store(i32, ptr, ptr, i32)
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
; LINUX-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
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
; LINUX-NEXT: declare double @__hexagon_adddf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_divdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__hexagon_divdi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__hexagon_divsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__hexagon_divsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_fast2_sqrtdf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__hexagon_fast2_sqrtf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_fast_adddf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_fast_divdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__hexagon_fast_divsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_fast_muldf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_fast_subdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare void @__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(ptr, ptr, i32)
; LINUX-NEXT: declare i64 @__hexagon_moddi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__hexagon_modsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__hexagon_muldf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) float @__hexagon_sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @__hexagon_subdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__hexagon_udivdi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__hexagon_udivsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__hexagon_umoddi3(i64, i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__hexagon_umodsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ledf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare void @__llvm_deoptimize(...)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i32)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i32)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i32)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i32)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i32)
; LINUX-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i32)
; LINUX-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i32, i32)
; LINUX-NEXT: declare ptr @__memmove_chk(ptr, ptr, i32, i32)
; LINUX-NEXT: declare ptr @__memset_chk(ptr, i32, i32, i32)
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
; LINUX-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @cabs(...)
; LINUX-NEXT: declare void @cabsf(...)
; LINUX-NEXT: declare noalias noundef ptr @calloc(i32 noundef, i32 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @fegetenv(ptr)
; LINUX-NEXT: declare i32 @fegetmode(ptr)
; LINUX-NEXT: declare i32 @fesetenv(ptr)
; LINUX-NEXT: declare i32 @fesetmode(ptr)
; LINUX-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare void @hexagon_memcpy_forward_vp4cp4n2(ptr, ptr, i32)
; LINUX-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare noalias noundef ptr @malloc(i32 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare ptr @memcpy(ptr, ptr, i32)
; LINUX-NEXT: declare ptr @memmove(ptr, ptr, i32)
; LINUX-NEXT: declare ptr @memset(ptr, i32, i32)
; LINUX-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; LINUX-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; LINUX-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; LINUX-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; LINUX-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; LINUX-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; LINUX-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; LINUX-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }
