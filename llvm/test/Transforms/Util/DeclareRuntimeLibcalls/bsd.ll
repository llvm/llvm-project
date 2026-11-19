; REQUIRES: x86-registered-target
; Strip blank lines and inline attribute comments so the declaration
; stream is contiguous and can be checked with -NEXT: no unexpected call
; can sneak in between two checked declarations.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=LINUX %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-freebsd < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=FREEBSD %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-openbsd < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=OPENBSD %s

; LINUX: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; LINUX-NEXT: @__stack_chk_guard = external global ptr
; LINUX-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; LINUX-NEXT: declare void @_Unwind_Resume(ptr)
; LINUX-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; LINUX-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
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
; LINUX-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__exp10f128_finite(fp128) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__exp2f128_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__exp2l_finite(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__expf128_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__expl_finite(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
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
; LINUX-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floattixf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__floatuntixf(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
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
; LINUX-NEXT: declare fp128 @__log10f128_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__log10l_finite(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__log2f128_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__log2l_finite(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__logf128_finite(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__logl_finite(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; LINUX-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; LINUX-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; LINUX-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @__powf128_finite(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
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
; LINUX-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @acosf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @acosl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @asinf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @asinl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @atan2f128(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @atanf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @atanl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @cabs(...)
; LINUX-NEXT: declare void @cabsf(...)
; LINUX-NEXT: declare void @cabsl(...)
; LINUX-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @cbrtf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @cbrtl(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @ceilf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @ceill(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @copysignf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @cosf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @coshf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @coshl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @cosl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @exp10f128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @exp10l(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @exp2f128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @exp2l(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @expf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @expl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i32 @fegetenv(ptr)
; LINUX-NEXT: declare i32 @fegetmode(ptr)
; LINUX-NEXT: declare i32 @fesetenv(ptr)
; LINUX-NEXT: declare i32 @fesetmode(ptr)
; LINUX-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @floorf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @floorl(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaf128(fp128, fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaxf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaximum_numf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fmaximumf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fmaximuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminimum_numf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @fminimumf128(fp128, fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fminimuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @fmodf128(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare fp128 @frexpf128(fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @frexpl(x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @ldexpf128(fp128, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @ldexpl(x86_fp80, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llrintl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @llroundl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @log10f128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @log10l(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @log2f128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @log2l(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @logf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @logl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrintf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lrintl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lroundf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare i64 @lroundl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @memmove(ptr, ptr, i64)
; LINUX-NEXT: declare ptr @memset(ptr, i32, i64)
; LINUX-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare fp128 @modff128(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @modfl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_WRITE_ATTRS]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; LINUX-NEXT: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; LINUX-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @nearbyintf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @nearbyintl(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @nexttoward(double, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @nexttowardf(float, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @powf128(fp128, fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @rintf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @rintl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @roundevenf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @roundevenl(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @roundf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @roundl(x86_fp80) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @scalblnl(x86_fp80, i64) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @scalbnl(x86_fp80, i32) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosf128(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16), ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare void @sincosl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10), ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; LINUX-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @sinf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @sinhf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @sinhl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @sinl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) fp128 @sqrtf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare nofpclass(ninf nsub nnorm) x86_fp80 @sqrtl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @tanf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare fp128 @tanhf128(fp128) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @tanhl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @tanl(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; LINUX-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @tgammal(x86_fp80) [[SQRT_ATTRS]]
; LINUX-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare fp128 @truncf128(fp128) [[NOMEM_ATTRS]]
; LINUX-NEXT: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]
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

; FREEBSD: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; FREEBSD-NEXT: @__stack_chk_guard = external global ptr
; FREEBSD-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; FREEBSD-NEXT: declare void @_Unwind_Resume(ptr)
; FREEBSD-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; FREEBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; FREEBSD-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; FREEBSD-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; FREEBSD-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; FREEBSD-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; FREEBSD-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; FREEBSD-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; FREEBSD-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; FREEBSD-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; FREEBSD-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; FREEBSD-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; FREEBSD-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; FREEBSD-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; FREEBSD-NEXT: declare void @__clear_cache(ptr, ptr)
; FREEBSD-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare void @__cxa_end_cleanup()
; FREEBSD-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__exp2l_finite(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__expl_finite(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floattixf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__floatuntixf(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare void @__llvm_deoptimize(...)
; FREEBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; FREEBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; FREEBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; FREEBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; FREEBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; FREEBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; FREEBSD-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__log10l_finite(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__log2l_finite(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__logl_finite(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; FREEBSD-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; FREEBSD-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; FREEBSD-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; FREEBSD-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; FREEBSD-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; FREEBSD-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; FREEBSD-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; FREEBSD-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; FREEBSD-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; FREEBSD-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @acosl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @asinl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @atanl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare void @cabs(...)
; FREEBSD-NEXT: declare void @cabsf(...)
; FREEBSD-NEXT: declare void @cabsl(...)
; FREEBSD-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @cbrtl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @ceill(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @coshl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @cosl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @exp10l(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @exp2l(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @expl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i32 @fegetenv(ptr)
; FREEBSD-NEXT: declare i32 @fegetmode(ptr)
; FREEBSD-NEXT: declare i32 @fesetenv(ptr)
; FREEBSD-NEXT: declare i32 @fesetmode(ptr)
; FREEBSD-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @floorl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fmaximuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fminimuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @frexpl(x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @ldexpl(x86_fp80, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llrintl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @llroundl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @log10l(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @log2l(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @logl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lrintl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare i64 @lroundl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; FREEBSD-NEXT: declare ptr @memmove(ptr, ptr, i64)
; FREEBSD-NEXT: declare ptr @memset(ptr, i32, i64)
; FREEBSD-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @modfl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_WRITE_ATTRS]]
; FREEBSD-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; FREEBSD-NEXT: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; FREEBSD-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @nearbyintl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @nexttoward(double, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @nexttowardf(float, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; FREEBSD-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @rintl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @roundevenl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @roundl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @scalblnl(x86_fp80, i64) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @scalbnl(x86_fp80, i32) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @sinhl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @sinl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; FREEBSD-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare nofpclass(ninf nsub nnorm) x86_fp80 @sqrtl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @tanhl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @tanl(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @tgammal(x86_fp80) [[SQRT_ATTRS]]
; FREEBSD-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]
; FREEBSD-NEXT: declare ptr @wmemset(ptr, i32, i64)
; FREEBSD-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; FREEBSD-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; FREEBSD-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; FREEBSD-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; FREEBSD-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; FREEBSD-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; FREEBSD-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; FREEBSD-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; FREEBSD-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; OPENBSD: @__guard_local = external global ptr
; OPENBSD-NEXT: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; OPENBSD-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; OPENBSD-NEXT: declare void @_Unwind_Resume(ptr)
; OPENBSD-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__ashlti3(i128, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__ashrti3(i128, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, i128, i32, i32)
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; OPENBSD-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; OPENBSD-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_exchange_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_add_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_and_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_nand_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_or_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_sub_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare i128 @__atomic_fetch_xor_16(ptr, i128, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; OPENBSD-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; OPENBSD-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; OPENBSD-NEXT: declare i128 @__atomic_load_16(ptr, i32)
; OPENBSD-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; OPENBSD-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; OPENBSD-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; OPENBSD-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; OPENBSD-NEXT: declare void @__atomic_store_16(ptr, i128, i32)
; OPENBSD-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; OPENBSD-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; OPENBSD-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; OPENBSD-NEXT: declare void @__clear_cache(ptr, ptr)
; OPENBSD-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__clzti2(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__ctzti2(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare void @__cxa_end_cleanup()
; OPENBSD-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__divti3(i128, i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__exp2l_finite(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__expl_finite(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixdfti(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixhfti(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixsfti(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixtfti(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixunsdfti(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixunshfti(half) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixunssfti(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floattidf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floattihf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floattisf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floattitf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floattixf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__floatuntidf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__floatuntihf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__floatuntisf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__floatuntitf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__floatuntixf(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare void @__llvm_deoptimize(...)
; OPENBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; OPENBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; OPENBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; OPENBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; OPENBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; OPENBSD-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; OPENBSD-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__log10l_finite(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__log2l_finite(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__logl_finite(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__lshrti3(i128, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; OPENBSD-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; OPENBSD-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; OPENBSD-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__modti3(i128, i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__multi3(i128, i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__popcountti2(i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare void @__stack_smash_handler(ptr) [[NORETURN_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_add_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_and_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_max_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_min_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_nand_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_or_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_sub_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_umax_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_umin_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_fetch_and_xor_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_lock_test_and_set_16(ptr, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; OPENBSD-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; OPENBSD-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; OPENBSD-NEXT: declare i128 @__sync_val_compare_and_swap_16(ptr, i128, i128)
; OPENBSD-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; OPENBSD-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; OPENBSD-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; OPENBSD-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__udivti3(i128, i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i128 @__umodti3(i128, i128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @acosl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @asinl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @atanl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare void @cabs(...)
; OPENBSD-NEXT: declare void @cabsf(...)
; OPENBSD-NEXT: declare void @cabsl(...)
; OPENBSD-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @cbrtl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @ceill(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @coshl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @cosl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @exp10l(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @exp2l(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @expl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i32 @fegetenv(ptr)
; OPENBSD-NEXT: declare i32 @fegetmode(ptr)
; OPENBSD-NEXT: declare i32 @fesetenv(ptr)
; OPENBSD-NEXT: declare i32 @fesetmode(ptr)
; OPENBSD-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @floorl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fmaximuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fminimuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @frexpl(x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @ldexpl(x86_fp80, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llrintl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @llroundl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @log10l(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @log2l(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @logl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lrintl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare i64 @lroundl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; OPENBSD-NEXT: declare ptr @memmove(ptr, ptr, i64)
; OPENBSD-NEXT: declare ptr @memset(ptr, i32, i64)
; OPENBSD-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @modfl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_WRITE_ATTRS]]
; OPENBSD-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; OPENBSD-NEXT: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; OPENBSD-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @nearbyintl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @nexttoward(double, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @nexttowardf(float, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; OPENBSD-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @rintl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @roundevenl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @roundl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @scalblnl(x86_fp80, i64) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @scalbnl(x86_fp80, i32) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @sinhl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @sinl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; OPENBSD-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare nofpclass(ninf nsub nnorm) x86_fp80 @sqrtl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @tanhl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @tanl(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @tgammal(x86_fp80) [[SQRT_ATTRS]]
; OPENBSD-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]
; OPENBSD-NEXT: declare ptr @wmemset(ptr, i32, i64)
; OPENBSD-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; OPENBSD-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; OPENBSD-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; OPENBSD-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; OPENBSD-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; OPENBSD-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; OPENBSD-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; OPENBSD-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; OPENBSD-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }
