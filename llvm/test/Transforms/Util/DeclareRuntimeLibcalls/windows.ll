; REQUIRES: x86-registered-target
; Strip blank lines and inline attribute comments so the declaration
; stream is contiguous and can be checked with -NEXT: no unexpected call
; can sneak in between two checked declarations.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-pc-windows-msvc < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=MSVC %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-w64-windows-gnu < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=MINGW %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-pc-cygwin < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=CYGWIN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-uefi < %s | sed '/^$/d;/^; Function Attrs:/d' | FileCheck -check-prefix=UEFI %s

; MSVC: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; MSVC-NEXT: @__security_cookie = external global ptr
; MSVC-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; MSVC-NEXT: declare void @_Unwind_Resume(ptr)
; MSVC-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__ashlti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__ashrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, <2 x i64>, i32, i32)
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; MSVC-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; MSVC-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_exchange_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_add_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_and_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_nand_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_or_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_sub_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_fetch_xor_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; MSVC-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; MSVC-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; MSVC-NEXT: declare <2 x i64> @__atomic_load_16(ptr, i32)
; MSVC-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; MSVC-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; MSVC-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; MSVC-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; MSVC-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; MSVC-NEXT: declare void @__atomic_store_16(ptr, <2 x i64>, i32)
; MSVC-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; MSVC-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; MSVC-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; MSVC-NEXT: declare void @__chkstk()
; MSVC-NEXT: declare void @__clear_cache(ptr, ptr)
; MSVC-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__clzti2(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__ctzti2(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare void @__cxa_end_cleanup()
; MSVC-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__divti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixdfti(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixhfti(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixsfti(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixtfti(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixunsdfti(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixunshfti(half) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixunssfti(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floattidf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floattihf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floattisf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floattitf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floattixf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__floatuntidf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__floatuntihf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__floatuntisf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__floatuntitf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__floatuntixf(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare void @__llvm_deoptimize(...)
; MSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; MSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; MSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; MSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; MSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; MSVC-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; MSVC-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__lshrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; MSVC-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; MSVC-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; MSVC-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__modti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__multi3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__popcountti2(<2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare void @__security_check_cookie(ptr)
; MSVC-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_add_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_and_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_max_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_min_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_nand_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_or_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_sub_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_umax_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_umin_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_fetch_and_xor_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_lock_test_and_set_16(ptr, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; MSVC-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; MSVC-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; MSVC-NEXT: declare <2 x i64> @__sync_val_compare_and_swap_16(ptr, <2 x i64>, <2 x i64>)
; MSVC-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; MSVC-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; MSVC-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; MSVC-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__udivti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare <2 x i64> @__umodti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare void @cabs(...)
; MSVC-NEXT: declare void @cabsf(...)
; MSVC-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i32 @fegetenv(ptr)
; MSVC-NEXT: declare i32 @fegetmode(ptr)
; MSVC-NEXT: declare i32 @fesetenv(ptr)
; MSVC-NEXT: declare i32 @fesetmode(ptr)
; MSVC-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MSVC-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; MSVC-NEXT: declare ptr @memmove(ptr, ptr, i64)
; MSVC-NEXT: declare ptr @memset(ptr, i32, i64)
; MSVC-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; MSVC-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MSVC-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; MSVC-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MSVC-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; MSVC-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; MSVC-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; MSVC-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; MSVC-NEXT: declare ptr @wmemset(ptr, i16, i64)
; MSVC-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; MSVC-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; MSVC-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; MSVC-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; MSVC-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; MSVC-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; MSVC-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; MSVC-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; MINGW: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; MINGW-NEXT: @__stack_chk_guard = external global ptr
; MINGW-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; MINGW-NEXT: declare void @_Unwind_Resume(ptr)
; MINGW-NEXT: declare void @___chkstk_ms()
; MINGW-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__ashlti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__ashrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, <2 x i64>, i32, i32)
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; MINGW-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; MINGW-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_exchange_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_add_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_and_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_nand_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_or_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_sub_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_fetch_xor_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; MINGW-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; MINGW-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; MINGW-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; MINGW-NEXT: declare <2 x i64> @__atomic_load_16(ptr, i32)
; MINGW-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; MINGW-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; MINGW-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; MINGW-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; MINGW-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; MINGW-NEXT: declare void @__atomic_store_16(ptr, <2 x i64>, i32)
; MINGW-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; MINGW-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; MINGW-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; MINGW-NEXT: declare void @__clear_cache(ptr, ptr)
; MINGW-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__clzti2(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__ctzti2(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare void @__cxa_end_cleanup()
; MINGW-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__divti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__exp10f128_finite(fp128) [[SQRT_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__exp2f128_finite(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__exp2l_finite(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__expf128_finite(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__expl_finite(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixdfti(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixhfti(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixsfti(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixtfti(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixunsdfti(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixunshfti(half) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixunssfti(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floattidf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floattihf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floattisf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floattitf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floattixf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__floatuntidf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__floatuntihf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__floatuntisf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__floatuntitf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__floatuntixf(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
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
; MINGW-NEXT: declare fp128 @__log10f128_finite(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__log10l_finite(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__log2f128_finite(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__log2l_finite(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__logf128_finite(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__logl_finite(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__lshrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; MINGW-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; MINGW-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; MINGW-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__modti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__multi3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__popcountti2(<2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @__powf128_finite(fp128, fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_add_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_and_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_max_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_min_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_nand_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_or_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_sub_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_umax_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_umin_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_fetch_and_xor_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_lock_test_and_set_16(ptr, <2 x i64>)
; MINGW-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; MINGW-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; MINGW-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; MINGW-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; MINGW-NEXT: declare <2 x i64> @__sync_val_compare_and_swap_16(ptr, <2 x i64>, <2 x i64>)
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
; MINGW-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__udivti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare <2 x i64> @__umodti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @acosf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @acosl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @asinf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @asinl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @atan2f128(fp128, fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @atanf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @atanl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @cabs(...)
; MINGW-NEXT: declare void @cabsf(...)
; MINGW-NEXT: declare void @cabsl(...)
; MINGW-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @cbrtf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @cbrtl(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @ceilf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @ceill(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @copysignf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @cosf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @coshf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @coshl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @cosl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @exp10f128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @exp10l(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @exp2f128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @exp2l(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @expf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @expl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @fegetenv(ptr)
; MINGW-NEXT: declare i32 @fegetmode(ptr)
; MINGW-NEXT: declare i32 @fesetenv(ptr)
; MINGW-NEXT: declare i32 @fesetmode(ptr)
; MINGW-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @floorf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @floorl(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @fmaf128(fp128, fp128, fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fmaxf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fmaximum_numf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fmaximumf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fmaximuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fminf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fminimum_numf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @fminimumf128(fp128, fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fminimuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @fmodf128(fp128, fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare fp128 @frexpf128(fp128, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @frexpl(x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @ldexpf128(fp128, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @ldexpl(x86_fp80, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrintf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llrintl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llroundf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i64 @llroundl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @log10f128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @log10l(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @log2f128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @log2l(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @logf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @logl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrintf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lrintl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lroundf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare i32 @lroundl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @memmove(ptr, ptr, i64)
; MINGW-NEXT: declare ptr @memset(ptr, i32, i64)
; MINGW-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare fp128 @modff128(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @modfl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_WRITE_ATTRS]]
; MINGW-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; MINGW-NEXT: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; MINGW-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @nearbyintf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @nearbyintl(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @nexttoward(double, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @nexttowardf(float, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @powf128(fp128, fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @rintf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @rintl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @roundevenf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @roundevenl(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @roundf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @roundl(x86_fp80) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @scalblnl(x86_fp80, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @scalbnl(x86_fp80, i32) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare void @sincos(double, ptr noalias nonnull writeonly align 8 dereferenceable(8), ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare void @sincosf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4), ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare void @sincosf128(fp128, ptr noalias nonnull writeonly align 16 dereferenceable(16), ptr noalias nonnull writeonly align 16 dereferenceable(16)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare void @sincosl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10), ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; MINGW-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @sinf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @sinhf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @sinhl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @sinl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) fp128 @sqrtf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare nofpclass(ninf nsub nnorm) x86_fp80 @sqrtl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @tanf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare fp128 @tanhf128(fp128) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @tanhl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @tanl(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; MINGW-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @tgammal(x86_fp80) [[SQRT_ATTRS]]
; MINGW-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare fp128 @truncf128(fp128) [[NOMEM_ATTRS]]
; MINGW-NEXT: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]
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

; CYGWIN: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; CYGWIN-NEXT: @__stack_chk_guard = external global ptr
; CYGWIN-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; CYGWIN-NEXT: declare void @_Unwind_Resume(ptr)
; CYGWIN-NEXT: declare void @___chkstk_ms()
; CYGWIN-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__ashlti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__ashrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, <2 x i64>, i32, i32)
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; CYGWIN-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; CYGWIN-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_exchange_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_add_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_and_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_nand_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_or_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_sub_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_fetch_xor_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; CYGWIN-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; CYGWIN-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; CYGWIN-NEXT: declare <2 x i64> @__atomic_load_16(ptr, i32)
; CYGWIN-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; CYGWIN-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; CYGWIN-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; CYGWIN-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; CYGWIN-NEXT: declare void @__atomic_store_16(ptr, <2 x i64>, i32)
; CYGWIN-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; CYGWIN-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; CYGWIN-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; CYGWIN-NEXT: declare void @__clear_cache(ptr, ptr)
; CYGWIN-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__clzti2(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__ctzti2(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare void @__cxa_end_cleanup()
; CYGWIN-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__divti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__exp2l_finite(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__expl_finite(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixdfti(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixhfti(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixsfti(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixtfti(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixunsdfti(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixunshfti(half) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixunssfti(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floattidf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floattihf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floattisf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floattitf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floattixf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__floatuntidf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__floatuntihf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__floatuntisf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__floatuntitf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__floatuntixf(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare void @__llvm_deoptimize(...)
; CYGWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; CYGWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; CYGWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; CYGWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; CYGWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; CYGWIN-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; CYGWIN-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__log10l_finite(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__log2l_finite(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__logl_finite(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__lshrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; CYGWIN-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; CYGWIN-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; CYGWIN-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__modti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__multi3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__popcountti2(<2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_add_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_and_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_max_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_min_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_nand_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_or_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_sub_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_umax_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_umin_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_fetch_and_xor_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_lock_test_and_set_16(ptr, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; CYGWIN-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; CYGWIN-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; CYGWIN-NEXT: declare <2 x i64> @__sync_val_compare_and_swap_16(ptr, <2 x i64>, <2 x i64>)
; CYGWIN-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; CYGWIN-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; CYGWIN-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; CYGWIN-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__udivti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare <2 x i64> @__umodti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @acosl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @asinl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @atanl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare void @cabs(...)
; CYGWIN-NEXT: declare void @cabsf(...)
; CYGWIN-NEXT: declare void @cabsl(...)
; CYGWIN-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @cbrtl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @ceill(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @coshl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @cosl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @exp10l(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @exp2l(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @expl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fdiml(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i32 @fegetenv(ptr)
; CYGWIN-NEXT: declare i32 @fegetmode(ptr)
; CYGWIN-NEXT: declare i32 @fesetenv(ptr)
; CYGWIN-NEXT: declare i32 @fesetmode(ptr)
; CYGWIN-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @floorl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fmal(x86_fp80, x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fmaximuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fminimuml(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @frexpl(x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @ldexpl(x86_fp80, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llrintl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @llroundl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @log10l(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @log2l(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @logl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lrint(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lrintf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lrintl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lround(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lroundf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare i64 @lroundl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; CYGWIN-NEXT: declare ptr @memmove(ptr, ptr, i64)
; CYGWIN-NEXT: declare ptr @memset(ptr, i32, i64)
; CYGWIN-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @modfl(x86_fp80, ptr noalias nonnull writeonly align 16 dereferenceable(10)) [[ARGMEM_WRITE_ATTRS]]
; CYGWIN-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; CYGWIN-NEXT: declare nofpclass(snan inf zero sub norm) x86_fp80 @nanl(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; CYGWIN-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @nearbyintl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @nexttoward(double, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @nexttowardf(float, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @remainderl(x86_fp80, x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; CYGWIN-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @rintl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @roundevenl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @roundl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare double @scalbln(double, i64) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @scalblnf(float, i64) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @scalblnl(x86_fp80, i64) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @scalbnl(x86_fp80, i32) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @sinhl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @sinl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; CYGWIN-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare nofpclass(ninf nsub nnorm) x86_fp80 @sqrtl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @tanhl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @tanl(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @tgammal(x86_fp80) [[SQRT_ATTRS]]
; CYGWIN-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare x86_fp80 @truncl(x86_fp80) [[NOMEM_ATTRS]]
; CYGWIN-NEXT: declare ptr @wmemset(ptr, i16, i64)
; CYGWIN-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CYGWIN-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; CYGWIN-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; CYGWIN-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; CYGWIN-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CYGWIN-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CYGWIN-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; CYGWIN-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; CYGWIN-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; UEFI: @__safestack_unsafe_stack_ptr = external thread_local(initialexec) global ptr
; UEFI-NEXT: @__stack_chk_guard = external global ptr
; UEFI-NEXT: declare i32 @_Unwind_RaiseException(ptr)
; UEFI-NEXT: declare void @_Unwind_Resume(ptr)
; UEFI-NEXT: declare double @__adddf3(double, double) [[NOMEM_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare float @__addsf3(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__addtf3(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__addxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__ashldi3(i64, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i16 @__ashlhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__ashlsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__ashlti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__ashrdi3(i64, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i16 @__ashrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__ashrsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__ashrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32, i32)
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange_1(ptr, ptr, i8 zeroext, i32, i32)
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange_16(ptr, ptr, <2 x i64>, i32, i32)
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange_2(ptr, ptr, i16 zeroext, i32, i32)
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange_4(ptr, ptr, i32, i32, i32)
; UEFI-NEXT: declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i32, i32)
; UEFI-NEXT: declare void @__atomic_exchange(i64, ptr, ptr, ptr, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_exchange_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_exchange_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_exchange_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_exchange_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_exchange_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_add_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_add_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_add_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_add_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_add_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_and_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_and_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_and_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_and_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_and_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_nand_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_nand_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_nand_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_nand_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_nand_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_or_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_or_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_or_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_or_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_or_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_sub_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_sub_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_sub_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_sub_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_sub_8(ptr, i64, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_fetch_xor_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_fetch_xor_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_fetch_xor_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare i32 @__atomic_fetch_xor_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__atomic_fetch_xor_8(ptr, i64, i32)
; UEFI-NEXT: declare void @__atomic_load(i64, ptr, ptr, i32)
; UEFI-NEXT: declare zeroext i8 @__atomic_load_1(ptr, i32)
; UEFI-NEXT: declare <2 x i64> @__atomic_load_16(ptr, i32)
; UEFI-NEXT: declare zeroext i16 @__atomic_load_2(ptr, i32)
; UEFI-NEXT: declare i32 @__atomic_load_4(ptr, i32)
; UEFI-NEXT: declare i64 @__atomic_load_8(ptr, i32)
; UEFI-NEXT: declare void @__atomic_store(i64, ptr, ptr, i32)
; UEFI-NEXT: declare void @__atomic_store_1(ptr, i8 zeroext, i32)
; UEFI-NEXT: declare void @__atomic_store_16(ptr, <2 x i64>, i32)
; UEFI-NEXT: declare void @__atomic_store_2(ptr, i16 zeroext, i32)
; UEFI-NEXT: declare void @__atomic_store_4(ptr, i32, i32)
; UEFI-NEXT: declare void @__atomic_store_8(ptr, i64, i32)
; UEFI-NEXT: declare void @__chkstk()
; UEFI-NEXT: declare void @__clear_cache(ptr, ptr)
; UEFI-NEXT: declare i32 @__clzdi2(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__clzsi2(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__clzti2(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__ctzdi2(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__ctzsi2(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__ctzti2(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare void @__cxa_end_cleanup()
; UEFI-NEXT: declare double @__divdf3(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i16 @__divhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i8 @__divqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__divsf3(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__divsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__divtf3(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__divti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__divxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__eqdf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__eqsf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__eqtf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__exp2_finite(double) [[SQRT_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare float @__exp2f_finite(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @__exp_finite(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__expf_finite(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__extendbfsf2(bfloat) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__extenddftf2(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__extendhfdf2(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__extendhfsf2(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__extendhftf2(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__extendhfxf2(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__extendsfdf2(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__extendsftf2(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__extendxftf2(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixdfdi(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixdfsi(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixdfti(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixhfdi(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixhfsi(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixhfti(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixsfdi(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixsfsi(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixsfti(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixtfdi(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixtfsi(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixtfti(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixunsdfdi(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixunsdfsi(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixunsdfti(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixunshfdi(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixunshfsi(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixunshfti(half) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixunssfdi(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixunssfsi(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixunssfti(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixunstfdi(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixunstfsi(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixunstfti(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixunsxfdi(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixunsxfsi(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixunsxfti(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__fixxfdi(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__fixxfsi(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__fixxfti(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare bfloat @__floatdibf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floatdidf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floatdihf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floatdisf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floatditf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floatdixf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floatsidf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floatsihf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floatsisf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floatsitf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floatsixf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floattidf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floattihf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floattisf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floattitf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floattixf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare bfloat @__floatundibf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floatundidf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floatundihf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floatundisf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floatunditf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floatundixf(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floatunsidf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floatunsihf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floatunsisf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floatunsitf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floatunsixf(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__floatuntidf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__floatuntihf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__floatuntisf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__floatuntitf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__floatuntixf(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__gedf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__gesf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__getf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__gtdf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__gtsf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__gttf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__ledf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__lesf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__letf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare void @__llvm_deoptimize(...)
; UEFI-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_1(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_16(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_2(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_4(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memcpy_element_unordered_atomic_8(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memmove_element_unordered_atomic_1(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memmove_element_unordered_atomic_16(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memmove_element_unordered_atomic_2(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memmove_element_unordered_atomic_4(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memmove_element_unordered_atomic_8(ptr, ptr, i64)
; UEFI-NEXT: declare void @__llvm_memset_element_unordered_atomic_1(ptr, i32, i64)
; UEFI-NEXT: declare void @__llvm_memset_element_unordered_atomic_16(ptr, i32, i64)
; UEFI-NEXT: declare void @__llvm_memset_element_unordered_atomic_2(ptr, i32, i64)
; UEFI-NEXT: declare void @__llvm_memset_element_unordered_atomic_4(ptr, i32, i64)
; UEFI-NEXT: declare void @__llvm_memset_element_unordered_atomic_8(ptr, i32, i64)
; UEFI-NEXT: declare double @__log10_finite(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__log10f_finite(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @__log2_finite(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__log2f_finite(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @__log_finite(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__logf_finite(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i64 @__lshrdi3(i64, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i16 @__lshrhi3(i16 signext, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__lshrsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__lshrti3(<2 x i64>, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__ltdf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__ltsf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__lttf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; UEFI-NEXT: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; UEFI-NEXT: declare ptr @__memset_chk(ptr, i32, i64, i64)
; UEFI-NEXT: declare signext i16 @__modhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i8 @__modqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__modsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__modti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__muldf3(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare signext i16 @__mulhi3(i16 signext, i16 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__mulosi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare signext i8 @__mulqi3(i8 signext, i8 signext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__mulsf3(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__mulsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__multf3(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__multi3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__mulxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__nedf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__negdi2(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__negsi2(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__nesf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__netf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__popcountdi2(i64) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__popcountsi2(i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__popcountti2(<2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__pow_finite(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @__powf_finite(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @__powidf2(double, i32) [[NOMEM_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare float @__powisf2(float, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__powitf2(fp128, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__powixf2(x86_fp80, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare void @__stack_chk_fail() [[NORETURN_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare double @__subdf3(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__subsf3(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare fp128 @__subtf3(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__subxf3(x86_fp80, x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_add_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_add_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_add_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_add_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_add_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_and_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_and_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_and_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_and_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_and_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_max_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_max_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_max_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_max_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_max_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_min_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_min_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_min_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_min_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_min_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_nand_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_nand_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_nand_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_nand_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_nand_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_or_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_or_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_or_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_or_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_or_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_sub_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_sub_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_sub_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_sub_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_sub_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_umax_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_umax_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_umax_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_umax_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_umax_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_umin_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_umin_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_umin_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_umin_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_umin_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_fetch_and_xor_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_fetch_and_xor_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_fetch_and_xor_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_fetch_and_xor_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_fetch_and_xor_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_lock_test_and_set_1(ptr, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_lock_test_and_set_16(ptr, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_lock_test_and_set_2(ptr, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_lock_test_and_set_4(ptr, i32)
; UEFI-NEXT: declare i64 @__sync_lock_test_and_set_8(ptr, i64)
; UEFI-NEXT: declare zeroext i8 @__sync_val_compare_and_swap_1(ptr, i8 zeroext, i8 zeroext)
; UEFI-NEXT: declare <2 x i64> @__sync_val_compare_and_swap_16(ptr, <2 x i64>, <2 x i64>)
; UEFI-NEXT: declare zeroext i16 @__sync_val_compare_and_swap_2(ptr, i16 zeroext, i16 zeroext)
; UEFI-NEXT: declare i32 @__sync_val_compare_and_swap_4(ptr, i32, i32)
; UEFI-NEXT: declare i64 @__sync_val_compare_and_swap_8(ptr, i64, i64)
; UEFI-NEXT: declare bfloat @__truncdfbf2(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__truncdfhf2(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__truncdfsf2(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare bfloat @__truncsfbf2(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__truncsfhf2(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare bfloat @__trunctfbf2(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__trunctfdf2(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__trunctfhf2(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__trunctfsf2(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare x86_fp80 @__trunctfxf2(fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare bfloat @__truncxfbf2(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @__truncxfdf2(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare half @__truncxfhf2(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @__truncxfsf2(x86_fp80) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i16 @__udivhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i8 @__udivqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__udivsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__udivti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i16 @__umodhi3(i16 zeroext, i16 zeroext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare zeroext i8 @__umodqi3(i8 zeroext, i8 zeroext) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i32 @__umodsi3(i32, i32) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare <2 x i64> @__umodti3(<2 x i64>, <2 x i64>) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__unorddf2(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__unordsf2(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare i64 @__unordtf2(fp128, fp128) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @acos(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @acosf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @asin(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @asinf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @atan(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @atan2(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @atan2f(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @atanf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare void @cabs(...)
; UEFI-NEXT: declare void @cabsf(...)
; UEFI-NEXT: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare double @cbrt(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @cbrtf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @ceil(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @ceilf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @copysign(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @copysignf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @cos(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @cosf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @cosh(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @coshf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @exp(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @exp10(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @exp10f(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @exp2(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @exp2f(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @expf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @fdim(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @fdimf(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i32 @fegetenv(ptr)
; UEFI-NEXT: declare i32 @fegetmode(ptr)
; UEFI-NEXT: declare i32 @fesetenv(ptr)
; UEFI-NEXT: declare i32 @fesetmode(ptr)
; UEFI-NEXT: declare double @floor(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @floorf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fma(double, double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @fmaf(float, float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @fmax(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fmaxf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fmaximum(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fmaximum_num(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fmaximum_numf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fmaximumf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fmin(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fminf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fminimum(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fminimum_num(double, double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fminimum_numf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @fminimumf(float, float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @fmod(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @fmodf(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare double @frexp(double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; UEFI-NEXT: declare float @frexpf(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; UEFI-NEXT: declare double @ldexp(double, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @ldexpf(float, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i64 @llrint(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i64 @llrintf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i64 @llround(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i64 @llroundf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @log(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @log10(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @log10f(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @log2(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @log2f(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @logf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i32 @lrint(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i32 @lrintf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i32 @lround(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare i32 @lroundf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare ptr @memcpy(ptr, ptr, i64)
; UEFI-NEXT: declare ptr @memmove(ptr, ptr, i64)
; UEFI-NEXT: declare ptr @memset(ptr, i32, i64)
; UEFI-NEXT: declare double @modf(double, ptr noalias nonnull writeonly align 8 dereferenceable(8)) [[ARGMEM_WRITE_ATTRS]]
; UEFI-NEXT: declare float @modff(float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_WRITE_ATTRS]]
; UEFI-NEXT: declare nofpclass(snan inf zero sub norm) double @nan(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare nofpclass(snan inf zero sub norm) float @nanf(ptr readonly captures(none)) [[ARGMEM_READ_ATTRS]]
; UEFI-NEXT: declare double @nearbyint(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @nearbyintf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @nexttoward(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @nexttowardf(float, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @pow(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @powf(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @remainder(double, double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @remainderf(float, float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @remquo(double, double, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare float @remquof(float, float, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[ARGMEM_ERRNO_WRITE_ATTRS]]
; UEFI-NEXT: declare double @rint(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @rintf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @round(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @roundeven(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @roundevenf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @roundf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare double @scalbln(double, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @scalblnf(float, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @scalbn(double, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @scalbnf(float, i32) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @sin(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @sinf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @sinh(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @sinhf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]
; UEFI-NEXT: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @tan(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @tanf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @tanh(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @tanhf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @tgamma(double) [[SQRT_ATTRS]]
; UEFI-NEXT: declare float @tgammaf(float) [[SQRT_ATTRS]]
; UEFI-NEXT: declare double @trunc(double) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare float @truncf(float) [[NOMEM_ATTRS]]
; UEFI-NEXT: declare ptr @wmemset(ptr, i16, i64)
; UEFI-NEXT: attributes [[NOMEM_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; UEFI-NEXT: attributes [[SQRT_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; UEFI-NEXT: attributes [[ARGMEM_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; UEFI-NEXT: attributes [[NORETURN_ATTRS]] = { nofree noreturn nounwind }
; UEFI-NEXT: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; UEFI-NEXT: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; UEFI-NEXT: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: readwrite) "alloc-family"="malloc" }
; UEFI-NEXT: attributes [[ARGMEM_READ_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; UEFI-NEXT: attributes [[ARGMEM_ERRNO_WRITE_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }
