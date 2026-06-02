//===-- TargetLibraryInfo.td - File that describes library functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/IR/RuntimeLibcalls.td"

/// void *operator new(unsigned int);
def TLI_msvc_new_int : TargetLibCall_R<__2_YAPAXI_Z>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def TLI_msvc_new_int_nothrow : TargetLibCall_R<__2_YAPAXIABUnothrow_t_std___Z>;

/// void *operator new(unsigned long long);
def TLI_msvc_new_longlong : TargetLibCall_R<__2_YAPEAX_K_Z>;

/// void *operator new(unsigned long long, const std::nothrow_t&);
def TLI_msvc_new_longlong_nothrow
    : TargetLibCall_R<__2_YAPEAX_KAEBUnothrow_t_std___Z>;

/// void operator delete(void*);
def TLI_msvc_delete_ptr32 : TargetLibCall_R<__3_YAXPAX_Z>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_msvc_delete_ptr32_nothrow
    : TargetLibCall_R<__3_YAXPAXABUnothrow_t_std___Z>;

/// void operator delete(void*, unsigned int);
def TLI_msvc_delete_ptr32_int : TargetLibCall_R<__3_YAXPAXI_Z>;

/// void operator delete(void*);
def TLI_msvc_delete_ptr64 : TargetLibCall_R<__3_YAXPEAX_Z>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_msvc_delete_ptr64_nothrow
    : TargetLibCall_R<__3_YAXPEAXAEBUnothrow_t_std___Z>;

/// void operator delete(void*, unsigned long long);
def TLI_msvc_delete_ptr64_longlong : TargetLibCall_R<__3_YAXPEAX_K_Z>;

/// void *operator new[](unsigned int);
def TLI_msvc_new_array_int : TargetLibCall_R<___U_YAPAXI_Z>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def TLI_msvc_new_array_int_nothrow
    : TargetLibCall_R<___U_YAPAXIABUnothrow_t_std___Z>;

/// void *operator new[](unsigned long long);
def TLI_msvc_new_array_longlong : TargetLibCall_R<___U_YAPEAX_K_Z>;

/// void *operator new](unsigned long long, [const std::nothrow_t&);
def TLI_msvc_new_array_longlong_nothrow
    : TargetLibCall_R<___U_YAPEAX_KAEBUnothrow_t_std___Z>;

/// void operator delete[](void*);
def TLI_msvc_delete_array_ptr32 : TargetLibCall_R<___V_YAXPAX_Z>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_msvc_delete_array_ptr32_nothrow
    : TargetLibCall_R<___V_YAXPAXABUnothrow_t_std___Z>;

/// void operator delete](void*, [unsigned int);
def TLI_msvc_delete_array_ptr32_int : TargetLibCall_R<___V_YAXPAXI_Z>;

/// void operator delete[](void*);
def TLI_msvc_delete_array_ptr64 : TargetLibCall_R<___V_YAXPEAX_Z>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_msvc_delete_array_ptr64_nothrow
    : TargetLibCall_R<___V_YAXPEAXAEBUnothrow_t_std___Z>;

/// void operator delete](void*, [unsigned long long);
def TLI_msvc_delete_array_ptr64_longlong : TargetLibCall_R<___V_YAXPEAX_K_Z>;

/// int _IO_getc(_IO_FILE * __fp);
def TLI_under_IO_getc : TargetLibCall_R<_IO_getc>;

/// int _IO_putc(int __c, _IO_FILE * __fp);
def TLI_under_IO_putc : TargetLibCall_R<_IO_putc>;

/// void operator delete[](void*);
def TLI_ZdaPv : TargetLibCall_R<_ZdaPv>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_ZdaPvRKSt9nothrow_t : TargetLibCall_R<_ZdaPvRKSt9nothrow_t>;

/// void operator delete](void*, [std::align_val_t);
def TLI_ZdaPvSt11align_val_t : TargetLibCall_R<_ZdaPvSt11align_val_t>;

/// void operator delete](void*, [std::align_val_t, const std::nothrow_t&)
def TLI_ZdaPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZdaPvSt11align_val_tRKSt9nothrow_t>;

/// void operator delete](void*, [unsigned int);
def TLI_ZdaPvj : TargetLibCall_R<_ZdaPvj>;

/// void operator delete](void*, [unsigned int, std::align_val_t);
def TLI_ZdaPvjSt11align_val_t : TargetLibCall_R<_ZdaPvjSt11align_val_t>;

/// void operator delete](void*, [unsigned long);
def TLI_ZdaPvm : TargetLibCall_R<_ZdaPvm>;

/// void operator delete](void*, [unsigned long, std::align_val_t);
def TLI_ZdaPvmSt11align_val_t : TargetLibCall_R<_ZdaPvmSt11align_val_t>;

/// void operator delete(void*);
def TLI_ZdlPv : TargetLibCall_R<_ZdlPv>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_ZdlPvRKSt9nothrow_t : TargetLibCall_R<_ZdlPvRKSt9nothrow_t>;

/// void operator delete(void*, std::align_val_t)
def TLI_ZdlPvSt11align_val_t : TargetLibCall_R<_ZdlPvSt11align_val_t>;

/// void operator delete(void*, std::align_val_t, const std::nothrow_t&)
def TLI_ZdlPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZdlPvSt11align_val_tRKSt9nothrow_t>;

/// void operator delete(void*, unsigned int);
def TLI_ZdlPvj : TargetLibCall_R<_ZdlPvj>;

/// void operator delete(void*, unsigned int, std::align_val_t)
def TLI_ZdlPvjSt11align_val_t : TargetLibCall_R<_ZdlPvjSt11align_val_t>;

/// void operator delete(void*, unsigned long);
def TLI_ZdlPvm : TargetLibCall_R<_ZdlPvm>;

/// void operator delete(void*, unsigned long, std::align_val_t)
def TLI_ZdlPvmSt11align_val_t : TargetLibCall_R<_ZdlPvmSt11align_val_t>;

/// void *operator new[](unsigned int);
def TLI_Znaj : TargetLibCall_R<_Znaj>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def TLI_ZnajRKSt9nothrow_t : TargetLibCall_R<_ZnajRKSt9nothrow_t>;

/// void *operator new](unsigned int, [std::align_val_t)
def TLI_ZnajSt11align_val_t : TargetLibCall_R<_ZnajSt11align_val_t>;

/// void *operator new](unsigned int, [std::align_val_t, const std::nothrow_t&)
def TLI_ZnajSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZnajSt11align_val_tRKSt9nothrow_t>;

/// void *operator new[](unsigned long);
def TLI_Znam : TargetLibCall_R<_Znam>;

/// void *operator new](unsigned long, [__hot_cold_t)
/// Currently this and other operator new interfaces that take a __hot_cold_t
/// hint are supported by the open source version of tcmalloc, see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/new_extension.h
/// and for the definition of the __hot_cold_t parameter see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h
def TLI_Znam12__hot_cold_t : TargetLibCall_R<_Znam12__hot_cold_t>;

/// void *operator new](unsigned long, [const std::nothrow_t&);
def TLI_ZnamRKSt9nothrow_t : TargetLibCall_R<_ZnamRKSt9nothrow_t>;

/// void *operator new](unsigned long, [const std::nothrow_t&, __hot_cold_t)
def TLI_ZnamRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall_R<_ZnamRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new](unsigned long, [std::align_val_t)
def TLI_ZnamSt11align_val_t : TargetLibCall_R<_ZnamSt11align_val_t>;

/// void *operator new](unsigned long, [std::align_val_t, __hot_cold_t)
def TLI_ZnamSt11align_val_t12__hot_cold_t
    : TargetLibCall_R<_ZnamSt11align_val_t12__hot_cold_t>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&)
def TLI_ZnamSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZnamSt11align_val_tRKSt9nothrow_t>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def TLI_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall_R<_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new(unsigned int);
def TLI_Znwj : TargetLibCall_R<_Znwj>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def TLI_ZnwjRKSt9nothrow_t : TargetLibCall_R<_ZnwjRKSt9nothrow_t>;

/// void *operator new(unsigned int, std::align_val_t)
def TLI_ZnwjSt11align_val_t : TargetLibCall_R<_ZnwjSt11align_val_t>;

/// void *operator new(unsigned int, std::align_val_t, const std::nothrow_t&)
def TLI_ZnwjSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZnwjSt11align_val_tRKSt9nothrow_t>;

/// void *operator new(unsigned long);
def TLI_Znwm : TargetLibCall_R<_Znwm>;

/// void *operator new(unsigned long, __hot_cold_t)
def TLI_Znwm12__hot_cold_t : TargetLibCall_R<_Znwm12__hot_cold_t>;

/// void *operator new(unsigned long, const std::nothrow_t&);
def TLI_ZnwmRKSt9nothrow_t : TargetLibCall_R<_ZnwmRKSt9nothrow_t>;

/// void *operator new(unsigned long, const std::nothrow_t&, __hot_cold_t)
def TLI_ZnwmRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall_R<_ZnwmRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new(unsigned long, std::align_val_t)
def TLI_ZnwmSt11align_val_t : TargetLibCall_R<_ZnwmSt11align_val_t>;

/// void *operator new(unsigned long, std::align_val_t, __hot_cold_t)
def TLI_ZnwmSt11align_val_t12__hot_cold_t
    : TargetLibCall_R<_ZnwmSt11align_val_t12__hot_cold_t>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&)
def TLI_ZnwmSt11align_val_tRKSt9nothrow_t
    : TargetLibCall_R<_ZnwmSt11align_val_tRKSt9nothrow_t>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def TLI_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall_R<_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t>;

/// The following are variants of operator new which return the actual size
/// reserved by the allocator proposed in P0901R5 (Size feedback in operator
/// new). https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0901r5.html
/// They are implemented by tcmalloc, see source at
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h

/// __sized_ptr_t __size_returning_new(size_t size)
def TLI_size_returning_new : TargetLibCall_R<__size_returning_new>;

/// __sized_ptr_t __size_returning_new_hot_cold(size_t, __hot_cold_t)
def TLI_size_returning_new_hot_cold
    : TargetLibCall_R<__size_returning_new_hot_cold>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t)
def TLI_size_returning_new_aligned
    : TargetLibCall_R<__size_returning_new_aligned>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t,
/// __hot_cold_t)
def TLI_size_returning_new_aligned_hot_cold
    : TargetLibCall_R<__size_returning_new_aligned_hot_cold>;

/// double __acos_finite(double x);
def TLI_acos_finite : TargetLibCall_R<__acos_finite>;

/// float __acosf_finite(float x);
def TLI_acosf_finite : TargetLibCall_R<__acosf_finite>;

/// double __acosh_finite(double x);
def TLI_acosh_finite : TargetLibCall_R<__acosh_finite>;

/// float __acoshf_finite(float x);
def TLI_acoshf_finite : TargetLibCall_R<__acoshf_finite>;

/// long double __acoshl_finite(long double x);
def TLI_acoshl_finite : TargetLibCall_R<__acoshl_finite>;

/// long double __acosl_finite(long double x);
def TLI_acosl_finite : TargetLibCall_R<__acosl_finite>;

/// double __asin_finite(double x);
def TLI_asin_finite : TargetLibCall_R<__asin_finite>;

/// float __asinf_finite(float x);
def TLI_asinf_finite : TargetLibCall_R<__asinf_finite>;

/// long double __asinl_finite(long double x);
def TLI_asinl_finite : TargetLibCall_R<__asinl_finite>;

/// double atan2_finite(double y, double x);
def TLI_atan2_finite : TargetLibCall_R<__atan2_finite>;

/// float atan2f_finite(float y, float x);
def TLI_atan2f_finite : TargetLibCall_R<__atan2f_finite>;

/// long double atan2l_finite(long double y, long double x);
def TLI_atan2l_finite : TargetLibCall_R<__atan2l_finite>;

/// double __atanh_finite(double x);
def TLI_atanh_finite : TargetLibCall_R<__atanh_finite>;

/// float __atanhf_finite(float x);
def TLI_atanhf_finite : TargetLibCall_R<__atanhf_finite>;

/// long double __atanhl_finite(long double x);
def TLI_atanhl_finite : TargetLibCall_R<__atanhl_finite>;

/// void __atomic_load(size_t size, void *mptr, void *vptr, int smodel);
def TLI_atomic_load : TargetLibCall_R<__atomic_load>;

/// void __atomic_store(size_t size, void *mptr, void *vptr, int smodel);
def TLI_atomic_store : TargetLibCall_R<__atomic_store>;

/// double __cosh_finite(double x);
def TLI_cosh_finite : TargetLibCall_R<__cosh_finite>;

/// float __coshf_finite(float x);
def TLI_coshf_finite : TargetLibCall_R<__coshf_finite>;

/// long double __coshl_finite(long double x);
def TLI_coshl_finite : TargetLibCall_R<__coshl_finite>;

/// double __cospi(double x);
def TLI_cospi : TargetLibCall_R<__cospi>;

/// float __cospif(float x);
def TLI_cospif : TargetLibCall_R<__cospif>;

/// int __cxa_atexit(void (*f)(void *), void *p, void *d);
def TLI_cxa_atexit : TargetLibCall_R<__cxa_atexit>;

/// int atexit(void (*f)(void));
def TLI_atexit : TargetLibCall_R<atexit>;

/// void abort(void)
def TLI_abort : TargetLibCall_R<abort>;

/// void exit(int)
def TLI_exit : TargetLibCall_R<exit>;

/// void _Exit(int)
def TLI_Exit : TargetLibCall_R<_Exit>;

/// void std::terminate();
def TLI_terminate : TargetLibCall_R<_ZSt9terminatev>;

/// void __cxa_throw(void *, void *, void (*)(void *));
def TLI_cxa_throw : TargetLibCall_R<__cxa_throw>;

/// void __cxa_guard_abort(guard_t *guard);
/// guard_t is int64_t in Itanium ABI or int32_t on ARM eabi.
def TLI_cxa_guard_abort : TargetLibCall_R<__cxa_guard_abort>;

/// int __cxa_guard_acquire(guard_t *guard);
def TLI_cxa_guard_acquire : TargetLibCall_R<__cxa_guard_acquire>;

/// void __cxa_guard_release(guard_t *guard);
def TLI_cxa_guard_release : TargetLibCall_R<__cxa_guard_release>;

/// double __exp10_finite(double x);
def TLI_exp10_finite : TargetLibCall_R<__exp10_finite>;


/// float __exp10f_finite(float x);
def TLI_exp10f_finite : TargetLibCall_R<__exp10f_finite>;

/// long double __exp10l_finite(long double x);
def TLI_exp10l_finite : TargetLibCall_R<__exp10l_finite>;

/// double __exp2_finite(double x);
def TLI_exp2_finite : TargetLibCall_R<__exp2_finite>;

/// float __exp2f_finite(float x);
def TLI_exp2f_finite : TargetLibCall_R<__exp2f_finite>;

/// long double __exp2l_finite(long double x);
def TLI_exp2l_finite : TargetLibCall_R<__exp2l_finite_f128>;

/// double __exp_finite(double x);
def TLI_exp_finite : TargetLibCall_R<__exp_finite>;

/// float __expf_finite(float x);
def TLI_expf_finite : TargetLibCall_R<__expf_finite>;

/// long double __expl_finite(long double x);
def TLI_expl_finite : TargetLibCall_R<__expl_finite_f128>;

/// int __isoc99_scanf (const char *format, ...)
def TLI_dunder_isoc99_scanf : TargetLibCall_R<__isoc99_scanf>;

/// int __isoc99_sscanf(const char *s, const char *format, ...)
def TLI_dunder_isoc99_sscanf : TargetLibCall_R<__isoc99_sscanf>;

/// double __log10_finite(double x);
def TLI_log10_finite : TargetLibCall_R<__log10_finite>;

/// float __log10f_finite(float x);
def TLI_log10f_finite : TargetLibCall_R<__log10f_finite>;

/// long double __log10l_finite(long double x);
def TLI_log10l_finite : TargetLibCall_R<__log10l_finite_f128>;

/// double __log2_finite(double x);
def TLI_log2_finite : TargetLibCall_R<__log2_finite>;

/// float __log2f_finite(float x);
def TLI_log2f_finite : TargetLibCall_R<__log2f_finite>;

/// long double __log2l_finite(long double x);
def TLI_log2l_finite : TargetLibCall_R<__log2l_finite_f128>;

/// double __log_finite(double x);
def TLI_log_finite : TargetLibCall_R<__log_finite>;

/// float __logf_finite(float x);
def TLI_logf_finite : TargetLibCall_R<__logf_finite>;

/// long double __logl_finite(long double x);
def TLI_logl_finite : TargetLibCall_R<__logl_finite_f128>;

/// void *__memccpy_chk(void *dst, const void *src, int c, size_t n,
/// size_t dstsize)
def TLI_memccpy_chk : TargetLibCall_R<__memccpy_chk>;

/// void *__memcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_memcpy_chk : TargetLibCall_R<__memcpy_chk>;

/// void *__memmove_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_memmove_chk : TargetLibCall_R<__memmove_chk>;

/// void *__mempcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_mempcpy_chk : TargetLibCall_R<__mempcpy_chk>;

/// void *__memset_chk(void *s, int v, size_t n, size_t s1size);
def TLI_memset_chk : TargetLibCall_R<__memset_chk>;

// int __nvvm_reflect(const char *)
def TLI_nvvm_reflect : TargetLibCall_R<__nvvm_reflect>;

/// double __pow_finite(double x, double y);
def TLI_pow_finite : TargetLibCall_R<__pow_finite>;

/// float _powf_finite(float x, float y);
def TLI_powf_finite : TargetLibCall_R<__powf_finite>;

/// long double __powl_finite(long double x, long double y);
def TLI_powl_finite : TargetLibCall_R<__powl_finite_f128>;

/// double __sincospi_stret(double x);
def TLI_sincospi_stret : TargetLibCall_R<__sincospi_stret>;

/// float __sincospif_stret(float x);
def TLI_sincospif_stret : TargetLibCall_R<__sincospif_stret>;

/// double __sinh_finite(double x);
def TLI_sinh_finite : TargetLibCall_R<__sinh_finite>;

/// float _sinhf_finite(float x);
def TLI_sinhf_finite : TargetLibCall_R<__sinhf_finite>;

/// long double __sinhl_finite(long double x);
def TLI_sinhl_finite : TargetLibCall_R<__sinhl_finite>;

/// double __sinpi(double x);
def TLI_sinpi : TargetLibCall_R<__sinpi>;

/// float __sinpif(float x);
def TLI_sinpif : TargetLibCall_R<__sinpif>;

/// int __small_fprintf(FILE *stream, const char *format, ...);
def TLI_small_fprintf : TargetLibCall_R<__small_fprintf>;

/// int __small_printf(const char *format, ...);
def TLI_small_printf : TargetLibCall_R<__small_printf>;

/// int __small_sprintf(char *str, const char *format, ...);
def TLI_small_sprintf : TargetLibCall_R<__small_sprintf>;

/// int __snprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, ...);
def TLI_snprintf_chk : TargetLibCall_R<__snprintf_chk>;

/// int __sprintf_chk(char *str, int flags, size_t str_len,
/// const char *format, ...);
def TLI_sprintf_chk : TargetLibCall_R<__sprintf_chk>;

/// double __sqrt_finite(double x);
def TLI_sqrt_finite : TargetLibCall_R<__sqrt_finite>;

/// float __sqrt_finite(float x);
def TLI_sqrtf_finite : TargetLibCall_R<__sqrtf_finite>;

/// long double __sqrt_finite(long double x);
def TLI_sqrtl_finite : TargetLibCall_R<__sqrtl_finite>;

/// char *__stpcpy_chk(char *s1, const char *s2, size_t s1size);
def TLI_stpcpy_chk : TargetLibCall_R<__stpcpy_chk>;

/// char *__stpncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_stpncpy_chk : TargetLibCall_R<__stpncpy_chk>;

/// char *__strcat_chk(char *s1, const char *s2, size_t s1size);
def TLI_strcat_chk : TargetLibCall_R<__strcat_chk>; 

/// char *__strcpy_chk(char *s1, const char *s2, size_t s1size);
def TLI_strcpy_chk : TargetLibCall_R<__strcpy_chk>;

/// char * __strdup(const char *s);
def TLI_dunder_strdup : TargetLibCall_R<__strdup>;

/// size_t __strlcat_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def TLI_strlcat_chk : TargetLibCall_R<__strlcat_chk>;

/// size_t __strlcpy_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def TLI_strlcpy_chk : TargetLibCall_R<__strlcpy_chk>;

/// size_t __strlen_chk(const char *s1, size_t s1size);
def TLI_strlen_chk : TargetLibCall_R<__strlen_chk>;

/// char *strncat_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_strncat_chk : TargetLibCall_R<__strncat_chk>;

/// char *__strncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_strncpy_chk : TargetLibCall_R<__strncpy_chk>;

/// char *__strndup(const char *s, size_t n);
def TLI_dunder_strndup : TargetLibCall_R<__strndup>;

/// char * __strtok_r(char *s, const char *delim, char **save_ptr);
def TLI_dunder_strtok_r : TargetLibCall_R<__strtok_r>;

/// int __vsnprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, va_list ap);
def TLI_vsnprintf_chk : TargetLibCall_R<__vsnprintf_chk>;

/// int __vsprintf_chk(char *s, int flags, size_t slen, const char *format,
/// va_list ap);
def TLI_vsprintf_chk : TargetLibCall_R<__vsprintf_chk>;

/// int abs(int j);
def TLI_abs : TargetLibCall_R<abs>; 

/// int access(const char *path, int amode);
def TLI_access : TargetLibCall_R<access>;

/// double acos(double x);
def TLI_acos : TargetLibCall_R<acos>;

/// float acosf(float x);
def TLI_acosf : TargetLibCall_R<acosf>;

/// double acosh(double x);
def TLI_acosh : TargetLibCall_R<acosh>;

/// float acoshf(float x);
def TLI_acoshf : TargetLibCall_R<acoshf>;

/// long double acoshl(long double x);
def TLI_acoshl : TargetLibCall_R<acoshl>;

/// long double acosl(long double x);
def TLI_acosl : TargetLibCall_R<acosl_f128>;

/// void *aligned_alloc(size_t alignment, size_t size);
def TLI_aligned_alloc : TargetLibCall_R<aligned_alloc>;

/// double asin(double x);
def TLI_asin : TargetLibCall_R<asin>;

/// float asinf(float x);
def TLI_asinf : TargetLibCall_R<asinf>;

/// double asinh(double x);
def TLI_asinh : TargetLibCall_R<asinh>;

/// float asinhf(float x);
def TLI_asinhf : TargetLibCall_R<asinhf>;

/// long double asinhl(long double x);
def TLI_asinhl : TargetLibCall_R<asinhl>;

/// long double asinl(long double x);
def TLI_asinl : TargetLibCall_R<asinl_f128>;

/// double atan(double x);
def TLI_atan : TargetLibCall_R<atan>;

/// double atan2(double y, double x);
def TLI_atan2 : TargetLibCall_R<atan2>;

/// float atan2f(float y, float x);
def TLI_atan2f : TargetLibCall_R<atan2f>;

/// long double atan2l(long double y, long double x);
def TLI_atan2l : TargetLibCall_R<atan2l_f128>;

/// float atanf(float x);
def TLI_atanf : TargetLibCall_R<atanf>;

/// double atanh(double x);
def TLI_atanh : TargetLibCall_R<atanh>;


/// float atanhf(float x);
def TLI_atanhf : TargetLibCall_R<atanhf>;

/// long double atanhl(long double x);
def TLI_atanhl : TargetLibCall_R<atanhl>;

/// long double atanl(long double x);
def TLI_atanl : TargetLibCall_R<atanl_f128>;

/// double atof(const char *str);
def TLI_atof : TargetLibCall_R<atof>;

/// int atoi(const char *str);
def TLI_atoi : TargetLibCall_R<atoi>;

/// long atol(const char *str);
def TLI_atol : TargetLibCall_R<atol>;

/// long long atoll(const char *nptr);
def TLI_atoll : TargetLibCall_R<atoll>;

/// int bcmp(const void *s1, const void *s2, size_t n);
def TLI_bcmp : TargetLibCall_R<bcmp>;

/// void bcopy(const void *s1, void *s2, size_t n);
def TLI_bcopy : TargetLibCall_R<bcopy>;

/// void bzero(void *s, size_t n);
def TLI_bzero : TargetLibCall_R<bzero>;

/// double cabs(double complex z)
def TLI_cabs : TargetLibCall_R<cabs>;

/// float cabs(float complex z)
def TLI_cabsf : TargetLibCall_R<cabsf>;

/// long double cabs(long double complex z)
def TLI_cabsl : TargetLibCall_R<cabsl_f128>;

/// void *calloc(size_t count, size_t size);
def TLI_calloc : TargetLibCall_R<calloc>;

/// double cbrt(double x);
def TLI_cbrt : TargetLibCall_R<cbrt>;

/// float cbrtf(float x);
def TLI_cbrtf : TargetLibCall_R<cbrtf>;

/// long double cbrtl(long double x);
def TLI_cbrtl : TargetLibCall_R<cbrtl_f128>;

/// double ceil(double x);
def TLI_ceil : TargetLibCall_R<ceil>;

/// float ceilf(float x);
def TLI_ceilf : TargetLibCall_R<ceilf>;

/// long double ceill(long double x);
def TLI_ceill : TargetLibCall_R<ceill_f128>;

/// int chmod(const char *path, mode_t mode);
def TLI_chmod : TargetLibCall_R<chmod>;

/// int chown(const char *path, uid_t owner, gid_t group);
def TLI_chown : TargetLibCall_R<chown>;

/// void clearerr(FILE *stream);
def TLI_clearerr : TargetLibCall_R<clearerr>;

/// int closedir(DIR *dirp);
def TLI_closedir : TargetLibCall_R<closedir>;

/// double copysign(double x, double y);
def TLI_copysign : TargetLibCall_R<copysign>;

/// float copysignf(float x, float y);
def TLI_copysignf : TargetLibCall_R<copysignf>;

/// long double copysignl(long double x, long double y);
def TLI_copysignl : TargetLibCall_R<copysignl_f128>;

/// double cos(double x);
def TLI_cos : TargetLibCall_R<cos>;

/// float cosf(float x);
def TLI_cosf : TargetLibCall_R<cosf>;

/// double cosh(double x);
def TLI_cosh : TargetLibCall_R<cosh>;

/// float coshf(float x);
def TLI_coshf : TargetLibCall_R<coshf>;

/// long double coshl(long double x);
def TLI_coshl : TargetLibCall_R<coshl_f128>;

/// long double cosl(long double x);
def TLI_cosl : TargetLibCall_R<cosl_f128>;

/// char *ctermid(char *s);
def TLI_ctermid : TargetLibCall_R<ctermid>;

/// double erf(double x);
def TLI_erf : TargetLibCall_R<erf>;


/// float erff(float x);
def TLI_erff : TargetLibCall_R<erff>;

/// long double erfl(long double x);
def TLI_erfl : TargetLibCall_R<erfl>;

/// double tgamma(double x);
def TLI_tgamma : TargetLibCall_R<tgamma>;

/// float tgammaf(float x);
def TLI_tgammaf : TargetLibCall_R<tgammaf>;

/// long double tgammal(long double x);
def TLI_tgammal : TargetLibCall_R<tgammal_f128>;

/// int execl(const char *path, const char *arg, ...);
def TLI_execl : TargetLibCall_R<execl>;

/// int execle(const char *file, const char *arg, ..., char * const envp[]);
def TLI_execle : TargetLibCall_R<execle>;

/// int execlp(const char *file, const char *arg, ...);
def TLI_execlp : TargetLibCall_R<execlp>;

/// int execv(const char *path, char *const argv[]);
def TLI_execv : TargetLibCall_R<execv>;

/// int execvP(const char *file, const char *search_path, char *const argv[]);
def TLI_execvP : TargetLibCall_R<execvP>;

/// int execve(const char *filename, char *const argv], [char *const envp[]);
def TLI_execve : TargetLibCall_R<execve>;

/// int execvp(const char *file, char *const argv[]);
def TLI_execvp : TargetLibCall_R<execvp>;

/// int execvpe(const char *file, char *const argv], [char *const envp[]);
def TLI_execvpe : TargetLibCall_R<execvpe>;

/// double exp(double x);
def TLI_exp : TargetLibCall_R<exp>;

/// double exp10(double x);
def TLI_exp10 : TargetLibCall_R<exp10>;

/// float exp10f(float x);
def TLI_exp10f : TargetLibCall_R<exp10f>;

/// long double exp10l(long double x);
def TLI_exp10l : TargetLibCall_R<exp10l_f128>;

/// double exp2(double x);
def TLI_exp2 : TargetLibCall_R<exp2>;

/// float exp2f(float x);
def TLI_exp2f : TargetLibCall_R<exp2f>;

/// long double exp2l(long double x);
def TLI_exp2l : TargetLibCall_R<exp2l_f128>;

/// float expf(float x);
def TLI_expf : TargetLibCall_R<expf>;

/// long double expl(long double x);
def TLI_expl : TargetLibCall_R<expl_f128>;

/// double expm1(double x);
def TLI_expm1 : TargetLibCall_R<expm1>;

/// float expm1f(float x);
def TLI_expm1f : TargetLibCall_R<expm1f>;

/// long double expm1l(long double x);
def TLI_expm1l : TargetLibCall_R<expm1l>;

/// double fabs(double x);
def TLI_fabs : TargetLibCall_R<fabs>;

/// float fabsf(float x);
def TLI_fabsf : TargetLibCall_R<fabsf>;

/// long double fabsl(long double x);
def TLI_fabsl : TargetLibCall_R<fabsl>;

/// int fclose(FILE *stream);
def TLI_fclose : TargetLibCall_R<fclose>;

/// FILE *fdopen(int fildes, const char *mode);
def TLI_fdopen : TargetLibCall_R<fdopen>;

/// int feof(FILE *stream);
def TLI_feof : TargetLibCall_R<feof>;

/// int ferror(FILE *stream);
def TLI_ferror : TargetLibCall_R<ferror>;

/// int fflush(FILE *stream);
def TLI_fflush : TargetLibCall_R<fflush>;

/// int ffs(int i);
def TLI_ffs : TargetLibCall_R<ffs>;

/// int ffsl(long int i);
def TLI_ffsl : TargetLibCall_R<ffsl>;

/// int ffsll(long long int i);
def TLI_ffsll : TargetLibCall_R<ffsll>;

/// int fgetc(FILE *stream);
def TLI_fgetc : TargetLibCall_R<fgetc>;

/// int fgetc_unlocked(FILE *stream);
def TLI_fgetc_unlocked : TargetLibCall_R<fgetc_unlocked>;

/// int fgetpos(FILE *stream, fpos_t *pos);
def TLI_fgetpos : TargetLibCall_R<fgetpos>;

/// char *fgets(char *s, int n, FILE *stream);
def TLI_fgets : TargetLibCall_R<fgets>;

/// char *fgets_unlocked(char *s, int n, FILE *stream);
def TLI_fgets_unlocked : TargetLibCall_R<fgets_unlocked>;

/// int fileno(FILE *stream);
def TLI_fileno : TargetLibCall_R<fileno>;

/// int fiprintf(FILE *stream, const char *format, ...);
def TLI_fiprintf : TargetLibCall_R<fiprintf>;

/// void flockfile(FILE *file);
def TLI_flockfile : TargetLibCall_R<flockfile>;

/// double floor(double x);
def TLI_floor : TargetLibCall_R<floor>;

/// float floorf(float x);
def TLI_floorf : TargetLibCall_R<floorf>;

/// long double floorl(long double x);
def TLI_floorl : TargetLibCall_R<floorl_f128>;

/// int fls(int i);
def TLI_fls : TargetLibCall_R<fls>;

/// int flsl(long int i);
def TLI_flsl : TargetLibCall_R<flsl>;

/// int flsll(long long int i);
def TLI_flsll : TargetLibCall_R<flsll>;

// Calls to fmax and fmin library functions expand to the llvm.maxnnum and
// llvm.minnum intrinsics with the correct parameter types for the arguments
// (all types must match).
/// double fmax(double x, double y);
def TLI_fmax : TargetLibCall_R<fmax>;

/// float fmaxf(float x, float y);
def TLI_fmaxf : TargetLibCall_R<fmaxf>;

/// long double fmaxl(long double x, long double y);
def TLI_fmaxl : TargetLibCall_R<fmaxl_f128>;

/// double fmin(double x, double y);
def TLI_fmin : TargetLibCall_R<fmin>;

/// float fminf(float x, float y);
def TLI_fminf : TargetLibCall_R<fminf>;

/// long double fminl(long double x, long double y);
def TLI_fminl : TargetLibCall_R<fminl_f128>;

// Calls to fmaximum_num and fminimum_num library functions expand to the
// llvm.maximumnum and llvm.minimumnum intrinsics with the correct parameter
// types for the arguments (all types must match).
/// double fmaximum_num(double x, double y);
def TLI_fmaximum_num : TargetLibCall_R<fmaximum_num>;

/// float fmaximum_numf(float x, float y);
def TLI_fmaximum_numf : TargetLibCall_R<fmaximum_numf>;

/// long double fmaximum_numl(long double x, long double y);
def TLI_fmaximum_numl : TargetLibCall_R<fmaximum_numl_f128>;

/// double fminimum_num(double x, double y);
def TLI_fminimum_num : TargetLibCall_R<fminimum_num>;

/// float fminimum_numf(float x, float y);
def TLI_fminimum_numf : TargetLibCall_R<fminimum_numf>;

/// long double fminimum_numl(long double x, long double y);
def TLI_fminimum_numl : TargetLibCall_R<fminimum_numl_f128>;

/// double fmod(double x, double y);
def TLI_fmod : TargetLibCall_R<fmod>;

/// float fmodf(float x, float y);
def TLI_fmodf : TargetLibCall_R<fmodf>;

/// long double fmodl(long double x, long double y);
def TLI_fmodl : TargetLibCall_R<fmodl_f128>;

/// FILE *fopen(const char *filename, const char *mode);
def TLI_fopen : TargetLibCall_R<fopen>;

/// FILE *fopen64(const char *filename, const char *opentype)
def TLI_fopen64 : TargetLibCall_R<fopen64>;

/// int fork();
def TLI_fork : TargetLibCall_R<fork>;

/// int fprintf(FILE *stream, const char *format, ...);
def TLI_fprintf : TargetLibCall_R<fprintf>;

/// int fputc(int c, FILE *stream);
def TLI_fputc : TargetLibCall_R<fputc>;

/// int fputc_unlocked(int c, FILE *stream);
def TLI_fputc_unlocked : TargetLibCall_R<fputc_unlocked>;

/// int fputs(const char *s, FILE *stream);
def TLI_fputs : TargetLibCall_R<fputs>;

/// int fputs_unlocked(const char *s, FILE *stream);
def TLI_fputs_unlocked : TargetLibCall_R<fputs_unlocked>;

/// size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fread : TargetLibCall_R<fread>;

/// size_t fread_unlocked(void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fread_unlocked : TargetLibCall_R<fread_unlocked>;

/// void free(void *ptr);
def TLI_free : TargetLibCall_R<free>;

/// double frexp(double num, int *exp);
def TLI_frexp : TargetLibCall_R<frexp>;

/// float frexpf(float num, int *exp);
def TLI_frexpf : TargetLibCall_R<frexpf>;

/// long double frexpl(long double num, int *exp);
def TLI_frexpl : TargetLibCall_R<frexpl_f128>;

/// int fscanf(FILE *stream, const char *format, ... );
def TLI_fscanf : TargetLibCall_R<fscanf>;

/// int fseek(FILE *stream, long offset, int whence);
def TLI_fseek : TargetLibCall_R<fseek>;

/// int fseeko(FILE *stream, off_t offset, int whence);
def TLI_fseeko : TargetLibCall_R<fseeko>;

/// int fseeko64(FILE *stream, off64_t offset, int whence)
def TLI_fseeko64 : TargetLibCall_R<fseeko64>;

/// int fsetpos(FILE *stream, const fpos_t *pos);
def TLI_fsetpos : TargetLibCall_R<fsetpos>;

/// int fstat(int fildes, struct stat *buf);
def TLI_fstat : TargetLibCall_R<fstat>;

/// int fstat64(int filedes, struct stat64 *buf)
def TLI_fstat64 : TargetLibCall_R<fstat64>;

/// int fstatvfs(int fildes, struct statvfs *buf);
def TLI_fstatvfs : TargetLibCall_R<fstatvfs>;

/// int fstatvfs64(int fildes, struct statvfs64 *buf);
def TLI_fstatvfs64 : TargetLibCall_R<fstatvfs64>;

/// long ftell(FILE *stream);
def TLI_ftell : TargetLibCall_R<ftell>;

/// off_t ftello(FILE *stream);
def TLI_ftello : TargetLibCall_R<ftello>;

/// off64_t ftello64(FILE *stream)
def TLI_ftello64 : TargetLibCall_R<ftello64>;

/// int ftrylockfile(FILE *file);
def TLI_ftrylockfile : TargetLibCall_R<ftrylockfile>;

/// void funlockfile(FILE *file);
def TLI_funlockfile : TargetLibCall_R<funlockfile>;

/// size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fwrite : TargetLibCall_R<fwrite>;

/// size_t fwrite_unlocked(const void *ptr, size_t size, size_t nitems,
/// FILE *stream);
def TLI_fwrite_unlocked : TargetLibCall_R<fwrite_unlocked>;

/// int getc(FILE *stream);
def TLI_getc : TargetLibCall_R<getc>;

/// int getc_unlocked(FILE *stream);
def TLI_getc_unlocked : TargetLibCall_R<getc_unlocked>;

/// int getchar(void);
def TLI_getchar : TargetLibCall_R<getchar>;

/// int getchar_unlocked(void);
def TLI_getchar_unlocked : TargetLibCall_R<getchar_unlocked>;

/// char *getenv(const char *name);
def TLI_getenv : TargetLibCall_R<getenv>;

/// int getitimer(int which, struct itimerval *value);
def TLI_getitimer : TargetLibCall_R<getitimer>;

/// int getlogin_r(char *name, size_t namesize);
def TLI_getlogin_r : TargetLibCall_R<getlogin_r>;

/// struct passwd *getpwnam(const char *name);
def TLI_getpwnam : TargetLibCall_R<getpwnam>;

/// char *gets(char *s);
def TLI_gets : TargetLibCall_R<gets>;

/// int gettimeofday(struct timeval *tp, void *tzp);
def TLI_gettimeofday : TargetLibCall_R<gettimeofday>;

/// uint32_t htonl(uint32_t hostlong);
def TLI_htonl : TargetLibCall_R<htonl>;

/// uint16_t htons(uint16_t hostshort);
def TLI_htons : TargetLibCall_R<htons>;

/// double hypot(double x, double y);
def TLI_hypot : TargetLibCall_R<hypot>;

/// float hypotf(float x, float y);
def TLI_hypotf : TargetLibCall_R<hypotf>;

/// long double hypotl(long double x, long double y);
def TLI_hypotl : TargetLibCall_R<hypotl>;

/// int iprintf(const char *format, ...);
def TLI_iprintf : TargetLibCall_R<iprintf>;

/// int isascii(int c);
def TLI_isascii : TargetLibCall_R<isascii>;

/// int isdigit(int c);
def TLI_isdigit : TargetLibCall_R<isdigit>;

/// long int labs(long int j);
def TLI_labs : TargetLibCall_R<labs>;

/// int lchown(const char *path, uid_t owner, gid_t group);
def TLI_lchown : TargetLibCall_R<lchown>;

/// double ldexp(double x, int n);
def TLI_ldexp : TargetLibCall_R<ldexp>;

/// float ldexpf(float x, int n);
def TLI_ldexpf : TargetLibCall_R<ldexpf>;

/// long double ldexpl(long double x, int n);
def TLI_ldexpl : TargetLibCall_R<ldexpl_f128>;

/// long long int llabs(long long int j);
def TLI_llabs : TargetLibCall_R<llabs>;

/// double log(double x);
def TLI_log : TargetLibCall_R<log>;

/// double log10(double x);
def TLI_log10 : TargetLibCall_R<log10>;

/// float log10f(float x);
def TLI_log10f : TargetLibCall_R<log10f>;

/// long double log10l(long double x);
def TLI_log10l : TargetLibCall_R<log10l_f128>;

/// double log1p(double x);
def TLI_log1p : TargetLibCall_R<log1p>;

/// float log1pf(float x);
def TLI_log1pf : TargetLibCall_R<log1pf>;

/// long double log1pl(long double x);
def TLI_log1pl : TargetLibCall_R<log1pl>;

/// double log2(double x);
def TLI_log2 : TargetLibCall_R<log2>;

/// float log2f(float x);
def TLI_log2f : TargetLibCall_R<log2f>;

/// double long double log2l(long double x);
def TLI_log2l : TargetLibCall_R<log2l_f128>;

/// int ilogb(double x);
def TLI_ilogb : TargetLibCall_R<ilogb>;

/// int ilogbf(float x);
def TLI_ilogbf : TargetLibCall_R<ilogbf>;

/// int ilogbl(long double x);
def TLI_ilogbl : TargetLibCall_R<ilogbl>;

/// double logb(double x);
def TLI_logb : TargetLibCall_R<logb>;

/// float logbf(float x);
def TLI_logbf : TargetLibCall_R<logbf>;

/// long double logbl(long double x);
def TLI_logbl : TargetLibCall_R<logbl>;

/// float logf(float x);
def TLI_logf : TargetLibCall_R<logf>;

/// long double logl(long double x);
def TLI_logl : TargetLibCall_R<logl_f128>;

/// int lstat(const char *path, struct stat *buf);
def TLI_lstat : TargetLibCall_R<lstat>;

/// int lstat64(const char *path, struct stat64 *buf);
def TLI_lstat64 : TargetLibCall_R<lstat64>;

/// void *malloc(size_t size);
def TLI_malloc : TargetLibCall_R<malloc>;

/// void *memalign(size_t boundary, size_t size);
def TLI_memalign : TargetLibCall_R<memalign>;

/// void *memccpy(void *s1, const void *s2, int c, size_t n);
def TLI_memccpy : TargetLibCall_R<memccpy>;

/// void *memchr(const void *s, int c, size_t n);
def TLI_memchr : TargetLibCall_R<memchr>;

/// int memcmp(const void *s1, const void *s2, size_t n);
def TLI_memcmp : TargetLibCall_R<memcmp>;

/// void *memcpy(void *s1, const void *s2, size_t n);
def TLI_memcpy : TargetLibCall_R<memcpy>;

/// void *memmove(void *s1, const void *s2, size_t n);
def TLI_memmove : TargetLibCall_R<memmove>;

/// void *mempcpy(void *s1, const void *s2, size_t n);
def TLI_mempcpy : TargetLibCall_R<mempcpy>;

/// void *memrchr(const void *s, int c, size_t n);
def TLI_memrchr : TargetLibCall_R<memrchr>;

/// void *memset(void *b, int c, size_t len);
def TLI_memset : TargetLibCall_R<memset>;

/// void memset_pattern16(void *b, const void *pattern16, size_t len);
def TLI_memset_pattern16 : TargetLibCall_R<memset_pattern16>;

/// void memset_pattern4(void *b, const void *pattern4, size_t len);
def TLI_memset_pattern4 : TargetLibCall_R<memset_pattern4>;

/// void memset_pattern8(void *b, const void *pattern8, size_t len);
def TLI_memset_pattern8 : TargetLibCall_R<memset_pattern8>;

/// int mkdir(const char *path, mode_t mode);
def TLI_mkdir : TargetLibCall_R<mkdir>;

/// time_t mktime(struct tm *timeptr);
def TLI_mktime : TargetLibCall_R<mktime>;

/// double modf(double x, double *iptr);
def TLI_modf : TargetLibCall_R<modf>;

/// float modff(float, float *iptr);
def TLI_modff : TargetLibCall_R<modff>;

/// long double modfl(long double value, long double *iptr);
def TLI_modfl : TargetLibCall_R<modfl_f128>;

/// double nan(const char *arg);
def TLI_nan : TargetLibCall_R<nan>;

/// float nanf(const char *arg);
def TLI_nanf : TargetLibCall_R<nanf>;

/// long double nanl(const char *arg);
def TLI_nanl : TargetLibCall_R<nanl_f128>;

/// double nearbyint(double x);
def TLI_nearbyint : TargetLibCall_R<nearbyint>;

/// float nearbyintf(float x);
def TLI_nearbyintf : TargetLibCall_R<nearbyintf>;

/// long double nearbyintl(long double x);
def TLI_nearbyintl : TargetLibCall_R<nearbyintl_f128>;

/// double nextafter(double x, double y);
def TLI_nextafter : TargetLibCall_R<nextafter>;

/// float nextafterf(float x, float y);
def TLI_nextafterf : TargetLibCall_R<nextafterf>;

/// long double nextafterl(long double x, long double y);
def TLI_nextafterl : TargetLibCall_R<nextafterl>;

/// double nexttoward(double x, long double y);
def TLI_nexttoward : TargetLibCall_R<nexttoward>;

/// float nexttowardf(float x, long double y);
def TLI_nexttowardf : TargetLibCall_R<nexttowardf>;

/// long double nexttowardl(long double x, long double y);
def TLI_nexttowardl : TargetLibCall_R<nexttowardl_f128>;

/// uint32_t ntohl(uint32_t netlong);
def TLI_ntohl : TargetLibCall_R<ntohl>;

/// uint16_t ntohs(uint16_t netshort);
def TLI_ntohs : TargetLibCall_R<ntohs>;

/// int open(const char *path, int oflag, ... );
def TLI_open : TargetLibCall_R<open>;

/// int open64(const char *filename, int flags, [mode_t mode])
def TLI_open64 : TargetLibCall_R<open64>;

/// DIR *opendir(const char *dirname);
def TLI_opendir : TargetLibCall_R<opendir>;

/// int pclose(FILE *stream);
def TLI_pclose : TargetLibCall_R<pclose>;

/// void perror(const char *s);
def TLI_perror : TargetLibCall_R<perror>;

/// FILE *popen(const char *command, const char *mode);
def TLI_popen : TargetLibCall_R<popen>;

/// int posix_memalign(void **memptr, size_t alignment, size_t size);
def TLI_posix_memalign : TargetLibCall_R<posix_memalign>;

/// double pow(double x, double y);
def TLI_pow : TargetLibCall_R<pow>;

/// float powf(float x, float y);
def TLI_powf : TargetLibCall_R<powf>;

/// long double powl(long double x, long double y);
def TLI_powl : TargetLibCall_R<powl_f128>;

/// ssize_t pread(int fildes, void *buf, size_t nbyte, off_t offset);
def TLI_pread : TargetLibCall_R<pread>;

/// int printf(const char *format, ...);
def TLI_printf : TargetLibCall_R<printf>;

/// int putc(int c, FILE *stream);
def TLI_putc : TargetLibCall_R<putc>;

/// int putc_unlocked(int c, FILE *stream);
def TLI_putc_unlocked : TargetLibCall_R<putc_unlocked>;

/// int putchar(int c);
def TLI_putchar : TargetLibCall_R<putchar>;

/// int putchar_unlocked(int c);
def TLI_putchar_unlocked : TargetLibCall_R<putchar_unlocked>;

/// int puts(const char *s);
def TLI_puts : TargetLibCall_R<puts>;

/// void *pvalloc(size_t size);
def TLI_pvalloc : TargetLibCall_R<pvalloc>;

/// ssize_t pwrite(int fildes, const void *buf, size_t nbyte, off_t offset);
def TLI_pwrite : TargetLibCall_R<pwrite>;

/// void qsort(void *base, size_t nel, size_t width,
///            int (*compar)(const void *, const void *));
def TLI_qsort : TargetLibCall_R<qsort>;

/// ssize_t read(int fildes, void *buf, size_t nbyte);
def TLI_read : TargetLibCall_R<read>;

/// ssize_t readlink(const char *path, char *buf, size_t bufsize);
def TLI_readlink : TargetLibCall_R<readlink>;

/// void *realloc(void *ptr, size_t size);
def TLI_realloc : TargetLibCall_R<realloc>;

/// void *reallocf(void *ptr, size_t size);
def TLI_reallocf : TargetLibCall_R<reallocf>;

/// void *reallocarray(void *ptr, size_t nmemb, size_t size);
def TLI_reallocarray : TargetLibCall_R<reallocarray>;

/// char *realpath(const char *file_name, char *resolved_name);
def TLI_realpath : TargetLibCall_R<realpath>;

/// double remainder(double x, double y);
def TLI_remainder : TargetLibCall_R<remainder>;

/// float remainderf(float x, float y);
def TLI_remainderf : TargetLibCall_R<remainderf>;

/// long double remainderl(long double x, long double y);
def TLI_remainderl : TargetLibCall_R<remainderl_f128>;

/// double remquo(double x, double y, int *quo);
def TLI_remquo : TargetLibCall_R<remquo>;

/// float remquof(float x, float y, int *quo);
def TLI_remquof : TargetLibCall_R<remquof>;

/// long double remquol(long double x, long double y, int *quo);
def TLI_remquol : TargetLibCall_R<remquol_f128>;

/// double fdim(double x, double y);
def TLI_fdim : TargetLibCall_R<fdim>;

/// float fdimf(float x, float y);
def TLI_fdimf : TargetLibCall_R<fdimf>;

/// long double fdiml(long double x, long double y);
def TLI_fdiml : TargetLibCall_R<fdiml_f128>;

/// int remove(const char *path);
def TLI_remove : TargetLibCall_R<remove>;

/// int rename(const char *old, const char *new);
def TLI_rename : TargetLibCall_R<rename>;

/// void rewind(FILE *stream);
def TLI_rewind : TargetLibCall_R<rewind>;

/// double rint(double x);
def TLI_rint : TargetLibCall_R<rint>; 

/// float rintf(float x);
def TLI_rintf : TargetLibCall_R<rintf>;

/// long double rintl(long double x);
def TLI_rintl : TargetLibCall_R<rintl_f128>;

/// int rmdir(const char *path);
def TLI_rmdir : TargetLibCall_R<rmdir>;

/// double round(double x);
def TLI_round : TargetLibCall_R<round>;

/// double roundeven(double x);
def TLI_roundeven : TargetLibCall_R<roundeven>;

/// float roundevenf(float x);
def TLI_roundevenf : TargetLibCall_R<roundevenf>;

/// long double roundevenl(long double x);
def TLI_roundevenl : TargetLibCall_R<roundevenl_f128>;

/// float roundf(float x);
def TLI_roundf : TargetLibCall_R<roundf>;

/// long double roundl(long double x);
def TLI_roundl : TargetLibCall_R<roundl_f128>;

/// double scalbln(double arg, long exp);
def TLI_scalbln : TargetLibCall_R<scalbln>;

/// float scalblnf(float arg, long exp);
def TLI_scalblnf : TargetLibCall_R<scalblnf>;

/// long double scalblnl(long double arg, long exp);
def TLI_scalblnl : TargetLibCall_R<scalblnl_f128>;

/// double scalbn(double arg, int exp);
def TLI_scalbn : TargetLibCall_R<scalbn>;

/// float scalbnf(float arg, int exp);
def TLI_scalbnf : TargetLibCall_R<scalbnf>;

/// long double scalbnl(long double arg, int exp);
def TLI_scalbnl : TargetLibCall_R<scalbnl_f128>;

/// int scanf(const char *restrict format, ... );
def TLI_scanf : TargetLibCall_R<scanf>;

/// void setbuf(FILE *stream, char *buf);
def TLI_setbuf : TargetLibCall_R<setbuf>;

/// int setitimer(int which, const struct itimerval *value,
/// struct itimerval *ovalue);
def TLI_setitimer : TargetLibCall_R<setitimer>;

/// int setvbuf(FILE *stream, char *buf, int type, size_t size);
def TLI_setvbuf : TargetLibCall_R<setvbuf>;

/// double sin(double x);
def TLI_sin : TargetLibCall_R<sin>;

/// float sinf(float x);
def TLI_sinf : TargetLibCall_R<sinf>;

/// double sinh(double x);
def TLI_sinh : TargetLibCall_R<sinh>;

/// float sinhf(float x);
def TLI_sinhf : TargetLibCall_R<sinhf>;

/// long double sinhl(long double x);
def TLI_sinhl : TargetLibCall_R<sinhl_f128>;

/// long double sinl(long double x);
def TLI_sinl : TargetLibCall_R<sinl_f128>;

/// void sincos(double x, double *sin_out, double *cos_out);
def TLI_sincos : TargetLibCall_R<sincos>;

/// void sincosf(float x, float *sin_out, float *cos_out);
def TLI_sincosf : TargetLibCall_R<sincosf>;

/// void sincosl(long double x, long double *sin_out, long double *cos_out);
def TLI_sincosl : TargetLibCall_R<sincosl_f128>;

/// int siprintf(char *str, const char *format, ...);
def TLI_siprintf : TargetLibCall_R<siprintf>;

/// int snprintf(char *s, size_t n, const char *format, ...);
def TLI_snprintf : TargetLibCall_R<snprintf>;

/// int sprintf(char *str, const char *format, ...);
def TLI_sprintf : TargetLibCall_R<sprintf>;

/// double sqrt(double x);
def TLI_sqrt : TargetLibCall_R<sqrt>;

/// float sqrtf(float x);
def TLI_sqrtf : TargetLibCall_R<sqrtf>;

/// long double sqrtl(long double x);
def TLI_sqrtl : TargetLibCall_R<sqrtl_f128>;

/// int sscanf(const char *s, const char *format, ... );
def TLI_sscanf : TargetLibCall_R<sscanf>;

/// int stat(const char *path, struct stat *buf);
def TLI_stat : TargetLibCall_R<stat>;

/// int stat64(const char *path, struct stat64 *buf);
def TLI_stat64 : TargetLibCall_R<stat64>;

/// int statvfs(const char *path, struct statvfs *buf);
def TLI_statvfs : TargetLibCall_R<statvfs>;

/// int statvfs64(const char *path, struct statvfs64 *buf)
def TLI_statvfs64 : TargetLibCall_R<statvfs64>;

/// char *stpcpy(char *s1, const char *s2);
def TLI_stpcpy : TargetLibCall_R<stpcpy>;

/// char *stpncpy(char *s1, const char *s2, size_t n);
def TLI_stpncpy : TargetLibCall_R<stpncpy>;

/// int strcasecmp(const char *s1, const char *s2);
def TLI_strcasecmp : TargetLibCall_R<strcasecmp>;

/// char *strcat(char *s1, const char *s2);
def TLI_strcat : TargetLibCall_R<strcat>;

/// char *strchr(const char *s, int c);
def TLI_strchr : TargetLibCall_R<strchr>;

/// int strcmp(const char *s1, const char *s2);
def TLI_strcmp : TargetLibCall_R<strcmp>;

/// int strcoll(const char *s1, const char *s2);
def TLI_strcoll : TargetLibCall_R<strcoll>;

/// char *strcpy(char *s1, const char *s2);
def TLI_strcpy : TargetLibCall_R<strcpy>;

/// size_t strcspn(const char *s1, const char *s2);
def TLI_strcspn : TargetLibCall_R<strcspn>;

/// char *strdup(const char *s1);
def TLI_strdup : TargetLibCall_R<strdup>;

/// size_t strlcat(char *dst, const char *src, size_t size);
def TLI_strlcat : TargetLibCall_R<strlcat>;

/// size_t strlcpy(char *dst, const char *src, size_t size);
def TLI_strlcpy : TargetLibCall_R<strlcpy>;

/// size_t strlen(const char *s);
def TLI_strlen : TargetLibCall_R<strlen>;

/// int strncasecmp(const char *s1, const char *s2, size_t n);
def TLI_strncasecmp : TargetLibCall_R<strncasecmp>;

/// char *strncat(char *s1, const char *s2, size_t n);
def TLI_strncat : TargetLibCall_R<strncat>;

/// int strncmp(const char *s1, const char *s2, size_t n);
def TLI_strncmp : TargetLibCall_R<strncmp>;

/// char *strncpy(char *s1, const char *s2, size_t n);
def TLI_strncpy : TargetLibCall_R<strncpy>;

/// char *strndup(const char *s1, size_t n);
def TLI_strndup : TargetLibCall_R<strndup>;

/// size_t strnlen(const char *s, size_t maxlen);
def TLI_strnlen : TargetLibCall_R<strnlen>;

/// char *strpbrk(const char *s1, const char *s2);
def TLI_strpbrk : TargetLibCall_R<strpbrk>;

/// char *strrchr(const char *s, int c);
def TLI_strrchr : TargetLibCall_R<strrchr>;

/// size_t strspn(const char *s1, const char *s2);
def TLI_strspn : TargetLibCall_R<strspn>;

/// char *strstr(const char *s1, const char *s2);
def TLI_strstr : TargetLibCall_R<strstr>;

/// double strtod(const char *nptr, char **endptr);
def TLI_strtod : TargetLibCall_R<strtod>;

/// float strtof(const char *nptr, char **endptr);
def TLI_strtof : TargetLibCall_R<strtof>;

/// char *strtok(char *s1, const char *s2);
def TLI_strtok : TargetLibCall_R<strtok>;

/// char *strtok_r(char *s, const char *sep, char **lasts);
def TLI_strtok_r : TargetLibCall_R<strtok_r>;

/// long int strtol(const char *nptr, char **endptr, int base);
def TLI_strtol : TargetLibCall_R<strtol>;

/// long double strtold(const char *nptr, char **endptr);
def TLI_strtold : TargetLibCall_R<strtold>;

/// long long int strtoll(const char *nptr, char **endptr, int base);
def TLI_strtoll : TargetLibCall_R<strtoll>;

/// unsigned long int strtoul(const char *nptr, char **endptr, int base);
def TLI_strtoul : TargetLibCall_R<strtoul>;

/// unsigned long long int strtoull(const char *nptr, char **endptr, int base);
def TLI_strtoull : TargetLibCall_R<strtoull>;

/// size_t strxfrm(char *s1, const char *s2, size_t n);
def TLI_strxfrm : TargetLibCall_R<strxfrm>;

/// int system(const char *command);
def TLI_system : TargetLibCall_R<system>;

/// double tan(double x);
def TLI_tan : TargetLibCall_R<tan>;

/// float tanf(float x);
def TLI_tanf : TargetLibCall_R<tanf>;

/// double tanh(double x);
def TLI_tanh : TargetLibCall_R<tanh>;

/// float tanhf(float x);
def TLI_tanhf : TargetLibCall_R<tanhf>;

/// long double tanhl(long double x);
def TLI_tanhl : TargetLibCall_R<tanhl_f128>;

/// long double tanl(long double x);
def TLI_tanl : TargetLibCall_R<tanl_f128>;

/// clock_t times(struct tms *buffer);
def TLI_times : TargetLibCall_R<times>;

/// FILE *tmpfile(void);
def TLI_tmpfile : TargetLibCall_R<tmpfile>;

/// FILE *tmpfile64(void)
def TLI_tmpfile64 : TargetLibCall_R<tmpfile64>;

/// int toascii(int c);
def TLI_toascii : TargetLibCall_R<toascii>;

/// double trunc(double x);
def TLI_trunc : TargetLibCall_R<trunc>;

/// float truncf(float x);
def TLI_truncf : TargetLibCall_R<truncf>;

/// long double truncl(long double x);
def TLI_truncl : TargetLibCall_R<truncl_f128>;

/// int uname(struct utsname *name);
def TLI_uname : TargetLibCall_R<uname>;

/// int ungetc(int c, FILE *stream);
def TLI_ungetc : TargetLibCall_R<ungetc>;

/// int unlink(const char *path);
def TLI_unlink : TargetLibCall_R<unlink>;

/// int unsetenv(const char *name);
def TLI_unsetenv : TargetLibCall_R<unsetenv>;

/// int utime(const char *path, const struct utimbuf *times);
def TLI_utime : TargetLibCall_R<utime>;

/// int utimes(const char *path, const struct timeval times[2]);
def TLI_utimes : TargetLibCall_R<utimes>;

/// void *valloc(size_t size);
def TLI_valloc : TargetLibCall_R<valloc>;

/// void *vec_calloc(size_t count, size_t size);
def TLI_vec_calloc : TargetLibCall_R<vec_calloc>;

/// void vec_free(void *ptr);
def TLI_vec_free : TargetLibCall_R<vec_free>;

/// void *vec_malloc(size_t size);
def TLI_vec_malloc : TargetLibCall_R<vec_malloc>;

/// void *vec_realloc(void *ptr, size_t size);
def TLI_vec_realloc : TargetLibCall_R<vec_realloc>;

/// int vfprintf(FILE *stream, const char *format, va_list ap);
def TLI_vfprintf : TargetLibCall_R<vfprintf>;

/// int vfscanf(FILE *stream, const char *format, va_list arg);
def TLI_vfscanf : TargetLibCall_R<vfscanf>;

/// int vprintf(const char *restrict format, va_list ap);
def TLI_vprintf : TargetLibCall_R<vprintf>;

/// int vscanf(const char *format, va_list arg);
def TLI_vscanf : TargetLibCall_R<vscanf>;

/// int vsnprintf(char *s, size_t n, const char *format, va_list ap);
def TLI_vsnprintf : TargetLibCall_R<vsnprintf>;

/// int vsprintf(char *s, const char *format, va_list ap);
def TLI_vsprintf : TargetLibCall_R<vsprintf>;

/// int vsscanf(const char *s, const char *format, va_list arg);
def TLI_vsscanf : TargetLibCall_R<vsscanf>;

/// size_t wcslen (const wchar_t* wcs);
def TLI_wcslen : TargetLibCall_R<wcslen>;

/// ssize_t write(int fildes, const void *buf, size_t nbyte);
def TLI_write : TargetLibCall_R<write>;
