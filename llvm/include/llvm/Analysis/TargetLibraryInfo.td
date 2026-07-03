//===-- TargetLibraryInfo.td - File that describes library functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/IR/RuntimeLibcalls.td"

/// void *operator new(unsigned int);
def TLI_msvc_new_int : TargetLibCall<__2_YAPAXI_Z>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def TLI_msvc_new_int_nothrow : TargetLibCall<__2_YAPAXIABUnothrow_t_std___Z>;

/// void *operator new(unsigned long long);
def TLI_msvc_new_longlong : TargetLibCall<__2_YAPEAX_K_Z>;

/// void *operator new(unsigned long long, const std::nothrow_t&);
def TLI_msvc_new_longlong_nothrow
    : TargetLibCall<__2_YAPEAX_KAEBUnothrow_t_std___Z>;

/// void operator delete(void*);
def TLI_msvc_delete_ptr32 : TargetLibCall<__3_YAXPAX_Z>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_msvc_delete_ptr32_nothrow
    : TargetLibCall<__3_YAXPAXABUnothrow_t_std___Z>;

/// void operator delete(void*, unsigned int);
def TLI_msvc_delete_ptr32_int : TargetLibCall<__3_YAXPAXI_Z>;

/// void operator delete(void*);
def TLI_msvc_delete_ptr64 : TargetLibCall<__3_YAXPEAX_Z>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_msvc_delete_ptr64_nothrow
    : TargetLibCall<__3_YAXPEAXAEBUnothrow_t_std___Z>;

/// void operator delete(void*, unsigned long long);
def TLI_msvc_delete_ptr64_longlong : TargetLibCall<__3_YAXPEAX_K_Z>;

/// void *operator new[](unsigned int);
def TLI_msvc_new_array_int : TargetLibCall<___U_YAPAXI_Z>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def TLI_msvc_new_array_int_nothrow
    : TargetLibCall<___U_YAPAXIABUnothrow_t_std___Z>;

/// void *operator new[](unsigned long long);
def TLI_msvc_new_array_longlong : TargetLibCall<___U_YAPEAX_K_Z>;

/// void *operator new](unsigned long long, [const std::nothrow_t&);
def TLI_msvc_new_array_longlong_nothrow
    : TargetLibCall<___U_YAPEAX_KAEBUnothrow_t_std___Z>;

/// void operator delete[](void*);
def TLI_msvc_delete_array_ptr32 : TargetLibCall<___V_YAXPAX_Z>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_msvc_delete_array_ptr32_nothrow
    : TargetLibCall<___V_YAXPAXABUnothrow_t_std___Z>;

/// void operator delete](void*, [unsigned int);
def TLI_msvc_delete_array_ptr32_int : TargetLibCall<___V_YAXPAXI_Z>;

/// void operator delete[](void*);
def TLI_msvc_delete_array_ptr64 : TargetLibCall<___V_YAXPEAX_Z>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_msvc_delete_array_ptr64_nothrow
    : TargetLibCall<___V_YAXPEAXAEBUnothrow_t_std___Z>;

/// void operator delete](void*, [unsigned long long);
def TLI_msvc_delete_array_ptr64_longlong : TargetLibCall<___V_YAXPEAX_K_Z>;

/// int _IO_getc(_IO_FILE * __fp);
def TLI_under_IO_getc : TargetLibCall<_IO_getc>;

/// int _IO_putc(int __c, _IO_FILE * __fp);
def TLI_under_IO_putc : TargetLibCall<_IO_putc>;

/// void operator delete[](void*);
def TLI_ZdaPv : TargetLibCall<_ZdaPv>;

/// void operator delete](void*, [const std::nothrow_t&);
def TLI_ZdaPvRKSt9nothrow_t : TargetLibCall<_ZdaPvRKSt9nothrow_t>;

/// void operator delete](void*, [std::align_val_t);
def TLI_ZdaPvSt11align_val_t : TargetLibCall<_ZdaPvSt11align_val_t>;

/// void operator delete](void*, [std::align_val_t, const std::nothrow_t&)
def TLI_ZdaPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZdaPvSt11align_val_tRKSt9nothrow_t>;

/// void operator delete](void*, [unsigned int);
def TLI_ZdaPvj : TargetLibCall<_ZdaPvj>;

/// void operator delete](void*, [unsigned int, std::align_val_t);
def TLI_ZdaPvjSt11align_val_t : TargetLibCall<_ZdaPvjSt11align_val_t>;

/// void operator delete](void*, [unsigned long);
def TLI_ZdaPvm : TargetLibCall<_ZdaPvm>;

/// void operator delete](void*, [unsigned long, std::align_val_t);
def TLI_ZdaPvmSt11align_val_t : TargetLibCall<_ZdaPvmSt11align_val_t>;

/// void operator delete(void*);
def TLI_ZdlPv : TargetLibCall<_ZdlPv>;

/// void operator delete(void*, const std::nothrow_t&);
def TLI_ZdlPvRKSt9nothrow_t : TargetLibCall<_ZdlPvRKSt9nothrow_t>;

/// void operator delete(void*, std::align_val_t)
def TLI_ZdlPvSt11align_val_t : TargetLibCall<_ZdlPvSt11align_val_t>;

/// void operator delete(void*, std::align_val_t, const std::nothrow_t&)
def TLI_ZdlPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZdlPvSt11align_val_tRKSt9nothrow_t>;

/// void operator delete(void*, unsigned int);
def TLI_ZdlPvj : TargetLibCall<_ZdlPvj>;

/// void operator delete(void*, unsigned int, std::align_val_t)
def TLI_ZdlPvjSt11align_val_t : TargetLibCall<_ZdlPvjSt11align_val_t>;

/// void operator delete(void*, unsigned long);
def TLI_ZdlPvm : TargetLibCall<_ZdlPvm>;

/// void operator delete(void*, unsigned long, std::align_val_t)
def TLI_ZdlPvmSt11align_val_t : TargetLibCall<_ZdlPvmSt11align_val_t>;

/// void *operator new[](unsigned int);
def TLI_Znaj : TargetLibCall<_Znaj>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def TLI_ZnajRKSt9nothrow_t : TargetLibCall<_ZnajRKSt9nothrow_t>;

/// void *operator new](unsigned int, [std::align_val_t)
def TLI_ZnajSt11align_val_t : TargetLibCall<_ZnajSt11align_val_t>;

/// void *operator new](unsigned int, [std::align_val_t, const std::nothrow_t&)
def TLI_ZnajSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZnajSt11align_val_tRKSt9nothrow_t>;

/// void *operator new[](unsigned long);
def TLI_Znam : TargetLibCall<_Znam>;

/// void *operator new](unsigned long, [__hot_cold_t)
/// Currently this and other operator new interfaces that take a __hot_cold_t
/// hint are supported by the open source version of tcmalloc, see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/new_extension.h
/// and for the definition of the __hot_cold_t parameter see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h
def TLI_Znam12__hot_cold_t : TargetLibCall<_Znam12__hot_cold_t>;

/// void *operator new](unsigned long, [const std::nothrow_t&);
def TLI_ZnamRKSt9nothrow_t : TargetLibCall<_ZnamRKSt9nothrow_t>;

/// void *operator new](unsigned long, [const std::nothrow_t&, __hot_cold_t)
def TLI_ZnamRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<_ZnamRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new](unsigned long, [std::align_val_t)
def TLI_ZnamSt11align_val_t : TargetLibCall<_ZnamSt11align_val_t>;

/// void *operator new](unsigned long, [std::align_val_t, __hot_cold_t)
def TLI_ZnamSt11align_val_t12__hot_cold_t
    : TargetLibCall<_ZnamSt11align_val_t12__hot_cold_t>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&)
def TLI_ZnamSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZnamSt11align_val_tRKSt9nothrow_t>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def TLI_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new(unsigned int);
def TLI_Znwj : TargetLibCall<_Znwj>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def TLI_ZnwjRKSt9nothrow_t : TargetLibCall<_ZnwjRKSt9nothrow_t>;

/// void *operator new(unsigned int, std::align_val_t)
def TLI_ZnwjSt11align_val_t : TargetLibCall<_ZnwjSt11align_val_t>;

/// void *operator new(unsigned int, std::align_val_t, const std::nothrow_t&)
def TLI_ZnwjSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZnwjSt11align_val_tRKSt9nothrow_t>;

/// void *operator new(unsigned long);
def TLI_Znwm : TargetLibCall<_Znwm>;

/// void *operator new(unsigned long, __hot_cold_t)
def TLI_Znwm12__hot_cold_t : TargetLibCall<_Znwm12__hot_cold_t>;

/// void *operator new(unsigned long, const std::nothrow_t&);
def TLI_ZnwmRKSt9nothrow_t : TargetLibCall<_ZnwmRKSt9nothrow_t>;

/// void *operator new(unsigned long, const std::nothrow_t&, __hot_cold_t)
def TLI_ZnwmRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<_ZnwmRKSt9nothrow_t12__hot_cold_t>;

/// void *operator new(unsigned long, std::align_val_t)
def TLI_ZnwmSt11align_val_t : TargetLibCall<_ZnwmSt11align_val_t>;

/// void *operator new(unsigned long, std::align_val_t, __hot_cold_t)
def TLI_ZnwmSt11align_val_t12__hot_cold_t
    : TargetLibCall<_ZnwmSt11align_val_t12__hot_cold_t>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&)
def TLI_ZnwmSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<_ZnwmSt11align_val_tRKSt9nothrow_t>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def TLI_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t>;

/// The following are variants of operator new which return the actual size
/// reserved by the allocator proposed in P0901R5 (Size feedback in operator
/// new). https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0901r5.html
/// They are implemented by tcmalloc, see source at
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h

/// __sized_ptr_t __size_returning_new(size_t size)
def TLI_size_returning_new : TargetLibCall<__size_returning_new>;

/// __sized_ptr_t __size_returning_new_hot_cold(size_t, __hot_cold_t)
def TLI_size_returning_new_hot_cold
    : TargetLibCall<__size_returning_new_hot_cold>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t)
def TLI_size_returning_new_aligned
    : TargetLibCall<__size_returning_new_aligned>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t,
/// __hot_cold_t)
def TLI_size_returning_new_aligned_hot_cold
    : TargetLibCall<__size_returning_new_aligned_hot_cold>;

/// double __acos_finite(double x);
def TLI_acos_finite : TargetLibCall<__acos_finite>;

/// float __acosf_finite(float x);
def TLI_acosf_finite : TargetLibCall<__acosf_finite>;

/// double __acosh_finite(double x);
def TLI_acosh_finite : TargetLibCall<__acosh_finite>;

/// float __acoshf_finite(float x);
def TLI_acoshf_finite : TargetLibCall<__acoshf_finite>;

/// long double __acoshl_finite(long double x);
def TLI_acoshl_finite : TargetLibCall<__acoshl_finite>;

/// long double __acosl_finite(long double x);
def TLI_acosl_finite : TargetLibCall<__acosl_finite>;

/// double __asin_finite(double x);
def TLI_asin_finite : TargetLibCall<__asin_finite>;

/// float __asinf_finite(float x);
def TLI_asinf_finite : TargetLibCall<__asinf_finite>;

/// long double __asinl_finite(long double x);
def TLI_asinl_finite : TargetLibCall<__asinl_finite>;

/// double atan2_finite(double y, double x);
def TLI_atan2_finite : TargetLibCall<__atan2_finite>;

/// float atan2f_finite(float y, float x);
def TLI_atan2f_finite : TargetLibCall<__atan2f_finite>;

/// long double atan2l_finite(long double y, long double x);
def TLI_atan2l_finite : TargetLibCall<__atan2l_finite>;

/// double __atanh_finite(double x);
def TLI_atanh_finite : TargetLibCall<__atanh_finite>;

/// float __atanhf_finite(float x);
def TLI_atanhf_finite : TargetLibCall<__atanhf_finite>;

/// long double __atanhl_finite(long double x);
def TLI_atanhl_finite : TargetLibCall<__atanhl_finite>;

/// void __atomic_load(size_t size, void *mptr, void *vptr, int smodel);
def TLI_atomic_load : TargetLibCall<__atomic_load>;

/// void __atomic_store(size_t size, void *mptr, void *vptr, int smodel);
def TLI_atomic_store : TargetLibCall<__atomic_store>;

/// double __cosh_finite(double x);
def TLI_cosh_finite : TargetLibCall<__cosh_finite>;

/// float __coshf_finite(float x);
def TLI_coshf_finite : TargetLibCall<__coshf_finite>;

/// long double __coshl_finite(long double x);
def TLI_coshl_finite : TargetLibCall<__coshl_finite>;

/// double __cospi(double x);
def TLI_cospi : TargetLibCall<__cospi>;

/// float __cospif(float x);
def TLI_cospif : TargetLibCall<__cospif>;

/// int __cxa_atexit(void (*f)(void *), void *p, void *d);
def TLI_cxa_atexit : TargetLibCall<__cxa_atexit>;

/// int atexit(void (*f)(void));
def TLI_atexit : TargetLibCall<atexit>;

/// void abort(void)
def TLI_abort : TargetLibCall<abort>;

/// void exit(int)
def TLI_exit : TargetLibCall<exit>;

/// void _Exit(int)
def TLI_Exit : TargetLibCall<_Exit>;

/// void std::terminate();
def TLI_terminate : TargetLibCall<_ZSt9terminatev>;

/// void __cxa_throw(void *, void *, void (*)(void *));
def TLI_cxa_throw : TargetLibCall<__cxa_throw>;

/// void __cxa_guard_abort(guard_t *guard);
/// guard_t is int64_t in Itanium ABI or int32_t on ARM eabi.
def TLI_cxa_guard_abort : TargetLibCall<__cxa_guard_abort>;

/// int __cxa_guard_acquire(guard_t *guard);
def TLI_cxa_guard_acquire : TargetLibCall<__cxa_guard_acquire>;

/// void __cxa_guard_release(guard_t *guard);
def TLI_cxa_guard_release : TargetLibCall<__cxa_guard_release>;

/// double __exp10_finite(double x);
def TLI_exp10_finite : TargetLibCall<__exp10_finite>;


/// float __exp10f_finite(float x);
def TLI_exp10f_finite : TargetLibCall<__exp10f_finite>;

/// long double __exp10l_finite(long double x);
def TLI_exp10l_finite : TargetLibCall<__exp10l_finite>;

/// double __exp2_finite(double x);
def TLI_exp2_finite : TargetLibCall<__exp2_finite>;

/// float __exp2f_finite(float x);
def TLI_exp2f_finite : TargetLibCall<__exp2f_finite>;

/// long double __exp2l_finite(long double x);
def TLI_exp2l_finite : TargetLibCall<__exp2l_finite_f128>;

/// double __exp_finite(double x);
def TLI_exp_finite : TargetLibCall<__exp_finite>;

/// float __expf_finite(float x);
def TLI_expf_finite : TargetLibCall<__expf_finite>;

/// long double __expl_finite(long double x);
def TLI_expl_finite : TargetLibCall<__expl_finite_f128>;

/// int __isoc99_scanf (const char *format, ...)
def TLI_dunder_isoc99_scanf : TargetLibCall<__isoc99_scanf>;

/// int __isoc99_sscanf(const char *s, const char *format, ...)
def TLI_dunder_isoc99_sscanf : TargetLibCall<__isoc99_sscanf>;

/// double __log10_finite(double x);
def TLI_log10_finite : TargetLibCall<__log10_finite>;

/// float __log10f_finite(float x);
def TLI_log10f_finite : TargetLibCall<__log10f_finite>;

/// long double __log10l_finite(long double x);
def TLI_log10l_finite : TargetLibCall<__log10l_finite_f128>;

/// double __log2_finite(double x);
def TLI_log2_finite : TargetLibCall<__log2_finite>;

/// float __log2f_finite(float x);
def TLI_log2f_finite : TargetLibCall<__log2f_finite>;

/// long double __log2l_finite(long double x);
def TLI_log2l_finite : TargetLibCall<__log2l_finite_f128>;

/// double __log_finite(double x);
def TLI_log_finite : TargetLibCall<__log_finite>;

/// float __logf_finite(float x);
def TLI_logf_finite : TargetLibCall<__logf_finite>;

/// long double __logl_finite(long double x);
def TLI_logl_finite : TargetLibCall<__logl_finite_f128>;

/// void *__memccpy_chk(void *dst, const void *src, int c, size_t n,
/// size_t dstsize)
def TLI_memccpy_chk : TargetLibCall<__memccpy_chk>;

/// void *__memcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_memcpy_chk : TargetLibCall<__memcpy_chk>;

/// void *__memmove_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_memmove_chk : TargetLibCall<__memmove_chk>;

/// void *__mempcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def TLI_mempcpy_chk : TargetLibCall<__mempcpy_chk>;

/// void *__memset_chk(void *s, int v, size_t n, size_t s1size);
def TLI_memset_chk : TargetLibCall<__memset_chk>;

// int __nvvm_reflect(const char *)
def TLI_nvvm_reflect : TargetLibCall<__nvvm_reflect>;

/// double __pow_finite(double x, double y);
def TLI_pow_finite : TargetLibCall<__pow_finite>;

/// float _powf_finite(float x, float y);
def TLI_powf_finite : TargetLibCall<__powf_finite>;

/// long double __powl_finite(long double x, long double y);
def TLI_powl_finite : TargetLibCall<__powl_finite_f128>;

/// double __sincospi_stret(double x);
def TLI_sincospi_stret : TargetLibCall<__sincospi_stret>;

/// float __sincospif_stret(float x);
def TLI_sincospif_stret : TargetLibCall<__sincospif_stret>;

/// double __sinh_finite(double x);
def TLI_sinh_finite : TargetLibCall<__sinh_finite>;

/// float _sinhf_finite(float x);
def TLI_sinhf_finite : TargetLibCall<__sinhf_finite>;

/// long double __sinhl_finite(long double x);
def TLI_sinhl_finite : TargetLibCall<__sinhl_finite>;

/// double __sinpi(double x);
def TLI_sinpi : TargetLibCall<__sinpi>;

/// float __sinpif(float x);
def TLI_sinpif : TargetLibCall<__sinpif>;

/// int __small_fprintf(FILE *stream, const char *format, ...);
def TLI_small_fprintf : TargetLibCall<__small_fprintf>;

/// int __small_printf(const char *format, ...);
def TLI_small_printf : TargetLibCall<__small_printf>;

/// int __small_sprintf(char *str, const char *format, ...);
def TLI_small_sprintf : TargetLibCall<__small_sprintf>;

/// int __snprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, ...);
def TLI_snprintf_chk : TargetLibCall<__snprintf_chk>;

/// int __sprintf_chk(char *str, int flags, size_t str_len,
/// const char *format, ...);
def TLI_sprintf_chk : TargetLibCall<__sprintf_chk>;

/// double __sqrt_finite(double x);
def TLI_sqrt_finite : TargetLibCall<__sqrt_finite>;

/// float __sqrt_finite(float x);
def TLI_sqrtf_finite : TargetLibCall<__sqrtf_finite>;

/// long double __sqrt_finite(long double x);
def TLI_sqrtl_finite : TargetLibCall<__sqrtl_finite>;

/// char *__stpcpy_chk(char *s1, const char *s2, size_t s1size);
def TLI_stpcpy_chk : TargetLibCall<__stpcpy_chk>;

/// char *__stpncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_stpncpy_chk : TargetLibCall<__stpncpy_chk>;

/// char *__strcat_chk(char *s1, const char *s2, size_t s1size);
def TLI_strcat_chk : TargetLibCall<__strcat_chk>; 

/// char *__strcpy_chk(char *s1, const char *s2, size_t s1size);
def TLI_strcpy_chk : TargetLibCall<__strcpy_chk>;

/// char * __strdup(const char *s);
def TLI_dunder_strdup : TargetLibCall<__strdup>;

/// size_t __strlcat_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def TLI_strlcat_chk : TargetLibCall<__strlcat_chk>;

/// size_t __strlcpy_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def TLI_strlcpy_chk : TargetLibCall<__strlcpy_chk>;

/// size_t __strlen_chk(const char *s1, size_t s1size);
def TLI_strlen_chk : TargetLibCall<__strlen_chk>;

/// char *strncat_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_strncat_chk : TargetLibCall<__strncat_chk>;

/// char *__strncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def TLI_strncpy_chk : TargetLibCall<__strncpy_chk>;

/// char *__strndup(const char *s, size_t n);
def TLI_dunder_strndup : TargetLibCall<__strndup>;

/// char * __strtok_r(char *s, const char *delim, char **save_ptr);
def TLI_dunder_strtok_r : TargetLibCall<__strtok_r>;

/// int __vsnprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, va_list ap);
def TLI_vsnprintf_chk : TargetLibCall<__vsnprintf_chk>;

/// int __vsprintf_chk(char *s, int flags, size_t slen, const char *format,
/// va_list ap);
def TLI_vsprintf_chk : TargetLibCall<__vsprintf_chk>;

/// int abs(int j);
def TLI_abs : TargetLibCall<abs>; 

/// int access(const char *path, int amode);
def TLI_access : TargetLibCall<access>;

/// double acos(double x);
def TLI_acos : TargetLibCall<acos>;

/// float acosf(float x);
def TLI_acosf : TargetLibCall<acosf>;

/// double acosh(double x);
def TLI_acosh : TargetLibCall<acosh>;

/// float acoshf(float x);
def TLI_acoshf : TargetLibCall<acoshf>;

/// long double acoshl(long double x);
def TLI_acoshl : TargetLibCall<acoshl>;

/// long double acosl(long double x);
def TLI_acosl : TargetLibCall<acosl_f128>;

/// void *aligned_alloc(size_t alignment, size_t size);
def TLI_aligned_alloc : TargetLibCall<aligned_alloc>;

/// double asin(double x);
def TLI_asin : TargetLibCall<asin>;

/// float asinf(float x);
def TLI_asinf : TargetLibCall<asinf>;

/// double asinh(double x);
def TLI_asinh : TargetLibCall<asinh>;

/// float asinhf(float x);
def TLI_asinhf : TargetLibCall<asinhf>;

/// long double asinhl(long double x);
def TLI_asinhl : TargetLibCall<asinhl>;

/// long double asinl(long double x);
def TLI_asinl : TargetLibCall<asinl_f128>;

/// double atan(double x);
def TLI_atan : TargetLibCall<atan>;

/// double atan2(double y, double x);
def TLI_atan2 : TargetLibCall<atan2>;

/// float atan2f(float y, float x);
def TLI_atan2f : TargetLibCall<atan2f>;

/// long double atan2l(long double y, long double x);
def TLI_atan2l : TargetLibCall<atan2l_f128>;

/// float atanf(float x);
def TLI_atanf : TargetLibCall<atanf>;

/// double atanh(double x);
def TLI_atanh : TargetLibCall<atanh>;


/// float atanhf(float x);
def TLI_atanhf : TargetLibCall<atanhf>;

/// long double atanhl(long double x);
def TLI_atanhl : TargetLibCall<atanhl>;

/// long double atanl(long double x);
def TLI_atanl : TargetLibCall<atanl_f128>;

/// double atof(const char *str);
def TLI_atof : TargetLibCall<atof>;

/// int atoi(const char *str);
def TLI_atoi : TargetLibCall<atoi>;

/// long atol(const char *str);
def TLI_atol : TargetLibCall<atol>;

/// long long atoll(const char *nptr);
def TLI_atoll : TargetLibCall<atoll>;

/// int bcmp(const void *s1, const void *s2, size_t n);
def TLI_bcmp : TargetLibCall<bcmp>;

/// void bcopy(const void *s1, void *s2, size_t n);
def TLI_bcopy : TargetLibCall<bcopy>;

/// void bzero(void *s, size_t n);
def TLI_bzero : TargetLibCall<bzero>;

/// double cabs(double complex z)
def TLI_cabs : TargetLibCall<cabs>;

/// float cabs(float complex z)
def TLI_cabsf : TargetLibCall<cabsf>;

/// long double cabs(long double complex z)
def TLI_cabsl : TargetLibCall<cabsl_f128>;

/// void *calloc(size_t count, size_t size);
def TLI_calloc : TargetLibCall<calloc>;

/// double cbrt(double x);
def TLI_cbrt : TargetLibCall<cbrt>;

/// float cbrtf(float x);
def TLI_cbrtf : TargetLibCall<cbrtf>;

/// long double cbrtl(long double x);
def TLI_cbrtl : TargetLibCall<cbrtl_f128>;

/// double ceil(double x);
def TLI_ceil : TargetLibCall<ceil>;

/// float ceilf(float x);
def TLI_ceilf : TargetLibCall<ceilf>;

/// long double ceill(long double x);
def TLI_ceill : TargetLibCall<ceill_f128>;

/// int chmod(const char *path, mode_t mode);
def TLI_chmod : TargetLibCall<chmod>;

/// int chown(const char *path, uid_t owner, gid_t group);
def TLI_chown : TargetLibCall<chown>;

/// void clearerr(FILE *stream);
def TLI_clearerr : TargetLibCall<clearerr>;

/// int closedir(DIR *dirp);
def TLI_closedir : TargetLibCall<closedir>;

/// double copysign(double x, double y);
def TLI_copysign : TargetLibCall<copysign>;

/// float copysignf(float x, float y);
def TLI_copysignf : TargetLibCall<copysignf>;

/// long double copysignl(long double x, long double y);
def TLI_copysignl : TargetLibCall<copysignl_f128>;

/// double cos(double x);
def TLI_cos : TargetLibCall<cos>;

/// float cosf(float x);
def TLI_cosf : TargetLibCall<cosf>;

/// double cosh(double x);
def TLI_cosh : TargetLibCall<cosh>;

/// float coshf(float x);
def TLI_coshf : TargetLibCall<coshf>;

/// long double coshl(long double x);
def TLI_coshl : TargetLibCall<coshl_f128>;

/// long double cosl(long double x);
def TLI_cosl : TargetLibCall<cosl_f128>;

/// char *ctermid(char *s);
def TLI_ctermid : TargetLibCall<ctermid>;

/// double erf(double x);
def TLI_erf : TargetLibCall<erf>;


/// float erff(float x);
def TLI_erff : TargetLibCall<erff>;

/// long double erfl(long double x);
def TLI_erfl : TargetLibCall<erfl>;

/// double tgamma(double x);
def TLI_tgamma : TargetLibCall<tgamma>;

/// float tgammaf(float x);
def TLI_tgammaf : TargetLibCall<tgammaf>;

/// long double tgammal(long double x);
def TLI_tgammal : TargetLibCall<tgammal_f128>;

/// int execl(const char *path, const char *arg, ...);
def TLI_execl : TargetLibCall<execl>;

/// int execle(const char *file, const char *arg, ..., char * const envp[]);
def TLI_execle : TargetLibCall<execle>;

/// int execlp(const char *file, const char *arg, ...);
def TLI_execlp : TargetLibCall<execlp>;

/// int execv(const char *path, char *const argv[]);
def TLI_execv : TargetLibCall<execv>;

/// int execvP(const char *file, const char *search_path, char *const argv[]);
def TLI_execvP : TargetLibCall<execvP>;

/// int execve(const char *filename, char *const argv], [char *const envp[]);
def TLI_execve : TargetLibCall<execve>;

/// int execvp(const char *file, char *const argv[]);
def TLI_execvp : TargetLibCall<execvp>;

/// int execvpe(const char *file, char *const argv], [char *const envp[]);
def TLI_execvpe : TargetLibCall<execvpe>;

/// double exp(double x);
def TLI_exp : TargetLibCall<exp>;

/// double exp10(double x);
def TLI_exp10 : TargetLibCall<exp10>;

/// float exp10f(float x);
def TLI_exp10f : TargetLibCall<exp10f>;

/// long double exp10l(long double x);
def TLI_exp10l : TargetLibCall<exp10l_f128>;

/// double exp2(double x);
def TLI_exp2 : TargetLibCall<exp2>;

/// float exp2f(float x);
def TLI_exp2f : TargetLibCall<exp2f>;

/// long double exp2l(long double x);
def TLI_exp2l : TargetLibCall<exp2l_f128>;

/// float expf(float x);
def TLI_expf : TargetLibCall<expf>;

/// long double expl(long double x);
def TLI_expl : TargetLibCall<expl_f128>;

/// double expm1(double x);
def TLI_expm1 : TargetLibCall<expm1>;

/// float expm1f(float x);
def TLI_expm1f : TargetLibCall<expm1f>;

/// long double expm1l(long double x);
def TLI_expm1l : TargetLibCall<expm1l>;

/// double fabs(double x);
def TLI_fabs : TargetLibCall<fabs>;

/// float fabsf(float x);
def TLI_fabsf : TargetLibCall<fabsf>;

/// long double fabsl(long double x);
def TLI_fabsl : TargetLibCall<fabsl>;

/// int fclose(FILE *stream);
def TLI_fclose : TargetLibCall<fclose>;

/// FILE *fdopen(int fildes, const char *mode);
def TLI_fdopen : TargetLibCall<fdopen>;

/// int feof(FILE *stream);
def TLI_feof : TargetLibCall<feof>;

/// int ferror(FILE *stream);
def TLI_ferror : TargetLibCall<ferror>;

/// int fflush(FILE *stream);
def TLI_fflush : TargetLibCall<fflush>;

/// int ffs(int i);
def TLI_ffs : TargetLibCall<ffs>;

/// int ffsl(long int i);
def TLI_ffsl : TargetLibCall<ffsl>;

/// int ffsll(long long int i);
def TLI_ffsll : TargetLibCall<ffsll>;

/// int fgetc(FILE *stream);
def TLI_fgetc : TargetLibCall<fgetc>;

/// int fgetc_unlocked(FILE *stream);
def TLI_fgetc_unlocked : TargetLibCall<fgetc_unlocked>;

/// int fgetpos(FILE *stream, fpos_t *pos);
def TLI_fgetpos : TargetLibCall<fgetpos>;

/// char *fgets(char *s, int n, FILE *stream);
def TLI_fgets : TargetLibCall<fgets>;

/// char *fgets_unlocked(char *s, int n, FILE *stream);
def TLI_fgets_unlocked : TargetLibCall<fgets_unlocked>;

/// int fileno(FILE *stream);
def TLI_fileno : TargetLibCall<fileno>;

/// int fiprintf(FILE *stream, const char *format, ...);
def TLI_fiprintf : TargetLibCall<fiprintf>;

/// void flockfile(FILE *file);
def TLI_flockfile : TargetLibCall<flockfile>;

/// double floor(double x);
def TLI_floor : TargetLibCall<floor>;

/// float floorf(float x);
def TLI_floorf : TargetLibCall<floorf>;

/// long double floorl(long double x);
def TLI_floorl : TargetLibCall<floorl_f128>;

/// int fls(int i);
def TLI_fls : TargetLibCall<fls>;

/// int flsl(long int i);
def TLI_flsl : TargetLibCall<flsl>;

/// int flsll(long long int i);
def TLI_flsll : TargetLibCall<flsll>;

// Calls to fmax and fmin library functions expand to the llvm.maxnnum and
// llvm.minnum intrinsics with the correct parameter types for the arguments
// (all types must match).
/// double fmax(double x, double y);
def TLI_fmax : TargetLibCall<fmax>;

/// float fmaxf(float x, float y);
def TLI_fmaxf : TargetLibCall<fmaxf>;

/// long double fmaxl(long double x, long double y);
def TLI_fmaxl : TargetLibCall<fmaxl_f128>;

/// double fmin(double x, double y);
def TLI_fmin : TargetLibCall<fmin>;

/// float fminf(float x, float y);
def TLI_fminf : TargetLibCall<fminf>;

/// long double fminl(long double x, long double y);
def TLI_fminl : TargetLibCall<fminl_f128>;

// Calls to fmaximum_num and fminimum_num library functions expand to the
// llvm.maximumnum and llvm.minimumnum intrinsics with the correct parameter
// types for the arguments (all types must match).
/// double fmaximum_num(double x, double y);
def TLI_fmaximum_num : TargetLibCall<fmaximum_num>;

/// float fmaximum_numf(float x, float y);
def TLI_fmaximum_numf : TargetLibCall<fmaximum_numf>;

/// long double fmaximum_numl(long double x, long double y);
def TLI_fmaximum_numl : TargetLibCall<fmaximum_numl_f128>;

/// double fminimum_num(double x, double y);
def TLI_fminimum_num : TargetLibCall<fminimum_num>;

/// float fminimum_numf(float x, float y);
def TLI_fminimum_numf : TargetLibCall<fminimum_numf>;

/// long double fminimum_numl(long double x, long double y);
def TLI_fminimum_numl : TargetLibCall<fminimum_numl_f128>;

/// double fmod(double x, double y);
def TLI_fmod : TargetLibCall<fmod>;

/// float fmodf(float x, float y);
def TLI_fmodf : TargetLibCall<fmodf>;

/// long double fmodl(long double x, long double y);
def TLI_fmodl : TargetLibCall<fmodl_f128>;

/// FILE *fopen(const char *filename, const char *mode);
def TLI_fopen : TargetLibCall<fopen>;

/// FILE *fopen64(const char *filename, const char *opentype)
def TLI_fopen64 : TargetLibCall<fopen64>;

/// int fork();
def TLI_fork : TargetLibCall<fork>;

/// int fprintf(FILE *stream, const char *format, ...);
def TLI_fprintf : TargetLibCall<fprintf>;

/// int fputc(int c, FILE *stream);
def TLI_fputc : TargetLibCall<fputc>;

/// int fputc_unlocked(int c, FILE *stream);
def TLI_fputc_unlocked : TargetLibCall<fputc_unlocked>;

/// int fputs(const char *s, FILE *stream);
def TLI_fputs : TargetLibCall<fputs>;

/// int fputs_unlocked(const char *s, FILE *stream);
def TLI_fputs_unlocked : TargetLibCall<fputs_unlocked>;

/// size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fread : TargetLibCall<fread>;

/// size_t fread_unlocked(void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fread_unlocked : TargetLibCall<fread_unlocked>;

/// void free(void *ptr);
def TLI_free : TargetLibCall<free>;

/// double frexp(double num, int *exp);
def TLI_frexp : TargetLibCall<frexp>;

/// float frexpf(float num, int *exp);
def TLI_frexpf : TargetLibCall<frexpf>;

/// long double frexpl(long double num, int *exp);
def TLI_frexpl : TargetLibCall<frexpl_f128>;

/// int fscanf(FILE *stream, const char *format, ... );
def TLI_fscanf : TargetLibCall<fscanf>;

/// int fseek(FILE *stream, long offset, int whence);
def TLI_fseek : TargetLibCall<fseek>;

/// int fseeko(FILE *stream, off_t offset, int whence);
def TLI_fseeko : TargetLibCall<fseeko>;

/// int fseeko64(FILE *stream, off64_t offset, int whence)
def TLI_fseeko64 : TargetLibCall<fseeko64>;

/// int fsetpos(FILE *stream, const fpos_t *pos);
def TLI_fsetpos : TargetLibCall<fsetpos>;

/// int fstat(int fildes, struct stat *buf);
def TLI_fstat : TargetLibCall<fstat>;

/// int fstat64(int filedes, struct stat64 *buf)
def TLI_fstat64 : TargetLibCall<fstat64>;

/// int fstatvfs(int fildes, struct statvfs *buf);
def TLI_fstatvfs : TargetLibCall<fstatvfs>;

/// int fstatvfs64(int fildes, struct statvfs64 *buf);
def TLI_fstatvfs64 : TargetLibCall<fstatvfs64>;

/// long ftell(FILE *stream);
def TLI_ftell : TargetLibCall<ftell>;

/// off_t ftello(FILE *stream);
def TLI_ftello : TargetLibCall<ftello>;

/// off64_t ftello64(FILE *stream)
def TLI_ftello64 : TargetLibCall<ftello64>;

/// int ftrylockfile(FILE *file);
def TLI_ftrylockfile : TargetLibCall<ftrylockfile>;

/// void funlockfile(FILE *file);
def TLI_funlockfile : TargetLibCall<funlockfile>;

/// size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
def TLI_fwrite : TargetLibCall<fwrite>;

/// size_t fwrite_unlocked(const void *ptr, size_t size, size_t nitems,
/// FILE *stream);
def TLI_fwrite_unlocked : TargetLibCall<fwrite_unlocked>;

/// int getc(FILE *stream);
def TLI_getc : TargetLibCall<getc>;

/// int getc_unlocked(FILE *stream);
def TLI_getc_unlocked : TargetLibCall<getc_unlocked>;

/// int getchar(void);
def TLI_getchar : TargetLibCall<getchar>;

/// int getchar_unlocked(void);
def TLI_getchar_unlocked : TargetLibCall<getchar_unlocked>;

/// char *getenv(const char *name);
def TLI_getenv : TargetLibCall<getenv>;

/// int getitimer(int which, struct itimerval *value);
def TLI_getitimer : TargetLibCall<getitimer>;

/// int getlogin_r(char *name, size_t namesize);
def TLI_getlogin_r : TargetLibCall<getlogin_r>;

/// struct passwd *getpwnam(const char *name);
def TLI_getpwnam : TargetLibCall<getpwnam>;

/// char *gets(char *s);
def TLI_gets : TargetLibCall<gets>;

/// int gettimeofday(struct timeval *tp, void *tzp);
def TLI_gettimeofday : TargetLibCall<gettimeofday>;

/// uint32_t htonl(uint32_t hostlong);
def TLI_htonl : TargetLibCall<htonl>;

/// uint16_t htons(uint16_t hostshort);
def TLI_htons : TargetLibCall<htons>;

/// double hypot(double x, double y);
def TLI_hypot : TargetLibCall<hypot>;

/// float hypotf(float x, float y);
def TLI_hypotf : TargetLibCall<hypotf>;

/// long double hypotl(long double x, long double y);
def TLI_hypotl : TargetLibCall<hypotl>;

/// int iprintf(const char *format, ...);
def TLI_iprintf : TargetLibCall<iprintf>;

/// int isascii(int c);
def TLI_isascii : TargetLibCall<isascii>;

/// int isdigit(int c);
def TLI_isdigit : TargetLibCall<isdigit>;

/// long int labs(long int j);
def TLI_labs : TargetLibCall<labs>;

/// int lchown(const char *path, uid_t owner, gid_t group);
def TLI_lchown : TargetLibCall<lchown>;

/// double ldexp(double x, int n);
def TLI_ldexp : TargetLibCall<ldexp>;

/// float ldexpf(float x, int n);
def TLI_ldexpf : TargetLibCall<ldexpf>;

/// long double ldexpl(long double x, int n);
def TLI_ldexpl : TargetLibCall<ldexpl_f128>;

/// long long int llabs(long long int j);
def TLI_llabs : TargetLibCall<llabs>;

/// double log(double x);
def TLI_log : TargetLibCall<log>;

/// double log10(double x);
def TLI_log10 : TargetLibCall<log10>;

/// float log10f(float x);
def TLI_log10f : TargetLibCall<log10f>;

/// long double log10l(long double x);
def TLI_log10l : TargetLibCall<log10l_f128>;

/// double log1p(double x);
def TLI_log1p : TargetLibCall<log1p>;

/// float log1pf(float x);
def TLI_log1pf : TargetLibCall<log1pf>;

/// long double log1pl(long double x);
def TLI_log1pl : TargetLibCall<log1pl>;

/// double log2(double x);
def TLI_log2 : TargetLibCall<log2>;

/// float log2f(float x);
def TLI_log2f : TargetLibCall<log2f>;

/// double long double log2l(long double x);
def TLI_log2l : TargetLibCall<log2l_f128>;

/// int ilogb(double x);
def TLI_ilogb : TargetLibCall<ilogb>;

/// int ilogbf(float x);
def TLI_ilogbf : TargetLibCall<ilogbf>;

/// int ilogbl(long double x);
def TLI_ilogbl : TargetLibCall<ilogbl>;

/// double logb(double x);
def TLI_logb : TargetLibCall<logb>;

/// float logbf(float x);
def TLI_logbf : TargetLibCall<logbf>;

/// long double logbl(long double x);
def TLI_logbl : TargetLibCall<logbl>;

/// float logf(float x);
def TLI_logf : TargetLibCall<logf>;

/// long double logl(long double x);
def TLI_logl : TargetLibCall<logl_f128>;

/// int lstat(const char *path, struct stat *buf);
def TLI_lstat : TargetLibCall<lstat>;

/// int lstat64(const char *path, struct stat64 *buf);
def TLI_lstat64 : TargetLibCall<lstat64>;

/// void *malloc(size_t size);
def TLI_malloc : TargetLibCall<malloc>;

/// void *memalign(size_t boundary, size_t size);
def TLI_memalign : TargetLibCall<memalign>;

/// void *memccpy(void *s1, const void *s2, int c, size_t n);
def TLI_memccpy : TargetLibCall<memccpy>;

/// void *memchr(const void *s, int c, size_t n);
def TLI_memchr : TargetLibCall<memchr>;

/// int memcmp(const void *s1, const void *s2, size_t n);
def TLI_memcmp : TargetLibCall<memcmp>;

/// void *memcpy(void *s1, const void *s2, size_t n);
def TLI_memcpy : TargetLibCall<memcpy>;

/// void *memmove(void *s1, const void *s2, size_t n);
def TLI_memmove : TargetLibCall<memmove>;

/// void *mempcpy(void *s1, const void *s2, size_t n);
def TLI_mempcpy : TargetLibCall<mempcpy>;

/// void *memrchr(const void *s, int c, size_t n);
def TLI_memrchr : TargetLibCall<memrchr>;

/// void *memset(void *b, int c, size_t len);
def TLI_memset : TargetLibCall<memset>;

/// void memset_pattern16(void *b, const void *pattern16, size_t len);
def TLI_memset_pattern16 : TargetLibCall<memset_pattern16>;

/// void memset_pattern4(void *b, const void *pattern4, size_t len);
def TLI_memset_pattern4 : TargetLibCall<memset_pattern4>;

/// void memset_pattern8(void *b, const void *pattern8, size_t len);
def TLI_memset_pattern8 : TargetLibCall<memset_pattern8>;

/// int mkdir(const char *path, mode_t mode);
def TLI_mkdir : TargetLibCall<mkdir>;

/// time_t mktime(struct tm *timeptr);
def TLI_mktime : TargetLibCall<mktime>;

/// double modf(double x, double *iptr);
def TLI_modf : TargetLibCall<modf>;

/// float modff(float, float *iptr);
def TLI_modff : TargetLibCall<modff>;

/// long double modfl(long double value, long double *iptr);
def TLI_modfl : TargetLibCall<modfl_f128>;

/// double nan(const char *arg);
def TLI_nan : TargetLibCall<nan>;

/// float nanf(const char *arg);
def TLI_nanf : TargetLibCall<nanf>;

/// long double nanl(const char *arg);
def TLI_nanl : TargetLibCall<nanl_f128>;

/// double nearbyint(double x);
def TLI_nearbyint : TargetLibCall<nearbyint>;

/// float nearbyintf(float x);
def TLI_nearbyintf : TargetLibCall<nearbyintf>;

/// long double nearbyintl(long double x);
def TLI_nearbyintl : TargetLibCall<nearbyintl_f128>;

/// double nextafter(double x, double y);
def TLI_nextafter : TargetLibCall<nextafter>;

/// float nextafterf(float x, float y);
def TLI_nextafterf : TargetLibCall<nextafterf>;

/// long double nextafterl(long double x, long double y);
def TLI_nextafterl : TargetLibCall<nextafterl>;

/// double nexttoward(double x, long double y);
def TLI_nexttoward : TargetLibCall<nexttoward>;

/// float nexttowardf(float x, long double y);
def TLI_nexttowardf : TargetLibCall<nexttowardf>;

/// long double nexttowardl(long double x, long double y);
def TLI_nexttowardl : TargetLibCall<nexttowardl_f128>;

/// uint32_t ntohl(uint32_t netlong);
def TLI_ntohl : TargetLibCall<ntohl>;

/// uint16_t ntohs(uint16_t netshort);
def TLI_ntohs : TargetLibCall<ntohs>;

/// int open(const char *path, int oflag, ... );
def TLI_open : TargetLibCall<open>;

/// int open64(const char *filename, int flags, [mode_t mode])
def TLI_open64 : TargetLibCall<open64>;

/// DIR *opendir(const char *dirname);
def TLI_opendir : TargetLibCall<opendir>;

/// int pclose(FILE *stream);
def TLI_pclose : TargetLibCall<pclose>;

/// void perror(const char *s);
def TLI_perror : TargetLibCall<perror>;

/// FILE *popen(const char *command, const char *mode);
def TLI_popen : TargetLibCall<popen>;

/// int posix_memalign(void **memptr, size_t alignment, size_t size);
def TLI_posix_memalign : TargetLibCall<posix_memalign>;

/// double pow(double x, double y);
def TLI_pow : TargetLibCall<pow>;

/// float powf(float x, float y);
def TLI_powf : TargetLibCall<powf>;

/// long double powl(long double x, long double y);
def TLI_powl : TargetLibCall<powl_f128>;

/// ssize_t pread(int fildes, void *buf, size_t nbyte, off_t offset);
def TLI_pread : TargetLibCall<pread>;

/// int printf(const char *format, ...);
def TLI_printf : TargetLibCall<printf>;

/// int putc(int c, FILE *stream);
def TLI_putc : TargetLibCall<putc>;

/// int putc_unlocked(int c, FILE *stream);
def TLI_putc_unlocked : TargetLibCall<putc_unlocked>;

/// int putchar(int c);
def TLI_putchar : TargetLibCall<putchar>;

/// int putchar_unlocked(int c);
def TLI_putchar_unlocked : TargetLibCall<putchar_unlocked>;

/// int puts(const char *s);
def TLI_puts : TargetLibCall<puts>;

/// void *pvalloc(size_t size);
def TLI_pvalloc : TargetLibCall<pvalloc>;

/// ssize_t pwrite(int fildes, const void *buf, size_t nbyte, off_t offset);
def TLI_pwrite : TargetLibCall<pwrite>;

/// void qsort(void *base, size_t nel, size_t width,
///            int (*compar)(const void *, const void *));
def TLI_qsort : TargetLibCall<qsort>;

/// ssize_t read(int fildes, void *buf, size_t nbyte);
def TLI_read : TargetLibCall<read>;

/// ssize_t readlink(const char *path, char *buf, size_t bufsize);
def TLI_readlink : TargetLibCall<readlink>;

/// void *realloc(void *ptr, size_t size);
def TLI_realloc : TargetLibCall<realloc>;

/// void *reallocf(void *ptr, size_t size);
def TLI_reallocf : TargetLibCall<reallocf>;

/// void *reallocarray(void *ptr, size_t nmemb, size_t size);
def TLI_reallocarray : TargetLibCall<reallocarray>;

/// char *realpath(const char *file_name, char *resolved_name);
def TLI_realpath : TargetLibCall<realpath>;

/// double remainder(double x, double y);
def TLI_remainder : TargetLibCall<remainder>;

/// float remainderf(float x, float y);
def TLI_remainderf : TargetLibCall<remainderf>;

/// long double remainderl(long double x, long double y);
def TLI_remainderl : TargetLibCall<remainderl_f128>;

/// double remquo(double x, double y, int *quo);
def TLI_remquo : TargetLibCall<remquo>;

/// float remquof(float x, float y, int *quo);
def TLI_remquof : TargetLibCall<remquof>;

/// long double remquol(long double x, long double y, int *quo);
def TLI_remquol : TargetLibCall<remquol_f128>;

/// double fdim(double x, double y);
def TLI_fdim : TargetLibCall<fdim>;

/// float fdimf(float x, float y);
def TLI_fdimf : TargetLibCall<fdimf>;

/// long double fdiml(long double x, long double y);
def TLI_fdiml : TargetLibCall<fdiml_f128>;

/// int remove(const char *path);
def TLI_remove : TargetLibCall<remove>;

/// int rename(const char *old, const char *new);
def TLI_rename : TargetLibCall<rename>;

/// void rewind(FILE *stream);
def TLI_rewind : TargetLibCall<rewind>;

/// double rint(double x);
def TLI_rint : TargetLibCall<rint>; 

/// float rintf(float x);
def TLI_rintf : TargetLibCall<rintf>;

/// long double rintl(long double x);
def TLI_rintl : TargetLibCall<rintl_f128>;

/// int rmdir(const char *path);
def TLI_rmdir : TargetLibCall<rmdir>;

/// double round(double x);
def TLI_round : TargetLibCall<round>;

/// double roundeven(double x);
def TLI_roundeven : TargetLibCall<roundeven>;

/// float roundevenf(float x);
def TLI_roundevenf : TargetLibCall<roundevenf>;

/// long double roundevenl(long double x);
def TLI_roundevenl : TargetLibCall<roundevenl_f128>;

/// float roundf(float x);
def TLI_roundf : TargetLibCall<roundf>;

/// long double roundl(long double x);
def TLI_roundl : TargetLibCall<roundl_f128>;

/// double scalbln(double arg, long exp);
def TLI_scalbln : TargetLibCall<scalbln>;

/// float scalblnf(float arg, long exp);
def TLI_scalblnf : TargetLibCall<scalblnf>;

/// long double scalblnl(long double arg, long exp);
def TLI_scalblnl : TargetLibCall<scalblnl_f128>;

/// double scalbn(double arg, int exp);
def TLI_scalbn : TargetLibCall<scalbn>;

/// float scalbnf(float arg, int exp);
def TLI_scalbnf : TargetLibCall<scalbnf>;

/// long double scalbnl(long double arg, int exp);
def TLI_scalbnl : TargetLibCall<scalbnl_f128>;

/// int scanf(const char *restrict format, ... );
def TLI_scanf : TargetLibCall<scanf>;

/// void setbuf(FILE *stream, char *buf);
def TLI_setbuf : TargetLibCall<setbuf>;

/// int setitimer(int which, const struct itimerval *value,
/// struct itimerval *ovalue);
def TLI_setitimer : TargetLibCall<setitimer>;

/// int setvbuf(FILE *stream, char *buf, int type, size_t size);
def TLI_setvbuf : TargetLibCall<setvbuf>;

/// double sin(double x);
def TLI_sin : TargetLibCall<sin>;

/// float sinf(float x);
def TLI_sinf : TargetLibCall<sinf>;

/// double sinh(double x);
def TLI_sinh : TargetLibCall<sinh>;

/// float sinhf(float x);
def TLI_sinhf : TargetLibCall<sinhf>;

/// long double sinhl(long double x);
def TLI_sinhl : TargetLibCall<sinhl_f128>;

/// long double sinl(long double x);
def TLI_sinl : TargetLibCall<sinl_f128>;

/// void sincos(double x, double *sin_out, double *cos_out);
def TLI_sincos : TargetLibCall<sincos>;

/// void sincosf(float x, float *sin_out, float *cos_out);
def TLI_sincosf : TargetLibCall<sincosf>;

/// void sincosl(long double x, long double *sin_out, long double *cos_out);
def TLI_sincosl : TargetLibCall<sincosl_f128>;

/// int siprintf(char *str, const char *format, ...);
def TLI_siprintf : TargetLibCall<siprintf>;

/// int snprintf(char *s, size_t n, const char *format, ...);
def TLI_snprintf : TargetLibCall<snprintf>;

/// int sprintf(char *str, const char *format, ...);
def TLI_sprintf : TargetLibCall<sprintf>;

/// double sqrt(double x);
def TLI_sqrt : TargetLibCall<sqrt>;

/// float sqrtf(float x);
def TLI_sqrtf : TargetLibCall<sqrtf>;

/// long double sqrtl(long double x);
def TLI_sqrtl : TargetLibCall<sqrtl_f128>;

/// int sscanf(const char *s, const char *format, ... );
def TLI_sscanf : TargetLibCall<sscanf>;

/// int stat(const char *path, struct stat *buf);
def TLI_stat : TargetLibCall<stat>;

/// int stat64(const char *path, struct stat64 *buf);
def TLI_stat64 : TargetLibCall<stat64>;

/// int statvfs(const char *path, struct statvfs *buf);
def TLI_statvfs : TargetLibCall<statvfs>;

/// int statvfs64(const char *path, struct statvfs64 *buf)
def TLI_statvfs64 : TargetLibCall<statvfs64>;

/// char *stpcpy(char *s1, const char *s2);
def TLI_stpcpy : TargetLibCall<stpcpy>;

/// char *stpncpy(char *s1, const char *s2, size_t n);
def TLI_stpncpy : TargetLibCall<stpncpy>;

/// int strcasecmp(const char *s1, const char *s2);
def TLI_strcasecmp : TargetLibCall<strcasecmp>;

/// char *strcat(char *s1, const char *s2);
def TLI_strcat : TargetLibCall<strcat>;

/// char *strchr(const char *s, int c);
def TLI_strchr : TargetLibCall<strchr>;

/// int strcmp(const char *s1, const char *s2);
def TLI_strcmp : TargetLibCall<strcmp>;

/// int strcoll(const char *s1, const char *s2);
def TLI_strcoll : TargetLibCall<strcoll>;

/// char *strcpy(char *s1, const char *s2);
def TLI_strcpy : TargetLibCall<strcpy>;

/// size_t strcspn(const char *s1, const char *s2);
def TLI_strcspn : TargetLibCall<strcspn>;

/// char *strdup(const char *s1);
def TLI_strdup : TargetLibCall<strdup>;

/// size_t strlcat(char *dst, const char *src, size_t size);
def TLI_strlcat : TargetLibCall<strlcat>;

/// size_t strlcpy(char *dst, const char *src, size_t size);
def TLI_strlcpy : TargetLibCall<strlcpy>;

/// size_t strlen(const char *s);
def TLI_strlen : TargetLibCall<strlen>;

/// int strncasecmp(const char *s1, const char *s2, size_t n);
def TLI_strncasecmp : TargetLibCall<strncasecmp>;

/// char *strncat(char *s1, const char *s2, size_t n);
def TLI_strncat : TargetLibCall<strncat>;

/// int strncmp(const char *s1, const char *s2, size_t n);
def TLI_strncmp : TargetLibCall<strncmp>;

/// char *strncpy(char *s1, const char *s2, size_t n);
def TLI_strncpy : TargetLibCall<strncpy>;

/// char *strndup(const char *s1, size_t n);
def TLI_strndup : TargetLibCall<strndup>;

/// size_t strnlen(const char *s, size_t maxlen);
def TLI_strnlen : TargetLibCall<strnlen>;

/// char *strpbrk(const char *s1, const char *s2);
def TLI_strpbrk : TargetLibCall<strpbrk>;

/// char *strrchr(const char *s, int c);
def TLI_strrchr : TargetLibCall<strrchr>;

/// size_t strspn(const char *s1, const char *s2);
def TLI_strspn : TargetLibCall<strspn>;

/// char *strstr(const char *s1, const char *s2);
def TLI_strstr : TargetLibCall<strstr>;

/// double strtod(const char *nptr, char **endptr);
def TLI_strtod : TargetLibCall<strtod>;

/// float strtof(const char *nptr, char **endptr);
def TLI_strtof : TargetLibCall<strtof>;

/// char *strtok(char *s1, const char *s2);
def TLI_strtok : TargetLibCall<strtok>;

/// char *strtok_r(char *s, const char *sep, char **lasts);
def TLI_strtok_r : TargetLibCall<strtok_r>;

/// long int strtol(const char *nptr, char **endptr, int base);
def TLI_strtol : TargetLibCall<strtol>;

/// long double strtold(const char *nptr, char **endptr);
def TLI_strtold : TargetLibCall<strtold>;

/// long long int strtoll(const char *nptr, char **endptr, int base);
def TLI_strtoll : TargetLibCall<strtoll>;

/// unsigned long int strtoul(const char *nptr, char **endptr, int base);
def TLI_strtoul : TargetLibCall<strtoul>;

/// unsigned long long int strtoull(const char *nptr, char **endptr, int base);
def TLI_strtoull : TargetLibCall<strtoull>;

/// size_t strxfrm(char *s1, const char *s2, size_t n);
def TLI_strxfrm : TargetLibCall<strxfrm>;

/// int system(const char *command);
def TLI_system : TargetLibCall<system>;

/// double tan(double x);
def TLI_tan : TargetLibCall<tan>;

/// float tanf(float x);
def TLI_tanf : TargetLibCall<tanf>;

/// double tanh(double x);
def TLI_tanh : TargetLibCall<tanh>;

/// float tanhf(float x);
def TLI_tanhf : TargetLibCall<tanhf>;

/// long double tanhl(long double x);
def TLI_tanhl : TargetLibCall<tanhl_f128>;

/// long double tanl(long double x);
def TLI_tanl : TargetLibCall<tanl_f128>;

/// clock_t times(struct tms *buffer);
def TLI_times : TargetLibCall<times>;

/// FILE *tmpfile(void);
def TLI_tmpfile : TargetLibCall<tmpfile>;

/// FILE *tmpfile64(void)
def TLI_tmpfile64 : TargetLibCall<tmpfile64>;

/// int toascii(int c);
def TLI_toascii : TargetLibCall<toascii>;

/// double trunc(double x);
def TLI_trunc : TargetLibCall<trunc>;

/// float truncf(float x);
def TLI_truncf : TargetLibCall<truncf>;

/// long double truncl(long double x);
def TLI_truncl : TargetLibCall<truncl_f128>;

/// int uname(struct utsname *name);
def TLI_uname : TargetLibCall<uname>;

/// int ungetc(int c, FILE *stream);
def TLI_ungetc : TargetLibCall<ungetc>;

/// int unlink(const char *path);
def TLI_unlink : TargetLibCall<unlink>;

/// int unsetenv(const char *name);
def TLI_unsetenv : TargetLibCall<unsetenv>;

/// int utime(const char *path, const struct utimbuf *times);
def TLI_utime : TargetLibCall<utime>;

/// int utimes(const char *path, const struct timeval times[2]);
def TLI_utimes : TargetLibCall<utimes>;

/// void *valloc(size_t size);
def TLI_valloc : TargetLibCall<valloc>;

/// void *vec_calloc(size_t count, size_t size);
def TLI_vec_calloc : TargetLibCall<vec_calloc>;

/// void vec_free(void *ptr);
def TLI_vec_free : TargetLibCall<vec_free>;

/// void *vec_malloc(size_t size);
def TLI_vec_malloc : TargetLibCall<vec_malloc>;

/// void *vec_realloc(void *ptr, size_t size);
def TLI_vec_realloc : TargetLibCall<vec_realloc>;

/// int vfprintf(FILE *stream, const char *format, va_list ap);
def TLI_vfprintf : TargetLibCall<vfprintf>;

/// int vfscanf(FILE *stream, const char *format, va_list arg);
def TLI_vfscanf : TargetLibCall<vfscanf>;

/// int vprintf(const char *restrict format, va_list ap);
def TLI_vprintf : TargetLibCall<vprintf>;

/// int vscanf(const char *format, va_list arg);
def TLI_vscanf : TargetLibCall<vscanf>;

/// int vsnprintf(char *s, size_t n, const char *format, va_list ap);
def TLI_vsnprintf : TargetLibCall<vsnprintf>;

/// int vsprintf(char *s, const char *format, va_list ap);
def TLI_vsprintf : TargetLibCall<vsprintf>;

/// int vsscanf(const char *s, const char *format, va_list arg);
def TLI_vsscanf : TargetLibCall<vsscanf>;

/// size_t wcslen (const wchar_t* wcs);
def TLI_wcslen : TargetLibCall<wcslen>;

/// ssize_t write(int fildes, const void *buf, size_t nbyte);
def TLI_write : TargetLibCall<write>;
