//===-- TargetLibraryInfo.td - File that describes library functions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Analysis/TargetLibraryInfoImpl.td"

/// void *operator new(unsigned int);
def msvc_new_int : TargetLibCall<"??2@YAPAXI@Z", Ptr, [Int]>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def msvc_new_int_nothrow
    : TargetLibCall<"??2@YAPAXIABUnothrow_t@std@@@Z", Ptr, [Int, Ptr]>;

/// void *operator new(unsigned long long);
def msvc_new_longlong : TargetLibCall<"??2@YAPEAX_K@Z", Ptr, [LLong]>;

/// void *operator new(unsigned long long, const std::nothrow_t&);
def msvc_new_longlong_nothrow
    : TargetLibCall<"??2@YAPEAX_KAEBUnothrow_t@std@@@Z", Ptr, [LLong, Ptr]>;

/// void operator delete(void*);
def msvc_delete_ptr32 : TargetLibCall<"??3@YAXPAX@Z", Void, [Ptr]>;

/// void operator delete(void*, const std::nothrow_t&);
def msvc_delete_ptr32_nothrow
    : TargetLibCall<"??3@YAXPAXABUnothrow_t@std@@@Z", Void, [Ptr, Ptr]>;

/// void operator delete(void*, unsigned int);
def msvc_delete_ptr32_int : TargetLibCall<"??3@YAXPAXI@Z", Void, [Ptr, Int]>;

/// void operator delete(void*);
def msvc_delete_ptr64 : TargetLibCall<"??3@YAXPEAX@Z", Void, [Ptr]>;

/// void operator delete(void*, const std::nothrow_t&);
def msvc_delete_ptr64_nothrow
    : TargetLibCall<"??3@YAXPEAXAEBUnothrow_t@std@@@Z", Void, [Ptr, Ptr]>;

/// void operator delete(void*, unsigned long long);
def msvc_delete_ptr64_longlong
    : TargetLibCall<"??3@YAXPEAX_K@Z", Void, [Ptr, LLong]>;

/// void *operator new[](unsigned int);
def msvc_new_array_int : TargetLibCall<"??_U@YAPAXI@Z", Ptr, [Int]>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def msvc_new_array_int_nothrow
    : TargetLibCall<"??_U@YAPAXIABUnothrow_t@std@@@Z", Ptr, [Int, Ptr]>;

/// void *operator new[](unsigned long long);
def msvc_new_array_longlong : TargetLibCall<"??_U@YAPEAX_K@Z", Ptr, [LLong]>;

/// void *operator new](unsigned long long, [const std::nothrow_t&);
def msvc_new_array_longlong_nothrow
    : TargetLibCall<"??_U@YAPEAX_KAEBUnothrow_t@std@@@Z", Ptr, [LLong, Ptr]>;

/// void operator delete[](void*);
def msvc_delete_array_ptr32 : TargetLibCall<"??_V@YAXPAX@Z", Void, [Ptr]>;

/// void operator delete](void*, [const std::nothrow_t&);
def msvc_delete_array_ptr32_nothrow
    : TargetLibCall<"??_V@YAXPAXABUnothrow_t@std@@@Z", Void, [Ptr, Ptr]>;

/// void operator delete](void*, [unsigned int);
def msvc_delete_array_ptr32_int
    : TargetLibCall<"??_V@YAXPAXI@Z", Void, [Ptr, Int]>;

/// void operator delete[](void*);
def msvc_delete_array_ptr64 : TargetLibCall<"??_V@YAXPEAX@Z", Void, [Ptr]>;

/// void operator delete](void*, [const std::nothrow_t&);
def msvc_delete_array_ptr64_nothrow
    : TargetLibCall<"??_V@YAXPEAXAEBUnothrow_t@std@@@Z", Void, [Ptr, Ptr]>;

/// void operator delete](void*, [unsigned long long);
def msvc_delete_array_ptr64_longlong
    : TargetLibCall<"??_V@YAXPEAX_K@Z", Void, [Ptr, LLong]>;

/// int _IO_getc(_IO_FILE * __fp);
def under_IO_getc : TargetLibCall<"_IO_getc", Int, [Ptr]>;

/// int _IO_putc(int __c, _IO_FILE * __fp);
def under_IO_putc : TargetLibCall<"_IO_putc", Int, [Int, Ptr]>;

/// void operator delete[](void*);
def ZdaPv : TargetLibCall<"_ZdaPv", Void, [Ptr]>;

/// void operator delete](void*, [const std::nothrow_t&);
def ZdaPvRKSt9nothrow_t
    : TargetLibCall<"_ZdaPvRKSt9nothrow_t", Void, [Ptr, Ptr]>;

/// void operator delete](void*, [std::align_val_t);
def ZdaPvSt11align_val_t
    : TargetLibCall<"_ZdaPvSt11align_val_t", Void, [Ptr, IntPlus]>;

/// void operator delete](void*, [std::align_val_t, const std::nothrow_t&)
def ZdaPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZdaPvSt11align_val_tRKSt9nothrow_t",
                    Void, [Ptr, IntPlus, Ptr]>;

/// void operator delete](void*, [unsigned int);
def ZdaPvj : TargetLibCall<"_ZdaPvj", Void, [Ptr, Int]>;

/// void operator delete](void*, [unsigned int, std::align_val_t);
def ZdaPvjSt11align_val_t
    : TargetLibCall<"_ZdaPvjSt11align_val_t", Void, [Ptr, Int, Int]>;

/// void operator delete](void*, [unsigned long);
def ZdaPvm : TargetLibCall<"_ZdaPvm", Void, [Ptr, Long]>;

/// void operator delete](void*, [unsigned long, std::align_val_t);
def ZdaPvmSt11align_val_t
    : TargetLibCall<"_ZdaPvmSt11align_val_t", Void, [Ptr, Long, Long]>;

/// void operator delete(void*);
def ZdlPv : TargetLibCall<"_ZdlPv", Void, [Ptr]>;

/// void operator delete(void*, const std::nothrow_t&);
def ZdlPvRKSt9nothrow_t
    : TargetLibCall<"_ZdlPvRKSt9nothrow_t", Void, [Ptr, Ptr]>;

/// void operator delete(void*, std::align_val_t)
def ZdlPvSt11align_val_t
    : TargetLibCall<"_ZdlPvSt11align_val_t", Void, [Ptr, IntPlus]>;

/// void operator delete(void*, std::align_val_t, const std::nothrow_t&)
def ZdlPvSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZdlPvSt11align_val_tRKSt9nothrow_t",
                    Void, [Ptr, IntPlus, Ptr]>;

/// void operator delete(void*, unsigned int);
def ZdlPvj : TargetLibCall<"_ZdlPvj", Void, [Ptr, Int]>;

/// void operator delete(void*, unsigned int, std::align_val_t)
def ZdlPvjSt11align_val_t
    : TargetLibCall<"_ZdlPvjSt11align_val_t", Void, [Ptr, Int, Int]>;

/// void operator delete(void*, unsigned long);
def ZdlPvm : TargetLibCall<"_ZdlPvm", Void, [Ptr, Long]>;

/// void operator delete(void*, unsigned long, std::align_val_t)
def ZdlPvmSt11align_val_t
    : TargetLibCall<"_ZdlPvmSt11align_val_t", Void, [Ptr, Long, Long]>;

/// void *operator new[](unsigned int);
def Znaj : TargetLibCall<"_Znaj", Ptr, [Int]>;

/// void *operator new](unsigned int, [const std::nothrow_t&);
def ZnajRKSt9nothrow_t : TargetLibCall<"_ZnajRKSt9nothrow_t", Ptr, [Int, Ptr]>;

/// void *operator new](unsigned int, [std::align_val_t)
def ZnajSt11align_val_t
    : TargetLibCall<"_ZnajSt11align_val_t", Ptr, [Int, Int]>;

/// void *operator new](unsigned int, [std::align_val_t, const std::nothrow_t&)
def ZnajSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZnajSt11align_val_tRKSt9nothrow_t", Ptr, [Int, Int, Ptr]>;

/// void *operator new[](unsigned long);
def Znam : TargetLibCall<"_Znam", Ptr, [Long]>;

/// void *operator new](unsigned long, [__hot_cold_t)
/// Currently this and other operator new interfaces that take a __hot_cold_t
/// hint are supported by the open source version of tcmalloc, see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/new_extension.h
/// and for the definition of the __hot_cold_t parameter see:
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h
def Znam12__hot_cold_t
    : TargetLibCall<"_Znam12__hot_cold_t", Ptr, [Long, Bool]>;

/// void *operator new](unsigned long, [const std::nothrow_t&);
def ZnamRKSt9nothrow_t : TargetLibCall<"_ZnamRKSt9nothrow_t", Ptr, [Long, Ptr]>;

/// void *operator new](unsigned long, [const std::nothrow_t&, __hot_cold_t)
def ZnamRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<"_ZnamRKSt9nothrow_t12__hot_cold_t",
                    Ptr, [Long, Ptr, Bool]>;

/// void *operator new](unsigned long, [std::align_val_t)
def ZnamSt11align_val_t
    : TargetLibCall<"_ZnamSt11align_val_t", Ptr, [Long, Long]>;

/// void *operator new](unsigned long, [std::align_val_t, __hot_cold_t)
def ZnamSt11align_val_t12__hot_cold_t
    : TargetLibCall<"_ZnamSt11align_val_t12__hot_cold_t",
                    Ptr, [Long, Long, Bool]>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&)
def ZnamSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZnamSt11align_val_tRKSt9nothrow_t",
                    Ptr, [Long, Long, Ptr]>;

/// void *operator new](unsigned long, [std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<"_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t",
                    Ptr, [Long, Long, Ptr, Bool]>;

/// void *operator new(unsigned int);
def Znwj : TargetLibCall<"_Znwj", Ptr, [Int]>;

/// void *operator new(unsigned int, const std::nothrow_t&);
def ZnwjRKSt9nothrow_t : TargetLibCall<"_ZnwjRKSt9nothrow_t", Ptr, [Int, Ptr]>;

/// void *operator new(unsigned int, std::align_val_t)
def ZnwjSt11align_val_t
    : TargetLibCall<"_ZnwjSt11align_val_t", Ptr, [Int, Int]>;

/// void *operator new(unsigned int, std::align_val_t, const std::nothrow_t&)
def ZnwjSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZnwjSt11align_val_tRKSt9nothrow_t", Ptr, [Int, Int, Ptr]>;

/// void *operator new(unsigned long);
def Znwm : TargetLibCall<"_Znwm", Ptr, [Long]>;

/// void *operator new(unsigned long, __hot_cold_t)
def Znwm12__hot_cold_t
    : TargetLibCall<"_Znwm12__hot_cold_t", Ptr, [Long, Bool]>;

/// void *operator new(unsigned long, const std::nothrow_t&);
def ZnwmRKSt9nothrow_t : TargetLibCall<"_ZnwmRKSt9nothrow_t", Ptr, [Long, Ptr]>;

/// void *operator new(unsigned long, const std::nothrow_t&, __hot_cold_t)
def ZnwmRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<"_ZnwmRKSt9nothrow_t12__hot_cold_t",
                    Ptr, [Long, Ptr, Bool]>;

/// void *operator new(unsigned long, std::align_val_t)
def ZnwmSt11align_val_t
    : TargetLibCall<"_ZnwmSt11align_val_t", Ptr, [Long, Long]>;

/// void *operator new(unsigned long, std::align_val_t, __hot_cold_t)
def ZnwmSt11align_val_t12__hot_cold_t
    : TargetLibCall<"_ZnwmSt11align_val_t12__hot_cold_t",
                    Ptr, [Long, Long, Bool]>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&)
def ZnwmSt11align_val_tRKSt9nothrow_t
    : TargetLibCall<"_ZnwmSt11align_val_tRKSt9nothrow_t",
                    Ptr, [Long, Long, Ptr]>;

/// void *operator new(unsigned long, std::align_val_t, const std::nothrow_t&,
/// __hot_cold_t)
def ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : TargetLibCall<"_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t",
                    Ptr, [Long, Long, Ptr, Bool]>;

/// The following are variants of operator new which return the actual size
/// reserved by the allocator proposed in P0901R5 (Size feedback in operator
/// new). https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0901r5.html
/// They are implemented by tcmalloc, see source at
/// https://github.com/google/tcmalloc/blob/master/tcmalloc/malloc_extension.h

/// __sized_ptr_t __size_returning_new(size_t size)
def size_returning_new
    : TargetLibCall<"__size_returning_new", ? /* Checked manually. */>;

/// __sized_ptr_t __size_returning_new_hot_cold(size_t, __hot_cold_t)
def size_returning_new_hot_cold
    : TargetLibCall<"__size_returning_new_hot_cold", ? /* Checked manually. */>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t)
def size_returning_new_aligned
    : TargetLibCall<"__size_returning_new_aligned", ? /* Checked manually. */>;

/// __sized_ptr_t __size_returning_new_aligned(size_t, std::align_val_t,
/// __hot_cold_t)
def size_returning_new_aligned_hot_cold
    : TargetLibCall<"__size_returning_new_aligned_hot_cold",
                    ? /* Checked manually. */>;

/// double __acos_finite(double x);
def acos_finite : TargetLibCall<"__acos_finite", Dbl, [Dbl]>;

/// float __acosf_finite(float x);
def acosf_finite : TargetLibCall<"__acosf_finite", Flt, [Flt]>;

/// double __acosh_finite(double x);
def acosh_finite : TargetLibCall<"__acosh_finite", Dbl, [Dbl]>;

/// float __acoshf_finite(float x);
def acoshf_finite : TargetLibCall<"__acoshf_finite", Flt, [Flt]>;

/// long double __acoshl_finite(long double x);
def acoshl_finite : TargetLibCall<"__acoshl_finite", LDbl, [LDbl]>;

/// long double __acosl_finite(long double x);
def acosl_finite : TargetLibCall<"__acosl_finite", LDbl, [LDbl]>;

/// double __asin_finite(double x);
def asin_finite : TargetLibCall<"__asin_finite", Dbl, [Dbl]>;

/// float __asinf_finite(float x);
def asinf_finite : TargetLibCall<"__asinf_finite", Flt, [Flt]>;

/// long double __asinl_finite(long double x);
def asinl_finite : TargetLibCall<"__asinl_finite", LDbl, [LDbl]>;

/// double atan2_finite(double y, double x);
def atan2_finite : TargetLibCall<"__atan2_finite", Dbl, [Dbl, Dbl]>;

/// float atan2f_finite(float y, float x);
def atan2f_finite : TargetLibCall<"__atan2f_finite", Flt, [Flt, Flt]>;

/// long double atan2l_finite(long double y, long double x);
def atan2l_finite : TargetLibCall<"__atan2l_finite", LDbl, [LDbl, LDbl]>;

/// double __atanh_finite(double x);
def atanh_finite : TargetLibCall<"__atanh_finite", Dbl, [Dbl]>;

/// float __atanhf_finite(float x);
def atanhf_finite : TargetLibCall<"__atanhf_finite", Flt, [Flt]>;

/// long double __atanhl_finite(long double x);
def atanhl_finite : TargetLibCall<"__atanhl_finite", LDbl, [LDbl]>;

/// void __atomic_load(size_t size, void *mptr, void *vptr, int smodel);
def atomic_load : TargetLibCall<"__atomic_load", Void, [SizeT, Ptr, Ptr, Int]>;

/// void __atomic_store(size_t size, void *mptr, void *vptr, int smodel);
def atomic_store
    : TargetLibCall<"__atomic_store", Void, [SizeT, Ptr, Ptr, Int]>;

/// double __cosh_finite(double x);
def cosh_finite : TargetLibCall<"__cosh_finite", Dbl, [Dbl]>;

/// float __coshf_finite(float x);
def coshf_finite : TargetLibCall<"__coshf_finite", Flt, [Flt]>;

/// long double __coshl_finite(long double x);
def coshl_finite : TargetLibCall<"__coshl_finite", LDbl, [LDbl]>;

/// double __cospi(double x);
def cospi : TargetLibCall<"__cospi", Dbl, [Dbl]>;

/// float __cospif(float x);
def cospif : TargetLibCall<"__cospif", Flt, [Flt]>;

/// int __cxa_atexit(void (*f)(void *), void *p, void *d);
def cxa_atexit : TargetLibCall<"__cxa_atexit", Int, [Ptr, Ptr, Ptr]>;

/// int atexit(void (*f)(void));
def atexit : TargetLibCall<"atexit", Int, [Ptr]>;

/// void abort(void)
def abort : TargetLibCall<"abort", Void, []>;

/// void exit(int)
def exit : TargetLibCall<"exit", Void, [Int]>;

/// void _Exit(int)
def Exit : TargetLibCall<"_Exit", Void, [Int]>;

/// void std::terminate();
def terminate : TargetLibCall<"_ZSt9terminatev", Void, []>;

/// void __cxa_throw(void *, void *, void (*)(void *));
def cxa_throw : TargetLibCall<"__cxa_throw", Void, [Ptr, Ptr, Ptr]>;

/// void __cxa_guard_abort(guard_t *guard);
/// guard_t is int64_t in Itanium ABI or int32_t on ARM eabi.
def cxa_guard_abort : TargetLibCall<"__cxa_guard_abort", Void, [Ptr]>;

/// int __cxa_guard_acquire(guard_t *guard);
def cxa_guard_acquire : TargetLibCall<"__cxa_guard_acquire", Int, [Ptr]>;

/// void __cxa_guard_release(guard_t *guard);
def cxa_guard_release : TargetLibCall<"__cxa_guard_release", Void, [Ptr]>;

/// double __exp10_finite(double x);
def exp10_finite : TargetLibCall<"__exp10_finite", Dbl, [Dbl]>;

/// float __exp10f_finite(float x);
def exp10f_finite : TargetLibCall<"__exp10f_finite", Flt, [Flt]>;

/// long double __exp10l_finite(long double x);
def exp10l_finite : TargetLibCall<"__exp10l_finite", LDbl, [LDbl]>;

/// double __exp2_finite(double x);
def exp2_finite : TargetLibCall<"__exp2_finite", Dbl, [Dbl]>;

/// float __exp2f_finite(float x);
def exp2f_finite : TargetLibCall<"__exp2f_finite", Flt, [Flt]>;

/// long double __exp2l_finite(long double x);
def exp2l_finite : TargetLibCall<"__exp2l_finite", LDbl, [LDbl]>;

/// double __exp_finite(double x);
def exp_finite : TargetLibCall<"__exp_finite", Dbl, [Dbl]>;

/// float __expf_finite(float x);
def expf_finite : TargetLibCall<"__expf_finite", Flt, [Flt]>;

/// long double __expl_finite(long double x);
def expl_finite : TargetLibCall<"__expl_finite", LDbl, [LDbl]>;

/// int __isoc99_scanf (const char *format, ...)
def dunder_isoc99_scanf : TargetLibCall<"__isoc99_scanf", Int, [Ptr, Ellip]>;

/// int __isoc99_sscanf(const char *s, const char *format, ...)
def dunder_isoc99_sscanf
    : TargetLibCall<"__isoc99_sscanf", Int, [Ptr, Ptr, Ellip]>;

/// void* __kmpc_alloc_shared(size_t nbyte);
def __kmpc_alloc_shared : TargetLibCall<"__kmpc_alloc_shared", Ptr, [SizeT]>;

/// void __kmpc_free_shared(void *ptr, size_t nbyte);
def __kmpc_free_shared
    : TargetLibCall<"__kmpc_free_shared", Void, [Ptr, SizeT]>;

/// double __log10_finite(double x);
def log10_finite : TargetLibCall<"__log10_finite", Dbl, [Dbl]>;

/// float __log10f_finite(float x);
def log10f_finite : TargetLibCall<"__log10f_finite", Flt, [Flt]>;

/// long double __log10l_finite(long double x);
def log10l_finite : TargetLibCall<"__log10l_finite", LDbl, [LDbl]>;

/// double __log2_finite(double x);
def log2_finite : TargetLibCall<"__log2_finite", Dbl, [Dbl]>;

/// float __log2f_finite(float x);
def log2f_finite : TargetLibCall<"__log2f_finite", Flt, [Flt]>;

/// long double __log2l_finite(long double x);
def log2l_finite : TargetLibCall<"__log2l_finite", LDbl, [LDbl]>;

/// double __log_finite(double x);
def log_finite : TargetLibCall<"__log_finite", Dbl, [Dbl]>;

/// float __logf_finite(float x);
def logf_finite : TargetLibCall<"__logf_finite", Flt, [Flt]>;

/// long double __logl_finite(long double x);
def logl_finite : TargetLibCall<"__logl_finite", LDbl, [LDbl]>;

/// void *__memccpy_chk(void *dst, const void *src, int c, size_t n,
/// size_t dstsize)
def memccpy_chk
    : TargetLibCall<"__memccpy_chk", Ptr, [Ptr, Ptr, Int, SizeT, SizeT]>;

/// void *__memcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def memcpy_chk : TargetLibCall<"__memcpy_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// void *__memmove_chk(void *s1, const void *s2, size_t n, size_t s1size);
def memmove_chk : TargetLibCall<"__memmove_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// void *__mempcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
def mempcpy_chk : TargetLibCall<"__mempcpy_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// void *__memset_chk(void *s, int v, size_t n, size_t s1size);
def memset_chk : TargetLibCall<"__memset_chk", Ptr, [Ptr, Int, SizeT, SizeT]>;

// int __nvvm_reflect(const char *)
def nvvm_reflect : TargetLibCall<"__nvvm_reflect", Int, [Ptr]>;

/// double __pow_finite(double x, double y);
def pow_finite : TargetLibCall<"__pow_finite", Dbl, [Dbl, Dbl]>;

/// float _powf_finite(float x, float y);
def powf_finite : TargetLibCall<"__powf_finite", Flt, [Flt, Flt]>;

/// long double __powl_finite(long double x, long double y);
def powl_finite : TargetLibCall<"__powl_finite", LDbl, [LDbl, LDbl]>;

/// double __sincospi_stret(double x);
def sincospi_stret
    : TargetLibCall<"__sincospi_stret", ? /* Checked manually. */>;

/// float __sincospif_stret(float x);
def sincospif_stret
    : TargetLibCall<"__sincospif_stret", ? /* Checked manually. */>;

/// double __sinh_finite(double x);
def sinh_finite : TargetLibCall<"__sinh_finite", Dbl, [Dbl]>;

/// float _sinhf_finite(float x);
def sinhf_finite : TargetLibCall<"__sinhf_finite", Flt, [Flt]>;

/// long double __sinhl_finite(long double x);
def sinhl_finite : TargetLibCall<"__sinhl_finite", LDbl, [LDbl]>;

/// double __sinpi(double x);
def sinpi : TargetLibCall<"__sinpi", Dbl, [Dbl]>;

/// float __sinpif(float x);
def sinpif : TargetLibCall<"__sinpif", Flt, [Flt]>;

/// int __small_fprintf(FILE *stream, const char *format, ...);
def small_fprintf : TargetLibCall<"__small_fprintf", Int, [Ptr, Ptr, Ellip]>;

/// int __small_printf(const char *format, ...);
def small_printf : TargetLibCall<"__small_printf", Int, [Ptr, Ellip]>;

/// int __small_sprintf(char *str, const char *format, ...);
def small_sprintf : TargetLibCall<"__small_sprintf", Int, [Ptr, Ptr, Ellip]>;

/// int __snprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, ...);
def snprintf_chk : TargetLibCall<"__snprintf_chk",
                                 Int, [Ptr, SizeT, Int, SizeT, Ptr, Ellip]>;

/// int __sprintf_chk(char *str, int flags, size_t str_len,
/// const char *format, ...);
def sprintf_chk
    : TargetLibCall<"__sprintf_chk", Int, [Ptr, Int, SizeT, Ptr, Ellip]>;

/// double __sqrt_finite(double x);
def sqrt_finite : TargetLibCall<"__sqrt_finite", Dbl, [Dbl]>;

/// float __sqrt_finite(float x);
def sqrtf_finite : TargetLibCall<"__sqrtf_finite", Flt, [Flt]>;

/// long double __sqrt_finite(long double x);
def sqrtl_finite : TargetLibCall<"__sqrtl_finite", LDbl, [LDbl]>;

/// char *__stpcpy_chk(char *s1, const char *s2, size_t s1size);
def stpcpy_chk : TargetLibCall<"__stpcpy_chk", Ptr, [Ptr, Ptr, SizeT]>;

/// char *__stpncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def stpncpy_chk : TargetLibCall<"__stpncpy_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// char *__strcat_chk(char *s1, const char *s2, size_t s1size);
def strcat_chk : TargetLibCall<"__strcat_chk", Ptr, [Ptr, Ptr, SizeT]>;

/// char *__strcpy_chk(char *s1, const char *s2, size_t s1size);
def strcpy_chk : TargetLibCall<"__strcpy_chk", Ptr, [Ptr, Ptr, SizeT]>;

/// char * __strdup(const char *s);
def dunder_strdup : TargetLibCall<"__strdup", Ptr, [Ptr]>;

/// size_t __strlcat_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def strlcat_chk
    : TargetLibCall<"__strlcat_chk", SizeT, [Ptr, Ptr, SizeT, SizeT]>;

/// size_t __strlcpy_chk(char *dst, const char *src, size_t size,
/// size_t dstsize);
def strlcpy_chk
    : TargetLibCall<"__strlcpy_chk", SizeT, [Ptr, Ptr, SizeT, SizeT]>;

/// size_t __strlen_chk(const char *s1, size_t s1size);
def strlen_chk : TargetLibCall<"__strlen_chk", SizeT, [Ptr, SizeT]>;

/// char *strncat_chk(char *s1, const char *s2, size_t n, size_t s1size);
def strncat_chk : TargetLibCall<"__strncat_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// char *__strncpy_chk(char *s1, const char *s2, size_t n, size_t s1size);
def strncpy_chk : TargetLibCall<"__strncpy_chk", Ptr, [Ptr, Ptr, SizeT, SizeT]>;

/// char *__strndup(const char *s, size_t n);
def dunder_strndup : TargetLibCall<"__strndup", Ptr, [Ptr, SizeT]>;

/// char * __strtok_r(char *s, const char *delim, char **save_ptr);
def dunder_strtok_r : TargetLibCall<"__strtok_r", Ptr, [Ptr, Ptr, Ptr]>;

/// int __vsnprintf_chk(char *s, size_t n, int flags, size_t slen,
/// const char *format, va_list ap);
def vsnprintf_chk
    : TargetLibCall<"__vsnprintf_chk", Int, [Ptr, SizeT, Int, SizeT, Ptr, Ptr]>;

/// int __vsprintf_chk(char *s, int flags, size_t slen, const char *format,
/// va_list ap);
def vsprintf_chk
    : TargetLibCall<"__vsprintf_chk", Int, [Ptr, Int, SizeT, Ptr, Ptr]>;

/// int abs(int j);
def abs : TargetLibCall<"abs", Int, [Int]>;

/// int access(const char *path, int amode);
def access : TargetLibCall<"access", Int, [Ptr, Int]>;

/// double acos(double x);
def acos : TargetLibCall<"acos", Dbl, [Dbl]>;

/// float acosf(float x);
def acosf : TargetLibCall<"acosf", Flt, [Flt]>;

/// double acosh(double x);
def acosh : TargetLibCall<"acosh", Dbl, [Dbl]>;

/// float acoshf(float x);
def acoshf : TargetLibCall<"acoshf", Flt, [Flt]>;

/// long double acoshl(long double x);
def acoshl : TargetLibCall<"acoshl", LDbl, [LDbl]>;

/// long double acosl(long double x);
def acosl : TargetLibCall<"acosl", LDbl, [LDbl]>;

/// void *aligned_alloc(size_t alignment, size_t size);
def aligned_alloc : TargetLibCall<"aligned_alloc", Ptr, [SizeT, SizeT]>;

/// double asin(double x);
def asin : TargetLibCall<"asin", Dbl, [Dbl]>;

/// float asinf(float x);
def asinf : TargetLibCall<"asinf", Flt, [Flt]>;

/// double asinh(double x);
def asinh : TargetLibCall<"asinh", Dbl, [Dbl]>;

/// float asinhf(float x);
def asinhf : TargetLibCall<"asinhf", Flt, [Flt]>;

/// long double asinhl(long double x);
def asinhl : TargetLibCall<"asinhl", LDbl, [LDbl]>;

/// long double asinl(long double x);
def asinl : TargetLibCall<"asinl", LDbl, [LDbl]>;

/// double atan(double x);
def atan : TargetLibCall<"atan", Dbl, [Dbl]>;

/// double atan2(double y, double x);
def atan2 : TargetLibCall<"atan2", Dbl, [Dbl, Dbl]>;

/// float atan2f(float y, float x);
def atan2f : TargetLibCall<"atan2f", Flt, [Flt, Flt]>;

/// long double atan2l(long double y, long double x);
def atan2l : TargetLibCall<"atan2l", LDbl, [LDbl, LDbl]>;

/// float atanf(float x);
def atanf : TargetLibCall<"atanf", Flt, [Flt]>;

/// double atanh(double x);
def atanh : TargetLibCall<"atanh", Dbl, [Dbl]>;

/// float atanhf(float x);
def atanhf : TargetLibCall<"atanhf", Flt, [Flt]>;

/// long double atanhl(long double x);
def atanhl : TargetLibCall<"atanhl", LDbl, [LDbl]>;

/// long double atanl(long double x);
def atanl : TargetLibCall<"atanl", LDbl, [LDbl]>;

/// double atof(const char *str);
def atof : TargetLibCall<"atof", Dbl, [Ptr]>;

/// int atoi(const char *str);
def atoi : TargetLibCall<"atoi", Int, [Ptr]>;

/// long atol(const char *str);
def atol : TargetLibCall<"atol", Long, [Ptr]>;

/// long long atoll(const char *nptr);
def atoll : TargetLibCall<"atoll", LLong, [Ptr]>;

/// int bcmp(const void *s1, const void *s2, size_t n);
def bcmp : TargetLibCall<"bcmp", Int, [Ptr, Ptr, SizeT]>;

/// void bcopy(const void *s1, void *s2, size_t n);
def bcopy : TargetLibCall<"bcopy", Void, [Ptr, Ptr, SizeT]>;

/// void bzero(void *s, size_t n);
def bzero : TargetLibCall<"bzero", Void, [Ptr, SizeT]>;

/// double cabs(double complex z)
def cabs : TargetLibCall<"cabs", ? /* Checked manually. */>;

/// float cabs(float complex z)
def cabsf : TargetLibCall<"cabsf", ? /* Checked manually. */>;

/// long double cabs(long double complex z)
def cabsl : TargetLibCall<"cabsl", ? /* Checked manually. */>;

/// void *calloc(size_t count, size_t size);
def calloc : TargetLibCall<"calloc", Ptr, [SizeT, SizeT]>;

/// double cbrt(double x);
def cbrt : TargetLibCall<"cbrt", Dbl, [Dbl]>;

/// float cbrtf(float x);
def cbrtf : TargetLibCall<"cbrtf", Flt, [Flt]>;

/// long double cbrtl(long double x);
def cbrtl : TargetLibCall<"cbrtl", LDbl, [LDbl]>;

/// double ceil(double x);
def ceil : TargetLibCall<"ceil", Dbl, [Dbl]>;

/// float ceilf(float x);
def ceilf : TargetLibCall<"ceilf", Flt, [Flt]>;

/// long double ceill(long double x);
def ceill : TargetLibCall<"ceill", LDbl, [LDbl]>;

/// int chmod(const char *path, mode_t mode);
def chmod : TargetLibCall<"chmod", Int, [Ptr, IntX]>;

/// int chown(const char *path, uid_t owner, gid_t group);
def chown : TargetLibCall<"chown", Int, [Ptr, IntX, IntX]>;

/// void clearerr(FILE *stream);
def clearerr : TargetLibCall<"clearerr", Void, [Ptr]>;

/// int closedir(DIR *dirp);
def closedir : TargetLibCall<"closedir", Int, [Ptr]>;

/// double copysign(double x, double y);
def copysign : TargetLibCall<"copysign", Dbl, [Dbl, Dbl]>;

/// float copysignf(float x, float y);
def copysignf : TargetLibCall<"copysignf", Flt, [Flt, Flt]>;

/// long double copysignl(long double x, long double y);
def copysignl : TargetLibCall<"copysignl", LDbl, [LDbl, LDbl]>;

/// double cos(double x);
def cos : TargetLibCall<"cos", Dbl, [Dbl]>;

/// float cosf(float x);
def cosf : TargetLibCall<"cosf", Flt, [Flt]>;

/// double cosh(double x);
def cosh : TargetLibCall<"cosh", Dbl, [Dbl]>;

/// float coshf(float x);
def coshf : TargetLibCall<"coshf", Flt, [Flt]>;

/// long double coshl(long double x);
def coshl : TargetLibCall<"coshl", LDbl, [LDbl]>;

/// long double cosl(long double x);
def cosl : TargetLibCall<"cosl", LDbl, [LDbl]>;

/// char *ctermid(char *s);
def ctermid : TargetLibCall<"ctermid", Ptr, [Ptr]>;

/// double erf(double x);
def erf : TargetLibCall<"erf", Dbl, [Dbl]>;

/// float erff(float x);
def erff : TargetLibCall<"erff", Flt, [Flt]>;

/// long double erfl(long double x);
def erfl : TargetLibCall<"erfl", LDbl, [LDbl]>;

/// double tgamma(double x);
def tgamma : TargetLibCall<"tgamma", Dbl, [Dbl]>;

/// float tgammaf(float x);
def tgammaf : TargetLibCall<"tgammaf", Flt, [Flt]>;

/// long double tgammal(long double x);
def tgammal : TargetLibCall<"tgammal", LDbl, [LDbl]>;

/// int execl(const char *path, const char *arg, ...);
def execl : TargetLibCall<"execl", Int, [Ptr, Ptr, Ellip]>;

/// int execle(const char *file, const char *arg, ..., char * const envp[]);
def execle : TargetLibCall<"execle", Int, [Ptr, Ptr, Ellip]>;

/// int execlp(const char *file, const char *arg, ...);
def execlp : TargetLibCall<"execlp", Int, [Ptr, Ptr, Ellip]>;

/// int execv(const char *path, char *const argv[]);
def execv : TargetLibCall<"execv", Int, [Ptr, Ptr]>;

/// int execvP(const char *file, const char *search_path, char *const argv[]);
def execvP : TargetLibCall<"execvP", Int, [Ptr, Ptr, Ptr]>;

/// int execve(const char *filename, char *const argv], [char *const envp[]);
def execve : TargetLibCall<"execve", Int, [Ptr, Ptr, Ptr]>;

/// int execvp(const char *file, char *const argv[]);
def execvp : TargetLibCall<"execvp", Int, [Ptr, Ptr]>;

/// int execvpe(const char *file, char *const argv], [char *const envp[]);
def execvpe : TargetLibCall<"execvpe", Int, [Ptr, Ptr, Ptr]>;

/// double exp(double x);
def exp : TargetLibCall<"exp", Dbl, [Dbl]>;

/// double exp10(double x);
def exp10 : TargetLibCall<"exp10", Dbl, [Dbl]>;

/// float exp10f(float x);
def exp10f : TargetLibCall<"exp10f", Flt, [Flt]>;

/// long double exp10l(long double x);
def exp10l : TargetLibCall<"exp10l", LDbl, [LDbl]>;

/// double exp2(double x);
def exp2 : TargetLibCall<"exp2", Dbl, [Dbl]>;

/// float exp2f(float x);
def exp2f : TargetLibCall<"exp2f", Flt, [Flt]>;

/// long double exp2l(long double x);
def exp2l : TargetLibCall<"exp2l", LDbl, [LDbl]>;

/// float expf(float x);
def expf : TargetLibCall<"expf", Flt, [Flt]>;

/// long double expl(long double x);
def expl : TargetLibCall<"expl", LDbl, [LDbl]>;

/// double expm1(double x);
def expm1 : TargetLibCall<"expm1", Dbl, [Dbl]>;

/// float expm1f(float x);
def expm1f : TargetLibCall<"expm1f", Flt, [Flt]>;

/// long double expm1l(long double x);
def expm1l : TargetLibCall<"expm1l", LDbl, [LDbl]>;

/// double fabs(double x);
def fabs : TargetLibCall<"fabs", Dbl, [Dbl]>;

/// float fabsf(float x);
def fabsf : TargetLibCall<"fabsf", Flt, [Flt]>;

/// long double fabsl(long double x);
def fabsl : TargetLibCall<"fabsl", LDbl, [LDbl]>;

/// int fclose(FILE *stream);
def fclose : TargetLibCall<"fclose", Int, [Ptr]>;

/// FILE *fdopen(int fildes, const char *mode);
def fdopen : TargetLibCall<"fdopen", Ptr, [Int, Ptr]>;

/// int feof(FILE *stream);
def feof : TargetLibCall<"feof", Int, [Ptr]>;

/// int ferror(FILE *stream);
def ferror : TargetLibCall<"ferror", Int, [Ptr]>;

/// int fflush(FILE *stream);
def fflush : TargetLibCall<"fflush", Int, [Ptr]>;

/// int ffs(int i);
def ffs : TargetLibCall<"ffs", Int, [Int]>;

/// int ffsl(long int i);
def ffsl : TargetLibCall<"ffsl", Int, [Long]>;

/// int ffsll(long long int i);
def ffsll : TargetLibCall<"ffsll", Int, [LLong]>;

/// int fgetc(FILE *stream);
def fgetc : TargetLibCall<"fgetc", Int, [Ptr]>;

/// int fgetc_unlocked(FILE *stream);
def fgetc_unlocked : TargetLibCall<"fgetc_unlocked", Int, [Ptr]>;

/// int fgetpos(FILE *stream, fpos_t *pos);
def fgetpos : TargetLibCall<"fgetpos", Int, [Ptr, Ptr]>;

/// char *fgets(char *s, int n, FILE *stream);
def fgets : TargetLibCall<"fgets", Ptr, [Ptr, Int, Ptr]>;

/// char *fgets_unlocked(char *s, int n, FILE *stream);
def fgets_unlocked : TargetLibCall<"fgets_unlocked", Ptr, [Ptr, Int, Ptr]>;

/// int fileno(FILE *stream);
def fileno : TargetLibCall<"fileno", Int, [Ptr]>;

/// int fiprintf(FILE *stream, const char *format, ...);
def fiprintf : TargetLibCall<"fiprintf", Int, [Ptr, Ptr, Ellip]>;

/// void flockfile(FILE *file);
def flockfile : TargetLibCall<"flockfile", Void, [Ptr]>;

/// double floor(double x);
def floor : TargetLibCall<"floor", Dbl, [Dbl]>;

/// float floorf(float x);
def floorf : TargetLibCall<"floorf", Flt, [Flt]>;

/// long double floorl(long double x);
def floorl : TargetLibCall<"floorl", LDbl, [LDbl]>;

/// int fls(int i);
def fls : TargetLibCall<"fls", Int, [Int]>;

/// int flsl(long int i);
def flsl : TargetLibCall<"flsl", Int, [Long]>;

/// int flsll(long long int i);
def flsll : TargetLibCall<"flsll", Int, [LLong]>;

// Calls to fmax and fmin library functions expand to the llvm.maxnnum and
// llvm.minnum intrinsics with the correct parameter types for the arguments
// (all types must match).
/// double fmax(double x, double y);
def fmax : TargetLibCall<"fmax", Floating, [Same, Same]>;

/// float fmaxf(float x, float y);
def fmaxf : TargetLibCall<"fmaxf", Floating, [Same, Same]>;

/// long double fmaxl(long double x, long double y);
def fmaxl : TargetLibCall<"fmaxl", Floating, [Same, Same]>;

/// double fmin(double x, double y);
def fmin : TargetLibCall<"fmin", Floating, [Same, Same]>;

/// float fminf(float x, float y);
def fminf : TargetLibCall<"fminf", Floating, [Same, Same]>;

/// long double fminl(long double x, long double y);
def fminl : TargetLibCall<"fminl", Floating, [Same, Same]>;

// Calls to fmaximum_num and fminimum_num library functions expand to the
// llvm.maximumnum and llvm.minimumnum intrinsics with the correct parameter
// types for the arguments (all types must match).
/// double fmaximum_num(double x, double y);
def fmaximum_num : TargetLibCall<"fmaximum_num", Floating, [Same, Same]>;

/// float fmaximum_numf(float x, float y);
def fmaximum_numf : TargetLibCall<"fmaximum_numf", Floating, [Same, Same]>;

/// long double fmaximum_numl(long double x, long double y);
def fmaximum_numl : TargetLibCall<"fmaximum_numl", Floating, [Same, Same]>;

/// double fminimum_num(double x, double y);
def fminimum_num : TargetLibCall<"fminimum_num", Floating, [Same, Same]>;

/// float fminimum_numf(float x, float y);
def fminimum_numf : TargetLibCall<"fminimum_numf", Floating, [Same, Same]>;

/// long double fminimum_numl(long double x, long double y);
def fminimum_numl : TargetLibCall<"fminimum_numl", Floating, [Same, Same]>;

/// double fmod(double x, double y);
def fmod : TargetLibCall<"fmod", Dbl, [Dbl, Dbl]>;

/// float fmodf(float x, float y);
def fmodf : TargetLibCall<"fmodf", Flt, [Flt, Flt]>;

/// long double fmodl(long double x, long double y);
def fmodl : TargetLibCall<"fmodl", LDbl, [LDbl, LDbl]>;

/// FILE *fopen(const char *filename, const char *mode);
def fopen : TargetLibCall<"fopen", Ptr, [Ptr, Ptr]>;

/// FILE *fopen64(const char *filename, const char *opentype)
def fopen64 : TargetLibCall<"fopen64", Ptr, [Ptr, Ptr]>;

/// int fork();
def fork : TargetLibCall<"fork", Int, []>;

/// int fprintf(FILE *stream, const char *format, ...);
def fprintf : TargetLibCall<"fprintf", Int, [Ptr, Ptr, Ellip]>;

/// int fputc(int c, FILE *stream);
def fputc : TargetLibCall<"fputc", Int, [Int, Ptr]>;

/// int fputc_unlocked(int c, FILE *stream);
def fputc_unlocked : TargetLibCall<"fputc_unlocked", Int, [Int, Ptr]>;

/// int fputs(const char *s, FILE *stream);
def fputs : TargetLibCall<"fputs", Int, [Ptr, Ptr]>;

/// int fputs_unlocked(const char *s, FILE *stream);
def fputs_unlocked : TargetLibCall<"fputs_unlocked", Int, [Ptr, Ptr]>;

/// size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
def fread : TargetLibCall<"fread", SizeT, [Ptr, SizeT, SizeT, Ptr]>;

/// size_t fread_unlocked(void *ptr, size_t size, size_t nitems, FILE *stream);
def fread_unlocked
    : TargetLibCall<"fread_unlocked", SizeT, [Ptr, SizeT, SizeT, Ptr]>;

/// void free(void *ptr);
def free : TargetLibCall<"free", Void, [Ptr]>;

/// double frexp(double num, int *exp);
def frexp : TargetLibCall<"frexp", Dbl, [Dbl, Ptr]>;

/// float frexpf(float num, int *exp);
def frexpf : TargetLibCall<"frexpf", Flt, [Flt, Ptr]>;

/// long double frexpl(long double num, int *exp);
def frexpl : TargetLibCall<"frexpl", LDbl, [LDbl, Ptr]>;

/// int fscanf(FILE *stream, const char *format, ... );
def fscanf : TargetLibCall<"fscanf", Int, [Ptr, Ptr, Ellip]>;

/// int fseek(FILE *stream, long offset, int whence);
def fseek : TargetLibCall<"fseek", Int, [Ptr, Long, Int]>;

/// int fseeko(FILE *stream, off_t offset, int whence);
def fseeko : TargetLibCall<"fseeko", Int, [Ptr, IntX, Int]>;

/// int fseeko64(FILE *stream, off64_t offset, int whence)
def fseeko64 : TargetLibCall<"fseeko64", Int, [Ptr, Int64, Int]>;

/// int fsetpos(FILE *stream, const fpos_t *pos);
def fsetpos : TargetLibCall<"fsetpos", Int, [Ptr, Ptr]>;

/// int fstat(int fildes, struct stat *buf);
def fstat : TargetLibCall<"fstat", Int, [Int, Ptr]>;

/// int fstat64(int filedes, struct stat64 *buf)
def fstat64 : TargetLibCall<"fstat64", Int, [Int, Ptr]>;

/// int fstatvfs(int fildes, struct statvfs *buf);
def fstatvfs : TargetLibCall<"fstatvfs", Int, [Int, Ptr]>;

/// int fstatvfs64(int fildes, struct statvfs64 *buf);
def fstatvfs64 : TargetLibCall<"fstatvfs64", Int, [Int, Ptr]>;

/// long ftell(FILE *stream);
def ftell : TargetLibCall<"ftell", Long, [Ptr]>;

/// off_t ftello(FILE *stream);
def ftello : TargetLibCall<"ftello", IntPlus, [Ptr]>;

/// off64_t ftello64(FILE *stream)
def ftello64 : TargetLibCall<"ftello64", Int64, [Ptr]>;

/// int ftrylockfile(FILE *file);
def ftrylockfile : TargetLibCall<"ftrylockfile", Int, [Ptr]>;

/// void funlockfile(FILE *file);
def funlockfile : TargetLibCall<"funlockfile", Void, [Ptr]>;

/// size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);
def fwrite : TargetLibCall<"fwrite", SizeT, [Ptr, SizeT, SizeT, Ptr]>;

/// size_t fwrite_unlocked(const void *ptr, size_t size, size_t nitems,
/// FILE *stream);
def fwrite_unlocked
    : TargetLibCall<"fwrite_unlocked", SizeT, [Ptr, SizeT, SizeT, Ptr]>;

/// int getc(FILE *stream);
def getc : TargetLibCall<"getc", Int, [Ptr]>;

/// int getc_unlocked(FILE *stream);
def getc_unlocked : TargetLibCall<"getc_unlocked", Int, [Ptr]>;

/// int getchar(void);
def getchar : TargetLibCall<"getchar", Int, []>;

/// int getchar_unlocked(void);
def getchar_unlocked : TargetLibCall<"getchar_unlocked", Int, []>;

/// char *getenv(const char *name);
def getenv : TargetLibCall<"getenv", Ptr, [Ptr]>;

/// int getitimer(int which, struct itimerval *value);
def getitimer : TargetLibCall<"getitimer", Int, [Int, Ptr]>;

/// int getlogin_r(char *name, size_t namesize);
def getlogin_r : TargetLibCall<"getlogin_r", Int, [Ptr, SizeT]>;

/// struct passwd *getpwnam(const char *name);
def getpwnam : TargetLibCall<"getpwnam", Ptr, [Ptr]>;

/// char *gets(char *s);
def gets : TargetLibCall<"gets", Ptr, [Ptr]>;

/// int gettimeofday(struct timeval *tp, void *tzp);
def gettimeofday : TargetLibCall<"gettimeofday", Int, [Ptr, Ptr]>;

/// uint32_t htonl(uint32_t hostlong);
def htonl : TargetLibCall<"htonl", Int32, [Int32]>;

/// uint16_t htons(uint16_t hostshort);
def htons : TargetLibCall<"htons", Int16, [Int16]>;

/// double hypot(double x, double y);
def hypot : TargetLibCall<"hypot", Dbl, [Dbl, Dbl]>;

/// float hypotf(float x, float y);
def hypotf : TargetLibCall<"hypotf", Flt, [Flt, Flt]>;

/// long double hypotl(long double x, long double y);
def hypotl : TargetLibCall<"hypotl", LDbl, [LDbl, LDbl]>;

/// int iprintf(const char *format, ...);
def iprintf : TargetLibCall<"iprintf", Int, [Ptr, Ellip]>;

/// int isascii(int c);
def isascii : TargetLibCall<"isascii", Int, [Int]>;

/// int isdigit(int c);
def isdigit : TargetLibCall<"isdigit", Int, [Int]>;

/// long int labs(long int j);
def labs : TargetLibCall<"labs", Long, [Same]>;

/// int lchown(const char *path, uid_t owner, gid_t group);
def lchown : TargetLibCall<"lchown", Int, [Ptr, IntX, IntX]>;

/// double ldexp(double x, int n);
def ldexp : TargetLibCall<"ldexp", Dbl, [Dbl, Int]>;

/// float ldexpf(float x, int n);
def ldexpf : TargetLibCall<"ldexpf", Flt, [Flt, Int]>;

/// long double ldexpl(long double x, int n);
def ldexpl : TargetLibCall<"ldexpl", LDbl, [LDbl, Int]>;

/// long long int llabs(long long int j);
def llabs : TargetLibCall<"llabs", LLong, [LLong]>;

/// double log(double x);
def log : TargetLibCall<"log", Dbl, [Dbl]>;

/// double log10(double x);
def log10 : TargetLibCall<"log10", Dbl, [Dbl]>;

/// float log10f(float x);
def log10f : TargetLibCall<"log10f", Flt, [Flt]>;

/// long double log10l(long double x);
def log10l : TargetLibCall<"log10l", LDbl, [LDbl]>;

/// double log1p(double x);
def log1p : TargetLibCall<"log1p", Dbl, [Dbl]>;

/// float log1pf(float x);
def log1pf : TargetLibCall<"log1pf", Flt, [Flt]>;

/// long double log1pl(long double x);
def log1pl : TargetLibCall<"log1pl", LDbl, [LDbl]>;

/// double log2(double x);
def log2 : TargetLibCall<"log2", Dbl, [Dbl]>;

/// float log2f(float x);
def log2f : TargetLibCall<"log2f", Flt, [Flt]>;

/// double long double log2l(long double x);
def log2l : TargetLibCall<"log2l", LDbl, [LDbl]>;

/// int ilogb(double x);
def ilogb : TargetLibCall<"ilogb", Int, [Dbl]>;

/// int ilogbf(float x);
def ilogbf : TargetLibCall<"ilogbf", Int, [Flt]>;

/// int ilogbl(long double x);
def ilogbl : TargetLibCall<"ilogbl", Int, [LDbl]>;

/// double logb(double x);
def logb : TargetLibCall<"logb", Dbl, [Dbl]>;

/// float logbf(float x);
def logbf : TargetLibCall<"logbf", Flt, [Flt]>;

/// long double logbl(long double x);
def logbl : TargetLibCall<"logbl", LDbl, [LDbl]>;

/// float logf(float x);
def logf : TargetLibCall<"logf", Flt, [Flt]>;

/// long double logl(long double x);
def logl : TargetLibCall<"logl", LDbl, [LDbl]>;

/// int lstat(const char *path, struct stat *buf);
def lstat : TargetLibCall<"lstat", Int, [Ptr, Ptr]>;

/// int lstat64(const char *path, struct stat64 *buf);
def lstat64 : TargetLibCall<"lstat64", Int, [Ptr, Ptr]>;

/// void *malloc(size_t size);
def malloc : TargetLibCall<"malloc", Ptr, [SizeT]>;

/// void *memalign(size_t boundary, size_t size);
def memalign : TargetLibCall<"memalign", Ptr, [SizeT, SizeT]>;

/// void *memccpy(void *s1, const void *s2, int c, size_t n);
def memccpy : TargetLibCall<"memccpy", Ptr, [Ptr, Ptr, Int, SizeT]>;

/// void *memchr(const void *s, int c, size_t n);
def memchr : TargetLibCall<"memchr", Ptr, [Ptr, Int, SizeT]>;

/// int memcmp(const void *s1, const void *s2, size_t n);
def memcmp : TargetLibCall<"memcmp", Int, [Ptr, Ptr, SizeT]>;

/// void *memcpy(void *s1, const void *s2, size_t n);
def memcpy : TargetLibCall<"memcpy", Ptr, [Ptr, Ptr, SizeT]>;

/// void *memmove(void *s1, const void *s2, size_t n);
def memmove : TargetLibCall<"memmove", Ptr, [Ptr, Ptr, SizeT]>;

/// void *mempcpy(void *s1, const void *s2, size_t n);
def mempcpy : TargetLibCall<"mempcpy", Ptr, [Ptr, Ptr, SizeT]>;

/// void *memrchr(const void *s, int c, size_t n);
def memrchr : TargetLibCall<"memrchr", Ptr, [Ptr, Int, SizeT]>;

/// void *memset(void *b, int c, size_t len);
def memset : TargetLibCall<"memset", Ptr, [Ptr, Int, SizeT]>;

/// void memset_pattern16(void *b, const void *pattern16, size_t len);
def memset_pattern16
    : TargetLibCall<"memset_pattern16", Void, [Ptr, Ptr, SizeT]>;

/// void memset_pattern4(void *b, const void *pattern4, size_t len);
def memset_pattern4 : TargetLibCall<"memset_pattern4", Void, [Ptr, Ptr, SizeT]>;

/// void memset_pattern8(void *b, const void *pattern8, size_t len);
def memset_pattern8 : TargetLibCall<"memset_pattern8", Void, [Ptr, Ptr, SizeT]>;

/// int mkdir(const char *path, mode_t mode);
def mkdir : TargetLibCall<"mkdir", Int, [Ptr, IntX]>;

/// time_t mktime(struct tm *timeptr);
def mktime : TargetLibCall<"mktime", IntPlus, [Ptr]>;

/// double modf(double x, double *iptr);
def modf : TargetLibCall<"modf", Dbl, [Dbl, Ptr]>;

/// float modff(float, float *iptr);
def modff : TargetLibCall<"modff", Flt, [Flt, Ptr]>;

/// long double modfl(long double value, long double *iptr);
def modfl : TargetLibCall<"modfl", LDbl, [LDbl, Ptr]>;

/// double nan(const char *arg);
def nan : TargetLibCall<"nan", Dbl, [Ptr]>;

/// float nanf(const char *arg);
def nanf : TargetLibCall<"nanf", Flt, [Ptr]>;

/// long double nanl(const char *arg);
def nanl : TargetLibCall<"nanl", LDbl, [Ptr]>;

/// double nearbyint(double x);
def nearbyint : TargetLibCall<"nearbyint", Dbl, [Dbl]>;

/// float nearbyintf(float x);
def nearbyintf : TargetLibCall<"nearbyintf", Flt, [Flt]>;

/// long double nearbyintl(long double x);
def nearbyintl : TargetLibCall<"nearbyintl", LDbl, [LDbl]>;

/// double nextafter(double x, double y);
def nextafter : TargetLibCall< "nextafter", Dbl, [Dbl, Dbl]>;

/// float nextafterf(float x, float y);
def nextafterf : TargetLibCall< "nextafterf", Flt, [Flt, Flt]>;

/// long double nextafterl(long double x, long double y);
def nextafterl : TargetLibCall< "nextafterl", LDbl, [LDbl, LDbl]>;

/// double nexttoward(double x, long double y);
def nexttoward : TargetLibCall< "nexttoward", Dbl, [Dbl, LDbl]>;

/// float nexttowardf(float x, long double y);
def nexttowardf : TargetLibCall< "nexttowardf", Flt, [Flt, LDbl]>;

/// long double nexttowardl(long double x, long double y);
def nexttowardl : TargetLibCall< "nexttowardl", LDbl, [LDbl, LDbl]>;

/// uint32_t ntohl(uint32_t netlong);
def ntohl : TargetLibCall<"ntohl", Int32, [Int32]>;

/// uint16_t ntohs(uint16_t netshort);
def ntohs : TargetLibCall<"ntohs", Int16, [Int16]>;

/// int open(const char *path, int oflag, ... );
def open : TargetLibCall<"open", Int, [Ptr, Int, Ellip]>;

/// int open64(const char *filename, int flags, [mode_t mode])
def open64 : TargetLibCall<"open64", Int, [Ptr, Int, Ellip]>;

/// DIR *opendir(const char *dirname);
def opendir : TargetLibCall<"opendir", Ptr, [Ptr]>;

/// int pclose(FILE *stream);
def pclose : TargetLibCall<"pclose", Int, [Ptr]>;

/// void perror(const char *s);
def perror : TargetLibCall<"perror", Void, [Ptr]>;

/// FILE *popen(const char *command, const char *mode);
def popen : TargetLibCall<"popen", Ptr, [Ptr, Ptr]>;

/// int posix_memalign(void **memptr, size_t alignment, size_t size);
def posix_memalign : TargetLibCall<"posix_memalign", Int, [Ptr, SizeT, SizeT]>;

/// double pow(double x, double y);
def pow : TargetLibCall<"pow", Dbl, [Dbl, Dbl]>;

/// float powf(float x, float y);
def powf : TargetLibCall<"powf", Flt, [Flt, Flt]>;

/// long double powl(long double x, long double y);
def powl : TargetLibCall<"powl", LDbl, [LDbl, LDbl]>;

/// ssize_t pread(int fildes, void *buf, size_t nbyte, off_t offset);
def pread : TargetLibCall<"pread", SSizeT, [Int, Ptr, SizeT, IntPlus]>;

/// int printf(const char *format, ...);
def printf : TargetLibCall<"printf", Int, [Ptr, Ellip]>;

/// int putc(int c, FILE *stream);
def putc : TargetLibCall<"putc", Int, [Int, Ptr]>;

/// int putc_unlocked(int c, FILE *stream);
def putc_unlocked : TargetLibCall<"putc_unlocked", Int, [Int, Ptr]>;

/// int putchar(int c);
def putchar : TargetLibCall<"putchar", Int, [Int]>;

/// int putchar_unlocked(int c);
def putchar_unlocked : TargetLibCall<"putchar_unlocked", Int, [Int]>;

/// int puts(const char *s);
def puts : TargetLibCall<"puts", Int, [Ptr]>;

/// void *pvalloc(size_t size);
def pvalloc : TargetLibCall<"pvalloc", Ptr, [SizeT]>;

/// ssize_t pwrite(int fildes, const void *buf, size_t nbyte, off_t offset);
def pwrite : TargetLibCall<"pwrite", SSizeT, [Int, Ptr, SizeT, IntPlus]>;

/// void qsort(void *base, size_t nel, size_t width,
///            int (*compar)(const void *, const void *));
def qsort : TargetLibCall<"qsort", Void, [Ptr, SizeT, SizeT, Ptr]>;

/// ssize_t read(int fildes, void *buf, size_t nbyte);
def read : TargetLibCall<"read", SSizeT, [Int, Ptr, SizeT]>;

/// ssize_t readlink(const char *path, char *buf, size_t bufsize);
def readlink : TargetLibCall<"readlink", SSizeT, [Ptr, Ptr, SizeT]>;

/// void *realloc(void *ptr, size_t size);
def realloc : TargetLibCall<"realloc", Ptr, [Ptr, SizeT]>;

/// void *reallocf(void *ptr, size_t size);
def reallocf : TargetLibCall<"reallocf", Ptr, [Ptr, SizeT]>;

/// void *reallocarray(void *ptr, size_t nmemb, size_t size);
def reallocarray : TargetLibCall<"reallocarray", Ptr, [Ptr, SizeT, SizeT]>;

/// char *realpath(const char *file_name, char *resolved_name);
def realpath : TargetLibCall<"realpath", Ptr, [Ptr, Ptr]>;

/// double remainder(double x, double y);
def remainder : TargetLibCall<"remainder", Dbl, [Dbl, Dbl]>;

/// float remainderf(float x, float y);
def remainderf : TargetLibCall<"remainderf", Flt, [Flt, Flt]>;

/// long double remainderl(long double x, long double y);
def remainderl : TargetLibCall<"remainderl", LDbl, [LDbl, LDbl]>;

/// double remquo(double x, double y, int *quo);
def remquo : TargetLibCall<"remquo", Dbl, [Dbl, Dbl, Ptr]>;

/// float remquof(float x, float y, int *quo);
def remquof : TargetLibCall<"remquof", Flt, [Flt, Flt, Ptr]>;

/// long double remquol(long double x, long double y, int *quo);
def remquol : TargetLibCall<"remquol", LDbl, [LDbl, LDbl, Ptr]>;

/// double fdim(double x, double y);
def fdim : TargetLibCall<"fdim", Dbl, [Dbl, Dbl]>;

/// float fdimf(float x, float y);
def fdimf : TargetLibCall<"fdimf", Flt, [Flt, Flt]>;

/// long double fdiml(long double x, long double y);
def fdiml : TargetLibCall<"fdiml", LDbl, [LDbl, LDbl]>;

/// int remove(const char *path);
def remove : TargetLibCall<"remove", Int, [Ptr]>;

/// int rename(const char *old, const char *new);
def rename : TargetLibCall<"rename", Int, [Ptr, Ptr]>;

/// void rewind(FILE *stream);
def rewind : TargetLibCall<"rewind", Void, [Ptr]>;

/// double rint(double x);
def rint : TargetLibCall<"rint", Dbl, [Dbl]>;

/// float rintf(float x);
def rintf : TargetLibCall<"rintf", Flt, [Flt]>;

/// long double rintl(long double x);
def rintl : TargetLibCall<"rintl", LDbl, [LDbl]>;

/// int rmdir(const char *path);
def rmdir : TargetLibCall<"rmdir", Int, [Ptr]>;

/// double round(double x);
def round : TargetLibCall<"round", Dbl, [Dbl]>;

/// double roundeven(double x);
def roundeven : TargetLibCall<"roundeven", Dbl, [Dbl]>;

/// float roundevenf(float x);
def roundevenf : TargetLibCall<"roundevenf", Flt, [Flt]>;

/// long double roundevenl(long double x);
def roundevenl : TargetLibCall<"roundevenl", LDbl, [LDbl]>;

/// float roundf(float x);
def roundf : TargetLibCall<"roundf", Flt, [Flt]>;

/// long double roundl(long double x);
def roundl : TargetLibCall<"roundl", LDbl, [LDbl]>;

/// double scalbln(double arg, long exp);
def scalbln : TargetLibCall<"scalbln", Dbl, [Dbl, Long]>;

/// float scalblnf(float arg, long exp);
def scalblnf : TargetLibCall<"scalblnf", Flt, [Flt, Long]>;

/// long double scalblnl(long double arg, long exp);
def scalblnl : TargetLibCall<"scalblnl", LDbl, [LDbl, Long]>;

/// double scalbn(double arg, int exp);
def scalbn : TargetLibCall<"scalbn", Dbl, [Dbl, Int]>;

/// float scalbnf(float arg, int exp);
def scalbnf : TargetLibCall<"scalbnf", Flt, [Flt, Int]>;

/// long double scalbnl(long double arg, int exp);
def scalbnl : TargetLibCall<"scalbnl", LDbl, [LDbl, Int]>;

/// int scanf(const char *restrict format, ... );
def scanf : TargetLibCall<"scanf", Int, [Ptr, Ellip]>;

/// void setbuf(FILE *stream, char *buf);
def setbuf : TargetLibCall<"setbuf", Void, [Ptr, Ptr]>;

/// int setitimer(int which, const struct itimerval *value,
/// struct itimerval *ovalue);
def setitimer : TargetLibCall<"setitimer", Int, [Int, Ptr, Ptr]>;

/// int setvbuf(FILE *stream, char *buf, int type, size_t size);
def setvbuf : TargetLibCall<"setvbuf", Int, [Ptr, Ptr, Int, SizeT]>;

/// double sin(double x);
def sin : TargetLibCall<"sin", Dbl, [Dbl]>;

/// float sinf(float x);
def sinf : TargetLibCall<"sinf", Flt, [Flt]>;

/// double sinh(double x);
def sinh : TargetLibCall<"sinh", Dbl, [Dbl]>;

/// float sinhf(float x);
def sinhf : TargetLibCall<"sinhf", Flt, [Flt]>;

/// long double sinhl(long double x);
def sinhl : TargetLibCall<"sinhl", LDbl, [LDbl]>;

/// long double sinl(long double x);
def sinl : TargetLibCall<"sinl", LDbl, [LDbl]>;

/// void sincos(double x, double *sin_out, double *cos_out);
def sincos : TargetLibCall<"sincos", Void, [Dbl, Ptr, Ptr]>;

/// void sincosf(float x, float *sin_out, float *cos_out);
def sincosf : TargetLibCall<"sincosf", Void, [Flt, Ptr, Ptr]>;

/// void sincosl(long double x, long double *sin_out, long double *cos_out);
def sincosl : TargetLibCall<"sincosl", Void, [LDbl, Ptr, Ptr]>;

/// int siprintf(char *str, const char *format, ...);
def siprintf : TargetLibCall<"siprintf", Int, [Ptr, Ptr, Ellip]>;

/// int snprintf(char *s, size_t n, const char *format, ...);
def snprintf : TargetLibCall<"snprintf", Int, [Ptr, SizeT, Ptr, Ellip]>;

/// int sprintf(char *str, const char *format, ...);
def sprintf : TargetLibCall<"sprintf", Int, [Ptr, Ptr, Ellip]>;

/// double sqrt(double x);
def sqrt : TargetLibCall<"sqrt", Dbl, [Dbl]>;

/// float sqrtf(float x);
def sqrtf : TargetLibCall<"sqrtf", Flt, [Flt]>;

/// long double sqrtl(long double x);
def sqrtl : TargetLibCall<"sqrtl", LDbl, [LDbl]>;

/// int sscanf(const char *s, const char *format, ... );
def sscanf : TargetLibCall<"sscanf", Int, [Ptr, Ptr, Ellip]>;

/// int stat(const char *path, struct stat *buf);
def stat : TargetLibCall<"stat", Int, [Ptr, Ptr]>;

/// int stat64(const char *path, struct stat64 *buf);
def stat64 : TargetLibCall<"stat64", Int, [Ptr, Ptr]>;

/// int statvfs(const char *path, struct statvfs *buf);
def statvfs : TargetLibCall<"statvfs", Int, [Ptr, Ptr]>;

/// int statvfs64(const char *path, struct statvfs64 *buf)
def statvfs64 : TargetLibCall<"statvfs64", Int, [Ptr, Ptr]>;

/// char *stpcpy(char *s1, const char *s2);
def stpcpy : TargetLibCall<"stpcpy", Ptr, [Ptr, Ptr]>;

/// char *stpncpy(char *s1, const char *s2, size_t n);
def stpncpy : TargetLibCall<"stpncpy", Ptr, [Ptr, Ptr, SizeT]>;

/// int strcasecmp(const char *s1, const char *s2);
def strcasecmp : TargetLibCall<"strcasecmp", Int, [Ptr, Ptr]>;

/// char *strcat(char *s1, const char *s2);
def strcat : TargetLibCall<"strcat", Ptr, [Ptr, Ptr]>;

/// char *strchr(const char *s, int c);
def strchr : TargetLibCall<"strchr", Ptr, [Ptr, Int]>;

/// int strcmp(const char *s1, const char *s2);
def strcmp : TargetLibCall<"strcmp", Int, [Ptr, Ptr]>;

/// int strcoll(const char *s1, const char *s2);
def strcoll : TargetLibCall<"strcoll", Int, [Ptr, Ptr]>;

/// char *strcpy(char *s1, const char *s2);
def strcpy : TargetLibCall<"strcpy", Ptr, [Ptr, Ptr]>;

/// size_t strcspn(const char *s1, const char *s2);
def strcspn : TargetLibCall<"strcspn", SizeT, [Ptr, Ptr]>;

/// char *strdup(const char *s1);
def strdup : TargetLibCall<"strdup", Ptr, [Ptr]>;

/// size_t strlcat(char *dst, const char *src, size_t size);
def strlcat : TargetLibCall<"strlcat", SizeT, [Ptr, Ptr, SizeT]>;

/// size_t strlcpy(char *dst, const char *src, size_t size);
def strlcpy : TargetLibCall<"strlcpy", SizeT, [Ptr, Ptr, SizeT]>;

/// size_t strlen(const char *s);
def strlen : TargetLibCall<"strlen", SizeT, [Ptr]>;

/// int strncasecmp(const char *s1, const char *s2, size_t n);
def strncasecmp : TargetLibCall<"strncasecmp", Int, [Ptr, Ptr, SizeT]>;

/// char *strncat(char *s1, const char *s2, size_t n);
def strncat : TargetLibCall<"strncat", Ptr, [Ptr, Ptr, SizeT]>;

/// int strncmp(const char *s1, const char *s2, size_t n);
def strncmp : TargetLibCall<"strncmp", Int, [Ptr, Ptr, SizeT]>;

/// char *strncpy(char *s1, const char *s2, size_t n);
def strncpy : TargetLibCall<"strncpy", Ptr, [Ptr, Ptr, SizeT]>;

/// char *strndup(const char *s1, size_t n);
def strndup : TargetLibCall<"strndup", Ptr, [Ptr, SizeT]>;

/// size_t strnlen(const char *s, size_t maxlen);
def strnlen : TargetLibCall<"strnlen", SizeT, [Ptr, SizeT]>;

/// char *strpbrk(const char *s1, const char *s2);
def strpbrk : TargetLibCall<"strpbrk", Ptr, [Ptr, Ptr]>;

/// char *strrchr(const char *s, int c);
def strrchr : TargetLibCall<"strrchr", Ptr, [Ptr, Int]>;

/// size_t strspn(const char *s1, const char *s2);
def strspn : TargetLibCall<"strspn", SizeT, [Ptr, Ptr]>;

/// char *strstr(const char *s1, const char *s2);
def strstr : TargetLibCall<"strstr", Ptr, [Ptr, Ptr]>;

/// double strtod(const char *nptr, char **endptr);
def strtod : TargetLibCall<"strtod", Dbl, [Ptr, Ptr]>;

/// float strtof(const char *nptr, char **endptr);
def strtof : TargetLibCall<"strtof", Flt, [Ptr, Ptr]>;

/// char *strtok(char *s1, const char *s2);
def strtok : TargetLibCall<"strtok", Ptr, [Ptr, Ptr]>;

/// char *strtok_r(char *s, const char *sep, char **lasts);
def strtok_r : TargetLibCall<"strtok_r", Ptr, [Ptr, Ptr, Ptr]>;

/// long int strtol(const char *nptr, char **endptr, int base);
def strtol : TargetLibCall<"strtol", Long, [Ptr, Ptr, Int]>;

/// long double strtold(const char *nptr, char **endptr);
def strtold : TargetLibCall<"strtold", LDbl, [Ptr, Ptr]>;

/// long long int strtoll(const char *nptr, char **endptr, int base);
def strtoll : TargetLibCall<"strtoll", LLong, [Ptr, Ptr, Int]>;

/// unsigned long int strtoul(const char *nptr, char **endptr, int base);
def strtoul : TargetLibCall<"strtoul", Long, [Ptr, Ptr, Int]>;

/// unsigned long long int strtoull(const char *nptr, char **endptr, int base);
def strtoull : TargetLibCall<"strtoull", LLong, [Ptr, Ptr, Int]>;

/// size_t strxfrm(char *s1, const char *s2, size_t n);
def strxfrm : TargetLibCall<"strxfrm", SizeT, [Ptr, Ptr, SizeT]>;

/// int system(const char *command);
def system : TargetLibCall<"system", Int, [Ptr]>;

/// double tan(double x);
def tan : TargetLibCall<"tan", Dbl, [Dbl]>;

/// float tanf(float x);
def tanf : TargetLibCall<"tanf", Flt, [Flt]>;

/// double tanh(double x);
def tanh : TargetLibCall<"tanh", Dbl, [Dbl]>;

/// float tanhf(float x);
def tanhf : TargetLibCall<"tanhf", Flt, [Flt]>;

/// long double tanhl(long double x);
def tanhl : TargetLibCall<"tanhl", LDbl, [LDbl]>;

/// long double tanl(long double x);
def tanl : TargetLibCall<"tanl", LDbl, [LDbl]>;

/// clock_t times(struct tms *buffer);
def times : TargetLibCall<"times", IntPlus, [Ptr]>;

/// FILE *tmpfile(void);
def tmpfile : TargetLibCall<"tmpfile", Ptr, []>;

/// FILE *tmpfile64(void)
def tmpfile64 : TargetLibCall<"tmpfile64", Ptr, []>;

/// int toascii(int c);
def toascii : TargetLibCall<"toascii", Int, [Int]>;

/// double trunc(double x);
def trunc : TargetLibCall<"trunc", Dbl, [Dbl]>;

/// float truncf(float x);
def truncf : TargetLibCall<"truncf", Flt, [Flt]>;

/// long double truncl(long double x);
def truncl : TargetLibCall<"truncl", LDbl, [LDbl]>;

/// int uname(struct utsname *name);
def uname : TargetLibCall<"uname", Int, [Ptr]>;

/// int ungetc(int c, FILE *stream);
def ungetc : TargetLibCall<"ungetc", Int, [Int, Ptr]>;

/// int unlink(const char *path);
def unlink : TargetLibCall<"unlink", Int, [Ptr]>;

/// int unsetenv(const char *name);
def unsetenv : TargetLibCall<"unsetenv", Int, [Ptr]>;

/// int utime(const char *path, const struct utimbuf *times);
def utime : TargetLibCall<"utime", Int, [Ptr, Ptr]>;

/// int utimes(const char *path, const struct timeval times[2]);
def utimes : TargetLibCall<"utimes", Int, [Ptr, Ptr]>;

/// void *valloc(size_t size);
def valloc : TargetLibCall<"valloc", Ptr, [SizeT]>;

/// void *vec_calloc(size_t count, size_t size);
def vec_calloc : TargetLibCall<"vec_calloc", Ptr, [SizeT, SizeT]>;

/// void vec_free(void *ptr);
def vec_free : TargetLibCall<"vec_free", Void, [Ptr]>;

/// void *vec_malloc(size_t size);
def vec_malloc : TargetLibCall<"vec_malloc", Ptr, [SizeT]>;

/// void *vec_realloc(void *ptr, size_t size);
def vec_realloc : TargetLibCall<"vec_realloc", Ptr, [Ptr, SizeT]>;

/// int vfprintf(FILE *stream, const char *format, va_list ap);
def vfprintf : TargetLibCall<"vfprintf", Int, [Ptr, Ptr, Ptr]>;

/// int vfscanf(FILE *stream, const char *format, va_list arg);
def vfscanf : TargetLibCall<"vfscanf", Int, [Ptr, Ptr, Ptr]>;

/// int vprintf(const char *restrict format, va_list ap);
def vprintf : TargetLibCall<"vprintf", Int, [Ptr, Ptr]>;

/// int vscanf(const char *format, va_list arg);
def vscanf : TargetLibCall<"vscanf", Int, [Ptr, Ptr]>;

/// int vsnprintf(char *s, size_t n, const char *format, va_list ap);
def vsnprintf : TargetLibCall<"vsnprintf", Int, [Ptr, SizeT, Ptr, Ptr]>;

/// int vsprintf(char *s, const char *format, va_list ap);
def vsprintf : TargetLibCall<"vsprintf", Int, [Ptr, Ptr, Ptr]>;

/// int vsscanf(const char *s, const char *format, va_list arg);
def vsscanf : TargetLibCall<"vsscanf", Int, [Ptr, Ptr, Ptr]>;

/// size_t wcslen (const wchar_t* wcs);
def wcslen : TargetLibCall<"wcslen", SizeT, [Ptr]>;

/// ssize_t write(int fildes, const void *buf, size_t nbyte);
def write : TargetLibCall<"write", SSizeT, [Int, Ptr, SizeT]>;
