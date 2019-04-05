//==----------- generic_type_traits - SYCL type traits ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/pointers.hpp>
#include <CL/sycl/types.hpp>

#include <type_traits>
#include <limits>

// TODO Delete when half type will supported by SYCL Runtime
#define __HALF_NO_ENABLED

namespace cl {
namespace sycl {
namespace detail {

template <typename... Types> struct type_list;

template <typename H, typename... T> struct type_list<H, T...> {
  using head = H;
  using tail = type_list<T...>;
};

template <> struct type_list<> {};

template <typename T, typename TL>
struct is_contained
    : std::conditional<std::is_same<typename std::remove_cv<T>::type,
                                    typename TL::head>::value,
                       std::true_type,
                       is_contained<T, typename TL::tail>>::type {};

template <typename T> struct is_contained<T, type_list<>> : std::false_type {};

// floatn: float2, float3, float4, float8, float16
template <typename T>
using is_floatn = typename is_contained<
    T, type_list<cl_float2, cl_float3, cl_float4, cl_float8, cl_float16>>::type;

// genfloatf: float, floatn
template <typename T>
using is_genfloatf =
    std::integral_constant<bool, is_contained<T, type_list<cl_float>>::value ||
                                     is_floatn<T>::value>;

// doublen: double2, double3, double4, double8, double16
template <typename T>
using is_doublen =
    typename is_contained<T, type_list<cl_double2, cl_double3, cl_double4,
                                       cl_double8, cl_double16>>::type;

// genfloatd: double, doublen
template <typename T>
using is_genfloatd =
    std::integral_constant<bool, is_contained<T, type_list<cl_double>>::value ||
                                     is_doublen<T>::value>;

#ifndef __HALF_NO_ENABLED
// halfn: half2, half3, half4, half8, half16
template <typename T>
using is_halfn = typename is_contained<
    T, type_list<cl_half2, cl_half3, cl_half4, cl_half8, cl_half16>>::type;

// genfloath: half, halfn
template <typename T>
using is_genfloath =
    std::integral_constant<bool, is_contained<T, type_list<cl_half>>::value ||
                                     is_halfn<T>::value>;
#endif

// genfloat: genfloatf, genfloatd, genfloath
template <typename T>
using is_genfloat = std::integral_constant<bool, is_genfloatf<T>::value ||
                                                     is_genfloatd<T>::value
#ifndef __HALF_NO_ENABLED
                                                     || is_genfloath<T>::value
#endif
                                           >;

// sgenfloat: float, double, half
template <typename T>
using is_sgenfloat = typename is_contained<T, type_list<cl_float, cl_double
#ifndef __HALF_NO_ENABLED
                                                        ,
                                                        cl_half
#endif
                                                        >>::type;

// vgenfloat: floatn, doublen, halfn
template <typename T>
using is_vgenfloat =
    std::integral_constant<bool, is_floatn<T>::value || is_doublen<T>::value
#ifndef __HALF_NO_ENABLED
                                     || is_halfn<T>::value
#endif
                           >;

// gengeofloat: float, float2, float3, float4
template <typename T>
using is_gengeofloat = typename is_contained<
    T, type_list<cl_float, cl_float2, cl_float3, cl_float4>>::type;

// gengeodouble: double, double2, double3, double4
template <typename T>
using is_gengeodouble = typename is_contained<
    T, type_list<cl_double, cl_double2, cl_double3, cl_double4>>::type;

#ifndef __HALF_NO_ENABLED
// gengeohalf: half, half2, half3, half4
template <typename T>
using is_gengeohalf = typename is_contained<
    T, type_list<cl_half, cl_half2, cl_half3, cl_half4>>::type;
#endif

// gengeofloat: float, float2, float3, float4
template <typename T>
using is_vgengeofloat =
    typename is_contained<T, type_list<cl_float2, cl_float3, cl_float4>>::type;

// gengeodouble: double, double2, double3, double4
template <typename T>
using is_vgengeodouble =
    typename is_contained<T,
                          type_list<cl_double2, cl_double3, cl_double4>>::type;

#ifndef __HALF_NO_ENABLED
// gengeohalf: half2, half3, half4
template <typename T>
using is_vgengeohalf =
    typename is_contained<T, type_list<cl_half2, cl_half3, cl_half4>>::type;
#endif

// sgengeo: float, double, half
template <typename T>
using is_sgengeo =
    std::integral_constant<bool, is_contained<T, type_list<cl_float, cl_double
#ifndef __HALF_NO_ENABLED
                                                           ,
                                                           cl_half
#endif
                                                           >>::value>;

// vgengeo: vgengeofloat, vgengeodouble, vgengeohalf
template <typename T>
using is_vgengeo = std::integral_constant<bool, is_vgengeofloat<T>::value ||
                                                    is_vgengeodouble<T>::value
#ifndef __HALF_NO_ENABLED
                                                    || is_vgengeohalf<T>::value
#endif
                                          >;

// gencrossfloat:  float3, float4
template <typename T>
using is_gencrossfloat =
    typename is_contained<T, type_list<cl_float3, cl_float4>>::type;

// gencrossdouble: double3, double4
template <typename T>
using is_gencrossdouble =
    typename is_contained<T, type_list<cl_double3, cl_double4>>::type;

#ifndef __HALF_NO_ENABLED
// gencrosshalf: half3, half4
template <typename T>
using is_gencrosshalf =
    typename is_contained<T, type_list<cl_half3, cl_half4>>::type;
#endif

// gencross: gencrossfloat, gencrossdouble, gencrosshalf
template <typename T>
using is_gencross =
    std::integral_constant<bool, is_gencrossfloat<T>::value ||
                                     is_gencrossdouble<T>::value
#ifndef __HALF_NO_ENABLED
                                     || is_gencrosshalf<T>::value
#endif
                           >;

// charn: char2, char3, char4, char8, char16
template <typename T>
using is_charn = typename is_contained<
    T, type_list<cl_char2, cl_char3, cl_char4, cl_char8, cl_char16>>::type;

// scharn: schar2, schar3, schar4, schar8, schar16
template <typename T>
using is_scharn = typename is_contained<
    T, type_list<cl_schar2, cl_schar3, cl_schar4, cl_schar8, cl_schar16>>::type;

// ucharn: uchar2, uchar3, uchar4, uchar8, uchar16
template <typename T>
using is_ucharn = typename is_contained<
    T, type_list<cl_uchar2, cl_uchar3, cl_uchar4, cl_uchar8, cl_uchar16>>::type;

// igenchar: signed char, scharn
template <typename T>
using is_igenchar =
    std::integral_constant<bool, is_contained<T, type_list<cl_schar>>::value ||
                                     is_scharn<T>::value>;

// ugenchar: unsigned char, ucharn
template <typename T>
using is_ugenchar =
    std::integral_constant<bool, is_contained<T, type_list<cl_uchar>>::value ||
                                     is_ucharn<T>::value>;

// genchar: char, charn, igenchar, ugenchar
template <typename T>
using is_genchar = std::integral_constant<
    bool, is_contained<T, type_list<cl_char>>::value || is_charn<T>::value ||
              is_igenchar<T>::value || is_ugenchar<T>::value>;

// shortn: short2, short3, short4, short8, short16
template <typename T>
using is_shortn = typename is_contained<
    T, type_list<cl_short2, cl_short3, cl_short4, cl_short8, cl_short16>>::type;

// genshort: short, shortn
template <typename T>
using is_genshort =
    std::integral_constant<bool, is_contained<T, type_list<cl_short>>::value ||
                                     is_shortn<T>::value>;

// ushortn: ushort2, ushort3, ushort4, ushort8, ushort16
template <typename T>
using is_ushortn =
    typename is_contained<T, type_list<cl_ushort2, cl_ushort3, cl_ushort4,
                                       cl_ushort8, cl_ushort16>>::type;

// genushort: ushort, ushortn
template <typename T>
using is_ugenshort =
    std::integral_constant<bool, is_contained<T, type_list<cl_ushort>>::value ||
                                     is_ushortn<T>::value>;

// uintn: uint2, uint3, uint4, uint8, uint16
template <typename T>
using is_uintn = typename is_contained<
    T, type_list<cl_uint2, cl_uint3, cl_uint4, cl_uint8, cl_uint16>>::type;

// ugenint: unsigned int, uintn
template <typename T>
using is_ugenint =
    std::integral_constant<bool, is_contained<T, type_list<cl_uint>>::value ||
                                     is_uintn<T>::value>;

// intn: int2, int3, int4, int8, int16
template <typename T>
using is_intn = typename is_contained<
    T, type_list<cl_int2, cl_int3, cl_int4, cl_int8, cl_int16>>::type;

// genint: int, intn
template <typename T>
using is_genint =
    std::integral_constant<bool, is_contained<T, type_list<cl_int>>::value ||
                                     is_intn<T>::value>;

// ulongn: ulong2, ulong3, ulong4, ulong8,ulong16
template <typename T>
using is_ulongn = typename is_contained<
    T, type_list<cl_ulong2, cl_ulong3, cl_ulong4, cl_ulong8, cl_ulong16>>::type;

// ugenlong: unsigned long int, ulongn
template <typename T>
using is_ugenlong =
    std::integral_constant<bool, is_contained<T, type_list<cl_ulong>>::value ||
                                     is_ulongn<T>::value>;

// longn: long2, long3, long4, long8, long16
template <typename T>
using is_longn = typename is_contained<
    T, type_list<cl_long2, cl_long3, cl_long4, cl_long8, cl_long16>>::type;

// genlong: long int, longn
template <typename T>
using is_genlong =
    std::integral_constant<bool, is_contained<T, type_list<cl_long>>::value ||
                                     is_longn<T>::value>;

// ulonglongn: ulonglong2, ulonglong3, ulonglong4,ulonglong8, ulonglong16
template <typename T>
using is_ulonglongn =
    typename is_contained<T, type_list<ulonglong2, ulonglong3, ulonglong4,
                                       ulonglong8, ulonglong16>>::type;

// ugenlonglong: unsigned long long int, ulonglongn
template <typename T>
using is_ugenlonglong =
    std::integral_constant<bool, is_contained<T, type_list<ulonglong>>::value ||
                                     is_ulonglongn<T>::value>;

// longlongn: longlong2, longlong3, longlong4,longlong8, longlong16
template <typename T>
using is_longlongn = typename is_contained<
    T, type_list<longlong2, longlong3, longlong4, longlong8, longlong16>>::type;

// genlonglong: long long int, longlongn
template <typename T>
using is_genlonglong =
    std::integral_constant<bool, is_contained<T, type_list<longlong>>::value ||
                                     is_longlongn<T>::value>;

// igenlonginteger: genlong, genlonglong
template <typename T>
using is_igenlonginteger =
    std::integral_constant<bool,
                           is_genlong<T>::value || is_genlonglong<T>::value>;

// ugenlonginteger ugenlong, ugenlonglong
template <typename T>
using is_ugenlonginteger =
    std::integral_constant<bool,
                           is_ugenlong<T>::value || is_ugenlonglong<T>::value>;

// geninteger: genchar, genshort, ugenshort, genint, ugenint, igenlonginteger,
// ugenlonginteger
template <typename T>
using is_geninteger = std::integral_constant<
    bool, is_genchar<T>::value || is_genshort<T>::value ||
              is_ugenshort<T>::value || is_genint<T>::value ||
              is_ugenint<T>::value || is_igenlonginteger<T>::value ||
              is_ugenlonginteger<T>::value>;

// igeninteger: igenchar, genshort, genint, igenlonginteger
template <typename T>
using is_igeninteger = std::integral_constant<
    bool, is_igenchar<T>::value || is_genshort<T>::value ||
              is_genint<T>::value || is_igenlonginteger<T>::value>;

// ugeninteger: ugenchar, ugenshort, ugenint, ugenlonginteger
template <typename T>
using is_ugeninteger = std::integral_constant<
    bool, is_ugenchar<T>::value || is_ugenshort<T>::value ||
              is_ugenint<T>::value || is_ugenlonginteger<T>::value>;

// sgeninteger: char, signed char, unsigned char, short, unsigned short, int,
// unsigned int, long int, unsigned long int, long long int, unsigned long long
// int
template <typename T>
using is_sgeninteger = typename is_contained<
    T, type_list<cl_char, cl_schar, cl_uchar, cl_short, cl_ushort, cl_int,
                 cl_uint, cl_long, cl_ulong, longlong, ulonglong>>::type;

// vgeninteger: charn, scharn, ucharn, shortn, ushortn, intn, uintn, longn,
// ulongn, longlongn, ulonglongn
template <typename T>
using is_vgeninteger = std::integral_constant<
    bool, is_charn<T>::value || is_scharn<T>::value || is_ucharn<T>::value ||
              is_shortn<T>::value || is_ushortn<T>::value ||
              is_intn<T>::value || is_uintn<T>::value || is_longn<T>::value ||
              is_ulongn<T>::value || is_longlongn<T>::value ||
              is_ulonglongn<T>::value>;

// sigeninteger: char, signed char, short, int, long int, , long long int
template <typename T>
using is_sigeninteger = typename is_contained<
    T, type_list<cl_char, cl_schar, cl_short, cl_int, cl_long, longlong>>::type;

// sugeninteger: unsigned char, unsigned short,  unsigned int, unsigned long
// int, unsigned long long int
template <typename T>
using is_sugeninteger = typename is_contained<
    T, type_list<cl_uchar, cl_ushort, cl_uint, cl_ulong, ulonglong>>::type;

// vigeninteger: charn, scharn, shortn, intn, longn, longlongn
template <typename T>
using is_vigeninteger =
    std::integral_constant<bool, is_charn<T>::value || is_scharn<T>::value ||
                                     is_shortn<T>::value || is_intn<T>::value ||
                                     is_longn<T>::value ||
                                     is_longlongn<T>::value>;

// vugeninteger: ucharn, ushortn, uintn, ulongn, ulonglongn
template <typename T>
using is_vugeninteger = std::integral_constant<
    bool, is_ucharn<T>::value || is_ushortn<T>::value || is_uintn<T>::value ||
              is_ulongn<T>::value || is_ulonglongn<T>::value>;

// gentype: genfloat, geninteger
template <typename T>
using is_gentype = std::integral_constant<bool, is_genfloat<T>::value ||
                                                    is_geninteger<T>::value>;

// forward declarations
template <typename T> class TryToGetElementType;

// genintegerNbit All types within geninteger whose base type are N bits in
// size, where N = 8, 16, 32, 64
template <typename T, int N>
using is_igenintegerNbit = typename std::integral_constant<
    bool, is_igeninteger<T>::value &&
              (sizeof(typename TryToGetElementType<T>::type) == N)>;

// igeninteger8bit All types within igeninteger whose base type are 8 bits in
// size
template <typename T> using is_igeninteger8bit = is_igenintegerNbit<T, 1>;

// igeninteger16bit All types within igeninteger whose base type are 16 bits in
// size
template <typename T> using is_igeninteger16bit = is_igenintegerNbit<T, 2>;

// igeninteger32bit All types within igeninteger whose base type are 32 bits in
// size
template <typename T> using is_igeninteger32bit = is_igenintegerNbit<T, 4>;

// igeninteger64bit All types within igeninteger whose base type are 64 bits in
// size
template <typename T> using is_igeninteger64bit = is_igenintegerNbit<T, 8>;

// ugenintegerNbit All types within ugeninteger whose base type are N bits in
// size, where N = 8, 16, 32, 64.
template <typename T, int N>
using is_ugenintegerNbit = typename std::integral_constant<
    bool, is_ugeninteger<T>::value &&
              (sizeof(typename TryToGetElementType<T>::type) == N)>;

// ugeninteger8bit All types within ugeninteger whose base type are 8 bits in
// size
template <typename T> using is_ugeninteger8bit = is_ugenintegerNbit<T, 1>;

// ugeninteger16bit All types within ugeninteger whose base type are 16 bits in
// size
template <typename T> using is_ugeninteger16bit = is_ugenintegerNbit<T, 2>;

// ugeninteger32bit All types within ugeninteger whose base type are 32 bits in
// size
template <typename T> using is_ugeninteger32bit = is_ugenintegerNbit<T, 4>;

// ugeninteger64bit All types within ugeninteger whose base type are 64 bits in
// size
template <typename T> using is_ugeninteger64bit = is_ugenintegerNbit<T, 8>;

// genintegerNbit All types within geninteger whose base type are N bits in
// size, where N = 8, 16, 32, 64.
template <typename T, int N>
using is_genintegerNbit = typename std::integral_constant<
    bool, is_geninteger<T>::value &&
              (sizeof(typename TryToGetElementType<T>::type) == N)>;

// geninteger8bit All types within geninteger whose base type are 8 bits in size
template <typename T> using is_geninteger8bit = is_genintegerNbit<T, 1>;

// geninteger16bit All types within geninteger whose base type are 16 bits in
// size
template <typename T> using is_geninteger16bit = is_genintegerNbit<T, 2>;

// geninteger32bit All types within geninteger whose base type are 32 bits in
// size
template <typename T> using is_geninteger32bit = is_genintegerNbit<T, 4>;

// geninteger64bit All types within geninteger whose base type are 64 bits in
// size
template <typename T> using is_geninteger64bit = is_genintegerNbit<T, 8>;

template <class P, typename T>
using is_MultiPtrOfGLR =
    std::integral_constant<bool, std::is_same<P, global_ptr<T>>::value ||
                                     std::is_same<P, local_ptr<T>>::value ||
                                     std::is_same<P, private_ptr<T>>::value>;

// genintptr All permutations of multi_ptr<dataT, addressSpace> where dataT is
// all types within genint and addressSpace is
// access::address_space::global_space, access::address_space::local_space and
// access::address_space::private_space
template <class P>
using is_genintptr =
    std::integral_constant<bool, is_MultiPtrOfGLR<P, cl_int>::value ||
                                     is_MultiPtrOfGLR<P, cl_int2>::value ||
                                     is_MultiPtrOfGLR<P, cl_int3>::value ||
                                     is_MultiPtrOfGLR<P, cl_int4>::value ||
                                     is_MultiPtrOfGLR<P, cl_int8>::value ||
                                     is_MultiPtrOfGLR<P, cl_int16>::value>;

// genfloatptr All permutations of multi_ptr<dataT, addressSpace> where dataT is
// all types within genfloat and addressSpace is
// access::address_space::global_space, access::address_space::local_space and
// access::address_space::private_space
template <class P>
using is_genfloatptr =
    std::integral_constant<bool, is_MultiPtrOfGLR<P, cl_float>::value ||
                                     is_MultiPtrOfGLR<P, cl_float2>::value ||
                                     is_MultiPtrOfGLR<P, cl_float3>::value ||
                                     is_MultiPtrOfGLR<P, cl_float4>::value ||
                                     is_MultiPtrOfGLR<P, cl_float8>::value ||
                                     is_MultiPtrOfGLR<P, cl_float16>::value ||
#ifndef __HALF_NO_ENABLED
                                     is_MultiPtrOfGLR<P, cl_half>::value ||
                                     is_MultiPtrOfGLR<P, cl_half2>::value ||
                                     is_MultiPtrOfGLR<P, cl_half3>::value ||
                                     is_MultiPtrOfGLR<P, cl_half4>::value ||
                                     is_MultiPtrOfGLR<P, cl_half8>::value ||
                                     is_MultiPtrOfGLR<P, cl_half16>::value ||
#endif
                                     is_MultiPtrOfGLR<P, cl_double>::value ||
                                     is_MultiPtrOfGLR<P, cl_double2>::value ||
                                     is_MultiPtrOfGLR<P, cl_double3>::value ||
                                     is_MultiPtrOfGLR<P, cl_double4>::value ||
                                     is_MultiPtrOfGLR<P, cl_double8>::value ||
                                     is_MultiPtrOfGLR<P, cl_double16>::value>;

// Used for nan built-in
template <typename T> struct unsign_integral_to_float_point;
template <> struct unsign_integral_to_float_point<cl_uint> {
  using type = cl_float;
};
template <> struct unsign_integral_to_float_point<cl_uint2> {
  using type = cl_float2;
};
template <> struct unsign_integral_to_float_point<cl_uint3> {
  using type = cl_float3;
};
template <> struct unsign_integral_to_float_point<cl_uint4> {
  using type = cl_float4;
};
template <> struct unsign_integral_to_float_point<cl_uint8> {
  using type = cl_float8;
};
template <> struct unsign_integral_to_float_point<cl_uint16> {
  using type = cl_float16;
};

#ifndef __HALF_NO_ENABLED
template <> struct unsign_integral_to_float_point<cl_ushort> {
  using type = cl_half;
};
template <> struct unsign_integral_to_float_point<cl_ushort2> {
  using type = cl_half2;
};
template <> struct unsign_integral_to_float_point<cl_ushort3> {
  using type = cl_half3;
};
template <> struct unsign_integral_to_float_point<cl_ushort4> {
  using type = cl_half4;
};
template <> struct unsign_integral_to_float_point<cl_ushort8> {
  using type = cl_half8;
};
template <> struct unsign_integral_to_float_point<cl_ushort16> {
  using type = cl_half16;
};
#endif

template <> struct unsign_integral_to_float_point<cl_ulong> {
  using type = cl_double;
};
template <> struct unsign_integral_to_float_point<cl_ulong2> {
  using type = cl_double2;
};
template <> struct unsign_integral_to_float_point<cl_ulong3> {
  using type = cl_double3;
};
template <> struct unsign_integral_to_float_point<cl_ulong4> {
  using type = cl_double4;
};
template <> struct unsign_integral_to_float_point<cl_ulong8> {
  using type = cl_double8;
};
template <> struct unsign_integral_to_float_point<cl_ulong16> {
  using type = cl_double16;
};

template <> struct unsign_integral_to_float_point<ulonglong> {
  using type = cl_double;
};
template <> struct unsign_integral_to_float_point<ulonglong2> {
  using type = cl_double2;
};
template <> struct unsign_integral_to_float_point<ulonglong3> {
  using type = cl_double3;
};
template <> struct unsign_integral_to_float_point<ulonglong4> {
  using type = cl_double4;
};
template <> struct unsign_integral_to_float_point<ulonglong8> {
  using type = cl_double8;
};
template <> struct unsign_integral_to_float_point<ulonglong16> {
  using type = cl_double16;
};

template <typename T>
using is_nan_type =
    std::integral_constant<bool, detail::is_ugenshort<T>::value ||
                                     detail::is_ugenint<T>::value ||
                                     detail::is_ugenlonginteger<T>::value>;

// Used for some relational functions
template <typename T> struct float_point_to_sign_integral;
template <> struct float_point_to_sign_integral<cl_float> {
  using type = cl_int;
};
template <> struct float_point_to_sign_integral<cl_float2> {
  using type = cl_int2;
};
template <> struct float_point_to_sign_integral<cl_float3> {
  using type = cl_int3;
};
template <> struct float_point_to_sign_integral<cl_float4> {
  using type = cl_int4;
};
template <> struct float_point_to_sign_integral<cl_float8> {
  using type = cl_int8;
};
template <> struct float_point_to_sign_integral<cl_float16> {
  using type = cl_int16;
};

#ifndef __HALF_NO_ENABLED
template <> struct float_point_to_sign_integral<cl_half> {
  using type = cl_int;
};
template <> struct float_point_to_sign_integral<cl_half2> {
  using type = cl_short2;
};
template <> struct float_point_to_sign_integral<cl_half3> {
  using type = cl_short3;
};
template <> struct float_point_to_sign_integral<cl_half4> {
  using type = cl_short4;
};
template <> struct float_point_to_sign_integral<cl_half8> {
  using type = cl_short8;
};
template <> struct float_point_to_sign_integral<cl_half16> {
  using type = cl_short16;
};
#endif

template <> struct float_point_to_sign_integral<cl_double> {
  using type = cl_int;
};
template <> struct float_point_to_sign_integral<cl_double2> {
  using type = cl_long2;
};
template <> struct float_point_to_sign_integral<cl_double3> {
  using type = cl_long3;
};
template <> struct float_point_to_sign_integral<cl_double4> {
  using type = cl_long4;
};
template <> struct float_point_to_sign_integral<cl_double8> {
  using type = cl_long8;
};
template <> struct float_point_to_sign_integral<cl_double16> {
  using type = cl_long16;
};

// Used for ilogb built-in
template <typename T> struct float_point_to_int;
template <> struct float_point_to_int<cl_float> { using type = cl_int; };
template <> struct float_point_to_int<cl_float2> { using type = cl_int2; };
template <> struct float_point_to_int<cl_float3> { using type = cl_int3; };
template <> struct float_point_to_int<cl_float4> { using type = cl_int4; };
template <> struct float_point_to_int<cl_float8> { using type = cl_int8; };
template <> struct float_point_to_int<cl_float16> { using type = cl_int16; };
#ifndef __HALF_NO_ENABLED
template <> struct float_point_to_int<cl_half> { using type = cl_int; };
template <> struct float_point_to_int<cl_half2> { using type = cl_int2; };
template <> struct float_point_to_int<cl_half3> { using type = cl_int3; };
template <> struct float_point_to_int<cl_half4> { using type = cl_int4; };
template <> struct float_point_to_int<cl_half8> { using type = cl_int8; };
template <> struct float_point_to_int<cl_half16> { using type = cl_int16; };
#endif
template <> struct float_point_to_int<cl_double> { using type = cl_int; };
template <> struct float_point_to_int<cl_double2> { using type = cl_int2; };
template <> struct float_point_to_int<cl_double3> { using type = cl_int3; };
template <> struct float_point_to_int<cl_double4> { using type = cl_int4; };
template <> struct float_point_to_int<cl_double8> { using type = cl_int8; };
template <> struct float_point_to_int<cl_double16> { using type = cl_int16; };

// Used for abs and abs_diff built-in
template <typename T> struct make_unsigned { using type = T; };

template <> struct make_unsigned<cl_char> { using type = cl_uchar; };
template <> struct make_unsigned<cl_char2> { using type = cl_uchar2; };
template <> struct make_unsigned<cl_char3> { using type = cl_uchar3; };
template <> struct make_unsigned<cl_char4> { using type = cl_uchar4; };
template <> struct make_unsigned<cl_char8> { using type = cl_uchar8; };
template <> struct make_unsigned<cl_char16> { using type = cl_uchar16; };

template <> struct make_unsigned<cl_short> { using type = cl_ushort; };
template <> struct make_unsigned<cl_short2> { using type = cl_ushort2; };
template <> struct make_unsigned<cl_short3> { using type = cl_ushort3; };
template <> struct make_unsigned<cl_short4> { using type = cl_ushort4; };
template <> struct make_unsigned<cl_short8> { using type = cl_ushort8; };
template <> struct make_unsigned<cl_short16> { using type = cl_ushort16; };

template <> struct make_unsigned<cl_int> { using type = cl_uint; };
template <> struct make_unsigned<cl_int2> { using type = cl_uint2; };
template <> struct make_unsigned<cl_int3> { using type = cl_uint3; };
template <> struct make_unsigned<cl_int4> { using type = cl_uint4; };
template <> struct make_unsigned<cl_int8> { using type = cl_uint8; };
template <> struct make_unsigned<cl_int16> { using type = cl_uint16; };

template <> struct make_unsigned<cl_long> { using type = cl_ulong; };
template <> struct make_unsigned<cl_long2> { using type = cl_ulong2; };
template <> struct make_unsigned<cl_long3> { using type = cl_ulong3; };
template <> struct make_unsigned<cl_long4> { using type = cl_ulong4; };
template <> struct make_unsigned<cl_long8> { using type = cl_ulong8; };
template <> struct make_unsigned<cl_long16> { using type = cl_ulong16; };

template <> struct make_unsigned<longlong> { using type = ulonglong; };
template <> struct make_unsigned<longlong2> { using type = ulonglong2; };
template <> struct make_unsigned<longlong3> { using type = ulonglong3; };
template <> struct make_unsigned<longlong4> { using type = ulonglong4; };
template <> struct make_unsigned<longlong8> { using type = ulonglong8; };
template <> struct make_unsigned<longlong16> { using type = ulonglong16; };

template <typename T> struct make_signed { using type = T; };

template <> struct make_signed<cl_uchar> { using type = cl_char; };
template <> struct make_signed<cl_uchar2> { using type = cl_char2; };
template <> struct make_signed<cl_uchar3> { using type = cl_char3; };
template <> struct make_signed<cl_uchar4> { using type = cl_char4; };
template <> struct make_signed<cl_uchar8> { using type = cl_char8; };
template <> struct make_signed<cl_uchar16> { using type = cl_char16; };

template <> struct make_signed<cl_ushort> { using type = cl_short; };
template <> struct make_signed<cl_ushort2> { using type = cl_short2; };
template <> struct make_signed<cl_ushort3> { using type = cl_short3; };
template <> struct make_signed<cl_ushort4> { using type = cl_short4; };
template <> struct make_signed<cl_ushort8> { using type = cl_short8; };
template <> struct make_signed<cl_ushort16> { using type = cl_short16; };

template <> struct make_signed<cl_uint> { using type = cl_int; };
template <> struct make_signed<cl_uint2> { using type = cl_int2; };
template <> struct make_signed<cl_uint3> { using type = cl_int3; };
template <> struct make_signed<cl_uint4> { using type = cl_int4; };
template <> struct make_signed<cl_uint8> { using type = cl_int8; };
template <> struct make_signed<cl_uint16> { using type = cl_int16; };

template <> struct make_signed<cl_ulong> { using type = cl_long; };
template <> struct make_signed<cl_ulong2> { using type = cl_long2; };
template <> struct make_signed<cl_ulong3> { using type = cl_long3; };
template <> struct make_signed<cl_ulong4> { using type = cl_long4; };
template <> struct make_signed<cl_ulong8> { using type = cl_long8; };
template <> struct make_signed<cl_ulong16> { using type = cl_long16; };

template <> struct make_signed<ulonglong> { using type = longlong; };
template <> struct make_signed<ulonglong2> { using type = longlong2; };
template <> struct make_signed<ulonglong3> { using type = longlong3; };
template <> struct make_signed<ulonglong4> { using type = longlong4; };
template <> struct make_signed<ulonglong8> { using type = longlong8; };
template <> struct make_signed<ulonglong16> { using type = longlong16; };

// Used for upsample built-in
// Bases on Table 4.93: Scalar data type aliases supported by SYCL
template <typename T> struct make_upper;

template <> struct make_upper<cl_char> { using type = cl_short; };
template <> struct make_upper<cl_char2> { using type = cl_short2; };
template <> struct make_upper<cl_char3> { using type = cl_short3; };
template <> struct make_upper<cl_char4> { using type = cl_short4; };
template <> struct make_upper<cl_char8> { using type = cl_short8; };
template <> struct make_upper<cl_char16> { using type = cl_short16; };

template <> struct make_upper<cl_uchar> { using type = cl_ushort; };
template <> struct make_upper<cl_uchar2> { using type = cl_ushort2; };
template <> struct make_upper<cl_uchar3> { using type = cl_ushort3; };
template <> struct make_upper<cl_uchar4> { using type = cl_ushort4; };
template <> struct make_upper<cl_uchar8> { using type = cl_ushort8; };
template <> struct make_upper<cl_uchar16> { using type = cl_ushort16; };

template <> struct make_upper<cl_short> { using type = cl_int; };
template <> struct make_upper<cl_short2> { using type = cl_int2; };
template <> struct make_upper<cl_short3> { using type = cl_int3; };
template <> struct make_upper<cl_short4> { using type = cl_int4; };
template <> struct make_upper<cl_short8> { using type = cl_int8; };
template <> struct make_upper<cl_short16> { using type = cl_int16; };

template <> struct make_upper<cl_ushort> { using type = cl_uint; };
template <> struct make_upper<cl_ushort2> { using type = cl_uint2; };
template <> struct make_upper<cl_ushort3> { using type = cl_uint3; };
template <> struct make_upper<cl_ushort4> { using type = cl_uint4; };
template <> struct make_upper<cl_ushort8> { using type = cl_uint8; };
template <> struct make_upper<cl_ushort16> { using type = cl_uint16; };

template <> struct make_upper<cl_int> { using type = cl_long; };
template <> struct make_upper<cl_int2> { using type = cl_long2; };
template <> struct make_upper<cl_int3> { using type = cl_long3; };
template <> struct make_upper<cl_int4> { using type = cl_long4; };
template <> struct make_upper<cl_int8> { using type = cl_long8; };
template <> struct make_upper<cl_int16> { using type = cl_long16; };

template <> struct make_upper<cl_uint> { using type = cl_ulong; };
template <> struct make_upper<cl_uint2> { using type = cl_ulong2; };
template <> struct make_upper<cl_uint3> { using type = cl_ulong3; };
template <> struct make_upper<cl_uint4> { using type = cl_ulong4; };
template <> struct make_upper<cl_uint8> { using type = cl_ulong8; };
template <> struct make_upper<cl_uint16> { using type = cl_ulong16; };

template <> struct make_upper<cl_long> { using type = longlong; };
template <> struct make_upper<cl_long2> { using type = longlong2; };
template <> struct make_upper<cl_long3> { using type = longlong3; };
template <> struct make_upper<cl_long4> { using type = longlong4; };
template <> struct make_upper<cl_long8> { using type = longlong8; };
template <> struct make_upper<cl_long16> { using type = longlong16; };

template <> struct make_upper<cl_ulong> { using type = ulonglong; };
template <> struct make_upper<cl_ulong2> { using type = ulonglong2; };
template <> struct make_upper<cl_ulong3> { using type = ulonglong3; };
template <> struct make_upper<cl_ulong4> { using type = ulonglong4; };
template <> struct make_upper<cl_ulong8> { using type = ulonglong8; };
template <> struct make_upper<cl_ulong16> { using type = ulonglong16; };

// Try to get pointer_t, otherwise T
template <typename T> class TryToGetPointerT {
  static T check(...);
  template <typename A> static typename A::pointer_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get element_type, otherwise T
template <typename T> class TryToGetElementType {
  static T check(...);
  template <typename A> static typename A::element_type check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get vector_t, otherwise T
template <typename T> class TryToGetVectorT {
  static T check(...);
  template <typename A> static typename A::vector_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get pointer_t (if pointer_t indicates on the type with vector_t
// creates a pointer type on vector_t), otherwise T
template <typename T> class TryToGetPointerVecT {
  static T check(...);
  template <typename A>
  static typename PtrValueType<
      typename TryToGetVectorT<typename TryToGetElementType<A>::type>::type,
      A::address_space>::type *
  check(const A &);

public:
  using type = decltype(check(T()));
};

template <typename T, typename = typename std::enable_if<
                          TryToGetPointerT<T>::value, std::true_type>::type>
typename TryToGetPointerVecT<T>::type TryToGetPointer(T &t) {
  // TODO find the better way to get the pointer to underlying data from vec
  // class
  return reinterpret_cast<typename TryToGetPointerVecT<T>::type>(t.get());
}

template <typename T, typename = typename std::enable_if<
                          !TryToGetPointerT<T>::value, std::false_type>::type>
T TryToGetPointer(T &t) {
  return t;
}

// Converts T to OpenCL friendly
template <typename T>
using ConvertToOpenCLType = std::conditional<
    TryToGetVectorT<T>::value, typename TryToGetVectorT<T>::type,
    typename std::conditional<TryToGetPointerT<T>::value,
                              typename TryToGetPointerVecT<T>::type, T>::type>;

// Used for all,any and select relational built-in functions
template <typename T> inline constexpr T msbMask(T) {
  using UT = typename std::make_unsigned<T>::type;
  return T(UT(1) << (sizeof(T) * 8 - 1));
}

template <typename T> inline constexpr bool msbIsSet(const T x) {
  return (x & msbMask(x));
}

template <typename T>
using common_rel_ret_t = typename detail::float_point_to_sign_integral<T>::type;

// forward declaration
template <int N> struct Boolean;

// Try to get vector element count or 1 otherwise
template <typename T, typename Enable = void> class TryToGetNumElements;

template <typename T>
struct TryToGetNumElements<
    T, typename std::enable_if<TryToGetVectorT<T>::value>::type> {
  static constexpr int value = T::get_count();
};
template <typename T>
struct TryToGetNumElements<
    T, typename std::enable_if<!TryToGetVectorT<T>::value>::type> {
  static constexpr int value = 1;
};

// Used for relational comparison built-in functions
template <typename T> struct RelationalReturnType {
#ifdef __SYCL_DEVICE_ONLY__
  using type = Boolean<TryToGetNumElements<T>::value>;
#else
  using type = common_rel_ret_t<T>;
#endif
};

// Used for select built-in function
template <typename T> struct SelectWrapperTypeArgC {
#ifdef __SYCL_DEVICE_ONLY__
  using type = Boolean<TryToGetNumElements<T>::value>;
#else
  using type = T;
#endif
};

template <typename T>
using select_arg_c_t = typename SelectWrapperTypeArgC<T>::type;

template <typename T> using rel_ret_t = typename RelationalReturnType<T>::type;

// Used for any and all built-in functions
template <typename T> struct RelationalTestForSignBitType {
#ifdef __SYCL_DEVICE_ONLY__
  using return_type = detail::Boolean<1>;
  using argument_type = detail::Boolean<TryToGetNumElements<T>::value>;
#else
  using return_type = cl::sycl::cl_int;
  using argument_type = T;
#endif
};

template <typename T>
using rel_sign_bit_test_ret_t =
    typename RelationalTestForSignBitType<T>::return_type;

template <typename T>
using rel_sign_bit_test_arg_t =
    typename RelationalTestForSignBitType<T>::argument_type;

template <typename T, typename Enable = void> struct RelConverter;

template <typename T>
struct RelConverter<
    T, typename std::enable_if<TryToGetElementType<T>::value>::type> {
  static const int N = T::get_count();
#ifdef __SYCL_DEVICE_ONLY__
  using bool_t = typename Boolean<N>::vector_t;
  using ret_t = common_rel_ret_t<T>;
#else
  using bool_t = Boolean<N>;
  using ret_t = rel_ret_t<T>;
#endif

  static ret_t apply(bool_t value) {
#ifdef __SYCL_DEVICE_ONLY__
    typename ret_t::vector_t result(0);
    for (size_t I = 0; I < N; ++I) {
      result[I] = 0 - value[I];
    }
    return result;
#else
    return value;
#endif
  }
};

template <typename T>
struct RelConverter<
    T, typename std::enable_if<!TryToGetElementType<T>::value>::type> {
  using R = rel_ret_t<T>;
#ifdef __SYCL_DEVICE_ONLY__
  using value_t = bool;
#else
  using value_t = R;
#endif

  static R apply(value_t value) { return value; }
};

template <typename T> static constexpr T max_v() {
  return std::numeric_limits<T>::max();
}

template <typename T> static constexpr T min_v() {
  return std::numeric_limits<T>::min();
}

} // namespace detail
} // namespace sycl
} // namespace cl

#undef __HALF_NO_ENABLED
