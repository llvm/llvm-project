//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Regular use of <type_traits> is not compatible with CUDA
//  Adds aliases to unify the support
//  Also adds convience overloads like is_same_v so we don't have to wait for C++17

#ifndef BOOST_MATH_TOOLS_TYPE_TRAITS
#define BOOST_MATH_TOOLS_TYPE_TRAITS

#include <boost/math/tools/config.hpp>

#ifdef BOOST_MATH_ENABLE_CUDA

#include <cuda/std/type_traits>

namespace boost {
namespace math {

// Helper classes
using cuda::std::integral_constant;
using cuda::std::true_type;
using cuda::std::false_type;

// Primary type categories
using cuda::std::is_void;
using cuda::std::is_null_pointer;
using cuda::std::is_integral;
using cuda::std::is_floating_point;
using cuda::std::is_array;
using cuda::std::is_enum;
using cuda::std::is_union;
using cuda::std::is_class;
using cuda::std::is_function;
using cuda::std::is_pointer;
using cuda::std::is_lvalue_reference;
using cuda::std::is_rvalue_reference;
using cuda::std::is_member_object_pointer;
using cuda::std::is_member_function_pointer;

// Composite Type Categories
using cuda::std::is_fundamental;
using cuda::std::is_arithmetic;
using cuda::std::is_scalar;
using cuda::std::is_object;
using cuda::std::is_compound;
using cuda::std::is_reference;
using cuda::std::is_member_pointer;

// Type properties
using cuda::std::is_const;
using cuda::std::is_volatile;
using cuda::std::is_trivial;
using cuda::std::is_trivially_copyable;
using cuda::std::is_standard_layout;
using cuda::std::is_empty;
using cuda::std::is_polymorphic;
using cuda::std::is_abstract;
using cuda::std::is_final;
using cuda::std::is_signed;
using cuda::std::is_unsigned; 

// Supported Operations
using cuda::std::is_constructible;
using cuda::std::is_trivially_constructible;
using cuda::std::is_nothrow_constructible;

using cuda::std::is_default_constructible;
using cuda::std::is_trivially_default_constructible;
using cuda::std::is_nothrow_default_constructible;

using cuda::std::is_copy_constructible;
using cuda::std::is_trivially_copy_constructible;
using cuda::std::is_nothrow_copy_constructible;

using cuda::std::is_move_constructible;
using cuda::std::is_trivially_move_constructible;
using cuda::std::is_nothrow_move_constructible;

using cuda::std::is_assignable;
using cuda::std::is_trivially_assignable;
using cuda::std::is_nothrow_assignable;

using cuda::std::is_copy_assignable;
using cuda::std::is_trivially_copy_assignable;
using cuda::std::is_nothrow_copy_assignable;

using cuda::std::is_move_assignable;
using cuda::std::is_trivially_move_assignable;
using cuda::std::is_nothrow_move_assignable;

using cuda::std::is_destructible;
using cuda::std::is_trivially_destructible;
using cuda::std::is_nothrow_destructible;

using cuda::std::has_virtual_destructor;

// Property Queries
using cuda::std::alignment_of;
using cuda::std::rank;
using cuda::std::extent;

// Type Relationships
using cuda::std::is_same;
using cuda::std::is_base_of;
using cuda::std::is_convertible;

// Const-volatility specifiers
using cuda::std::remove_cv;
using cuda::std::remove_cv_t;
using cuda::std::remove_const;
using cuda::std::remove_const_t;
using cuda::std::remove_volatile;
using cuda::std::remove_volatile_t;
using cuda::std::add_cv;
using cuda::std::add_cv_t;
using cuda::std::add_const;
using cuda::std::add_const_t;
using cuda::std::add_volatile;
using cuda::std::add_volatile_t;

// References
using cuda::std::remove_reference;
using cuda::std::remove_reference_t;
using cuda::std::add_lvalue_reference;
using cuda::std::add_lvalue_reference_t;
using cuda::std::add_rvalue_reference;
using cuda::std::add_rvalue_reference_t;

// Pointers
using cuda::std::remove_pointer;
using cuda::std::remove_pointer_t;
using cuda::std::add_pointer;
using cuda::std::add_pointer_t;

// Sign Modifiers
using cuda::std::make_signed;
using cuda::std::make_signed_t;
using cuda::std::make_unsigned;
using cuda::std::make_unsigned_t;

// Arrays
using cuda::std::remove_extent;
using cuda::std::remove_extent_t;
using cuda::std::remove_all_extents;
using cuda::std::remove_all_extents_t;

// Misc transformations
using cuda::std::decay;
using cuda::std::decay_t;
using cuda::std::enable_if;
using cuda::std::enable_if_t;
using cuda::std::conditional;
using cuda::std::conditional_t;
using cuda::std::common_type;
using cuda::std::common_type_t;
using cuda::std::underlying_type;
using cuda::std::underlying_type_t;

#else // STD versions

#include <type_traits>

namespace boost {
namespace math {

// Helper classes
using std::integral_constant;
using std::true_type;
using std::false_type;

// Primary type categories
using std::is_void;
using std::is_null_pointer;
using std::is_integral;
using std::is_floating_point;
using std::is_array;
using std::is_enum;
using std::is_union;
using std::is_class;
using std::is_function;
using std::is_pointer;
using std::is_lvalue_reference;
using std::is_rvalue_reference;
using std::is_member_object_pointer;
using std::is_member_function_pointer;

// Composite Type Categories
using std::is_fundamental;
using std::is_arithmetic;
using std::is_scalar;
using std::is_object;
using std::is_compound;
using std::is_reference;
using std::is_member_pointer;

// Type properties
using std::is_const;
using std::is_volatile;
using std::is_trivial;
using std::is_trivially_copyable;
using std::is_standard_layout;
using std::is_empty;
using std::is_polymorphic;
using std::is_abstract;
using std::is_final;
using std::is_signed;
using std::is_unsigned; 

// Supported Operations
using std::is_constructible;
using std::is_trivially_constructible;
using std::is_nothrow_constructible;

using std::is_default_constructible;
using std::is_trivially_default_constructible;
using std::is_nothrow_default_constructible;

using std::is_copy_constructible;
using std::is_trivially_copy_constructible;
using std::is_nothrow_copy_constructible;

using std::is_move_constructible;
using std::is_trivially_move_constructible;
using std::is_nothrow_move_constructible;

using std::is_assignable;
using std::is_trivially_assignable;
using std::is_nothrow_assignable;

using std::is_copy_assignable;
using std::is_trivially_copy_assignable;
using std::is_nothrow_copy_assignable;

using std::is_move_assignable;
using std::is_trivially_move_assignable;
using std::is_nothrow_move_assignable;

using std::is_destructible;
using std::is_trivially_destructible;
using std::is_nothrow_destructible;

using std::has_virtual_destructor;

// Property Queries
using std::alignment_of;
using std::rank;
using std::extent;

// Type Relationships
using std::is_same;
using std::is_base_of;
using std::is_convertible;

// Const-volatility specifiers
using std::remove_cv;
using std::remove_cv_t;
using std::remove_const;
using std::remove_const_t;
using std::remove_volatile;
using std::remove_volatile_t;
using std::add_cv;
using std::add_cv_t;
using std::add_const;
using std::add_const_t;
using std::add_volatile;
using std::add_volatile_t;

// References
using std::remove_reference;
using std::remove_reference_t;
using std::add_lvalue_reference;
using std::add_lvalue_reference_t;
using std::add_rvalue_reference;
using std::add_rvalue_reference_t;

// Pointers
using std::remove_pointer;
using std::remove_pointer_t;
using std::add_pointer;
using std::add_pointer_t;

// Sign Modifiers
using std::make_signed;
using std::make_signed_t;
using std::make_unsigned;
using std::make_unsigned_t;

// Arrays
using std::remove_extent;
using std::remove_extent_t;
using std::remove_all_extents;
using std::remove_all_extents_t;

// Misc transformations
using std::decay;
using std::decay_t;
using std::enable_if;
using std::enable_if_t;
using std::conditional;
using std::conditional_t;
using std::common_type;
using std::common_type_t;
using std::underlying_type;
using std::underlying_type_t;

#endif 

template <bool B>
using bool_constant = boost::math::integral_constant<bool, B>;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_void_v = boost::math::is_void<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_null_pointer_v = boost::math::is_null_pointer<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_integral_v = boost::math::is_integral<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_floating_point_v = boost::math::is_floating_point<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_array_v = boost::math::is_array<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_enum_v = boost::math::is_enum<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_union_v = boost::math::is_union<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_class_v = boost::math::is_class<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_function_v = boost::math::is_function<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_pointer_v = boost::math::is_pointer<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_lvalue_reference_v = boost::math::is_lvalue_reference<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_rvalue_reference_v = boost::math::is_rvalue_reference<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_member_object_pointer_v = boost::math::is_member_object_pointer<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_member_function_pointer_v = boost::math::is_member_function_pointer<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_fundamental_v = boost::math::is_fundamental<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_arithmetic_v = boost::math::is_arithmetic<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_scalar_v = boost::math::is_scalar<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_object_v = boost::math::is_object<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_compound_v = boost::math::is_compound<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_reference_v = boost::math::is_reference<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_member_pointer_v = boost::math::is_member_pointer<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_const_v = boost::math::is_const<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_volatile_v = boost::math::is_volatile<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivial_v = boost::math::is_trivial<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_copyable_v = boost::math::is_trivially_copyable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_standard_layout_v = boost::math::is_standard_layout<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_empty_v = boost::math::is_empty<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_polymorphic_v = boost::math::is_polymorphic<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_abstract_v = boost::math::is_abstract<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_final_v = boost::math::is_final<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_signed_v = boost::math::is_signed<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_unsigned_v = boost::math::is_unsigned<T>::value;

template <typename T, typename... Args>
BOOST_MATH_INLINE_CONSTEXPR bool is_constructible_v = boost::math::is_constructible<T, Args...>::value;

template <typename T, typename... Args>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_constructible_v = boost::math::is_trivially_constructible<T, Args...>::value;

template <typename T, typename... Args>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_constructible_v = boost::math::is_nothrow_constructible<T, Args...>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_default_constructible_v = boost::math::is_default_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_default_constructible_v = boost::math::is_trivially_default_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_default_constructible_v = boost::math::is_nothrow_default_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_copy_constructible_v = boost::math::is_copy_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_copy_constructible_v = boost::math::is_trivially_copy_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_copy_constructible_v = boost::math::is_nothrow_copy_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_move_constructible_v = boost::math::is_move_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_move_constructible_v = boost::math::is_trivially_move_constructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_move_constructible_v = boost::math::is_nothrow_move_constructible<T>::value;

template <typename T, typename U>
BOOST_MATH_INLINE_CONSTEXPR bool is_assignable_v = boost::math::is_assignable<T, U>::value;

template <typename T, typename U>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_assignable_v = boost::math::is_trivially_assignable<T, U>::value;

template <typename T, typename U>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_assignable_v = boost::math::is_nothrow_assignable<T, U>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_copy_assignable_v = boost::math::is_copy_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_copy_assignable_v = boost::math::is_trivially_copy_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_copy_assignable_v = boost::math::is_nothrow_copy_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_move_assignable_v = boost::math::is_move_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_move_assignable_v = boost::math::is_trivially_move_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_move_assignable_v = boost::math::is_nothrow_move_assignable<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_destructible_v = boost::math::is_destructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_trivially_destructible_v = boost::math::is_trivially_destructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool is_nothrow_destructible_v = boost::math::is_nothrow_destructible<T>::value;

template <typename T>
BOOST_MATH_INLINE_CONSTEXPR bool has_virtual_destructor_v = boost::math::has_virtual_destructor<T>::value;

template <typename T, typename U>
BOOST_MATH_INLINE_CONSTEXPR bool is_same_v = boost::math::is_same<T, U>::value;

template <typename T, typename U>
BOOST_MATH_INLINE_CONSTEXPR bool is_base_of_v = boost::math::is_base_of<T, U>::value;

} // namespace math
} // namespace boost

#endif // BOOST_MATH_TOOLS_TYPE_TRAITS
