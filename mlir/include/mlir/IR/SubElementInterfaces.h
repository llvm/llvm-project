//===- SubElementInterfaces.h - Attr and Type SubElements -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and utilities for querying the sub elements of
// an attribute or type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SUBELEMENTINTERFACES_H
#define MLIR_IR_SUBELEMENTINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
template <typename T>
using SubElementReplFn = function_ref<T(T)>;
template <typename T>
using SubElementResultReplFn = function_ref<std::pair<T, WalkResult>(T)>;

//===----------------------------------------------------------------------===//
/// AttrTypeSubElementHandler
//===----------------------------------------------------------------------===//

/// This class is used by AttrTypeSubElementHandler instances to walking sub
/// attributes and types.
class AttrTypeSubElementWalker {
public:
  AttrTypeSubElementWalker(function_ref<void(Attribute)> walkAttrsFn,
                           function_ref<void(Type)> walkTypesFn)
      : walkAttrsFn(walkAttrsFn), walkTypesFn(walkTypesFn) {}

  /// Walk an attribute.
  void walk(Attribute element) {
    if (element)
      walkAttrsFn(element);
  }
  /// Walk a type.
  void walk(Type element) {
    if (element)
      walkTypesFn(element);
  }
  /// Walk a range of attributes or types.
  template <typename RangeT>
  void walkRange(RangeT &&elements) {
    for (auto element : elements)
      walk(element);
  }

private:
  function_ref<void(Attribute)> walkAttrsFn;
  function_ref<void(Type)> walkTypesFn;
};

/// This class is used by AttrTypeSubElementHandler instances to process sub
/// element replacements.
template <typename T>
class AttrTypeSubElementReplacements {
public:
  AttrTypeSubElementReplacements(ArrayRef<T> repls) : repls(repls) {}

  /// Take the first N replacements as an ArrayRef, dropping them from
  /// this replacement list.
  ArrayRef<T> take_front(unsigned n) {
    ArrayRef<T> elements = repls.take_front(n);
    repls = repls.drop_front(n);
    return elements;
  }

private:
  /// The current set of replacements.
  ArrayRef<T> repls;
};
using AttrSubElementReplacements = AttrTypeSubElementReplacements<Attribute>;
using TypeSubElementReplacements = AttrTypeSubElementReplacements<Type>;

/// This class provides support for interacting with the
/// SubElementInterfaces for different types of parameters. An
/// implementation of this class should be provided for any parameter class
/// that may contain an attribute or type. There are two main methods of
/// this class that need to be implemented:
///
///  - walk
///
///   This method should traverse into any sub elements of the parameter
///   using the provided walker, or by invoking handlers for sub-types.
///
///  - replace
///
///   This method should extract any necessary sub elements using the
///   provided replacer, or by invoking handlers for sub-types. The new
///   post-replacement parameter value should be returned.
///
template <typename T, typename Enable = void>
struct AttrTypeSubElementHandler {
  /// Default walk implementation that does nothing.
  static inline void walk(const T &param, AttrTypeSubElementWalker &walker) {}

  /// Default replace implementation just forwards the parameter.
  template <typename ParamT>
  static inline decltype(auto) replace(ParamT &&param,
                                       AttrSubElementReplacements &attrRepls,
                                       TypeSubElementReplacements &typeRepls) {
    return std::forward<ParamT>(param);
  }

  /// Tag indicating that this handler does not support sub-elements.
  using DefaultHandlerTag = void;
};

/// Detect if any of the given parameter types has a sub-element handler.
namespace detail {
template <typename T>
using has_default_sub_element_handler_t = decltype(T::DefaultHandlerTag);
} // namespace detail
template <typename... Ts>
inline constexpr bool has_sub_attr_or_type_v =
    (!llvm::is_detected<detail::has_default_sub_element_handler_t, Ts>::value ||
     ...);

/// Implementation for derived Attributes and Types.
template <typename T>
struct AttrTypeSubElementHandler<
    T, std::enable_if_t<std::is_base_of_v<Attribute, T> ||
                        std::is_base_of_v<Type, T>>> {
  static void walk(T param, AttrTypeSubElementWalker &walker) {
    walker.walk(param);
  }
  static T replace(T param, AttrSubElementReplacements &attrRepls,
                   TypeSubElementReplacements &typeRepls) {
    if (!param)
      return T();
    if constexpr (std::is_base_of_v<Attribute, T>) {
      return cast<T>(attrRepls.take_front(1)[0]);
    } else {
      return cast<T>(typeRepls.take_front(1)[0]);
    }
  }
};
template <>
struct AttrTypeSubElementHandler<NamedAttribute> {
  template <typename T>
  static void walk(T param, AttrTypeSubElementWalker &walker) {
    walker.walk(param.getName());
    walker.walk(param.getValue());
  }
  template <typename T>
  static T replace(T param, AttrSubElementReplacements &attrRepls,
                   TypeSubElementReplacements &typeRepls) {
    ArrayRef<Attribute> paramRepls = attrRepls.take_front(2);
    return T(cast<decltype(param.getName())>(paramRepls[0]), paramRepls[1]);
  }
};
/// Implementation for derived ArrayRef.
template <typename T>
struct AttrTypeSubElementHandler<ArrayRef<T>,
                                 std::enable_if_t<has_sub_attr_or_type_v<T>>> {
  using EltHandler = AttrTypeSubElementHandler<T>;

  static void walk(ArrayRef<T> param, AttrTypeSubElementWalker &walker) {
    for (const T &subElement : param)
      EltHandler::walk(subElement, walker);
  }
  static auto replace(ArrayRef<T> param, AttrSubElementReplacements &attrRepls,
                      TypeSubElementReplacements &typeRepls) {
    // Normal attributes/types can extract using the replacer directly.
    if constexpr (std::is_base_of_v<Attribute, T> &&
                  sizeof(T) == sizeof(Attribute)) {
      ArrayRef<Attribute> attrs = attrRepls.take_front(param.size());
      return ArrayRef<T>((const T *)attrs.data(), attrs.size());
    } else if constexpr (std::is_base_of_v<Type, T> &&
                         sizeof(T) == sizeof(Type)) {
      ArrayRef<Type> types = typeRepls.take_front(param.size());
      return ArrayRef<T>((const T *)types.data(), types.size());
    } else {
      // Otherwise, we need to allocate storage for the new elements.
      SmallVector<T> newElements;
      for (const T &element : param)
        newElements.emplace_back(
            EltHandler::replace(element, attrRepls, typeRepls));
      return newElements;
    }
  }
};
/// Implementation for Tuple.
template <typename... Ts>
struct AttrTypeSubElementHandler<
    std::tuple<Ts...>, std::enable_if_t<has_sub_attr_or_type_v<Ts...>>> {
  static void walk(const std::tuple<Ts...> &param,
                   AttrTypeSubElementWalker &walker) {
    std::apply(
        [&](const Ts &...params) {
          (AttrTypeSubElementHandler<Ts>::walk(params, walker), ...);
        },
        param);
  }
  static auto replace(const std::tuple<Ts...> &param,
                      AttrSubElementReplacements &attrRepls,
                      TypeSubElementReplacements &typeRepls) {
    return std::apply(
        [&](const Ts &...params)
            -> std::tuple<decltype(AttrTypeSubElementHandler<Ts>::replace(
                params, attrRepls, typeRepls))...> {
          return {AttrTypeSubElementHandler<Ts>::replace(params, attrRepls,
                                                         typeRepls)...};
        },
        param);
  }
};

namespace detail {
template <typename T>
struct is_tuple : public std::false_type {};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : public std::true_type {};
template <typename T, typename... Ts>
using has_get_method = decltype(T::get(std::declval<Ts>()...));

/// This function provides the underlying implementation for the
/// SubElementInterface walk method, using the key type of the derived
/// attribute/type to interact with the individual parameters.
template <typename T>
void walkImmediateSubElementsImpl(T derived,
                                  function_ref<void(Attribute)> walkAttrsFn,
                                  function_ref<void(Type)> walkTypesFn) {
  auto key = static_cast<typename T::ImplType *>(derived.getImpl())->getAsKey();

  // If we don't have any sub-elements, there is nothing to do.
  if constexpr (!has_sub_attr_or_type_v<decltype(key)>) {
    return;
  } else {
    AttrTypeSubElementWalker walker(walkAttrsFn, walkTypesFn);
    AttrTypeSubElementHandler<decltype(key)>::walk(key, walker);
  }
}

/// This function invokes the proper `get` method for  a type `T` with the given
/// values.
template <typename T, typename... Ts>
T constructSubElementReplacement(MLIRContext *ctx, Ts &&...params) {
  // Prefer a direct `get` method if one exists.
  if constexpr (llvm::is_detected<has_get_method, T, Ts...>::value) {
    (void)ctx;
    return T::get(std::forward<Ts>(params)...);
  } else if constexpr (llvm::is_detected<has_get_method, T, MLIRContext *,
                                         Ts...>::value) {
    return T::get(ctx, std::forward<Ts>(params)...);
  } else {
    // Otherwise, pass to the base get.
    return T::Base::get(ctx, std::forward<Ts>(params)...);
  }
}

/// This function provides the underlying implementation for the
/// SubElementInterface replace method, using the key type of the derived
/// attribute/type to interact with the individual parameters.
template <typename T>
T replaceImmediateSubElementsImpl(T derived, ArrayRef<Attribute> &replAttrs,
                                  ArrayRef<Type> &replTypes) {
  auto key = static_cast<typename T::ImplType *>(derived.getImpl())->getAsKey();

  // If we don't have any sub-elements, we can just return the original.
  if constexpr (!has_sub_attr_or_type_v<decltype(key)>) {
    return derived;

    // Otherwise, we need to replace any necessary sub-elements.
  } else {
    AttrSubElementReplacements attrRepls(replAttrs);
    TypeSubElementReplacements typeRepls(replTypes);
    auto newKey = AttrTypeSubElementHandler<decltype(key)>::replace(
        key, attrRepls, typeRepls);
    if constexpr (is_tuple<decltype(key)>::value) {
      return std::apply(
          [&](auto &&...params) {
            return constructSubElementReplacement<T>(
                derived.getContext(),
                std::forward<decltype(params)>(params)...);
          },
          newKey);
    } else {
      return constructSubElementReplacement<T>(derived.getContext(), newKey);
    }
  }
}
} // namespace detail
} // namespace mlir

/// Include the definitions of the sub elemnt interfaces.
#include "mlir/IR/SubElementAttrInterfaces.h.inc"
#include "mlir/IR/SubElementTypeInterfaces.h.inc"

#endif // MLIR_IR_SUBELEMENTINTERFACES_H
