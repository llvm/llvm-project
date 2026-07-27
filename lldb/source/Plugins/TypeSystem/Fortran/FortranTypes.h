//===-- FortranTypes.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H

#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/ConstString.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private {
namespace plugin {
namespace fortran {
using DWARFValue =
    std::variant<std::monostate, // Represents "Default" or "Not Present"
                 uint64_t,       // Constant value (e.g., constant byte stride)
                 int64_t,
                 DWARFExpressionList // Dynamic expression (e.g.,
                                     // variable/runtime stride)
                 >;

struct FortranDimension {
  DWARFValue lower_bound;
  DWARFValue upper_bound;
  DWARFValue element_count;
  DWARFValue byte_stride;
};

struct FortranArrayMetadata {
  CompilerType element_type;
  llvm::SmallVector<FortranDimension, 4> dimensions;
  bool is_allocatable = false;
  bool is_dynamic = false;
  bool is_star = false;
  DWARFExpressionList allocated_exp;
  DWARFExpressionList data_location_exp;
};

/// A simplified internal representation of a Fortran type.
/// In the future, this will likely be replaced by a Flang-backed AST.
class FortranType {
public:
  enum TypeKind {
    KIND_INTEGER,
    KIND_LOGICAL,
    KIND_REAL,
    KIND_COMPLEX,
    KIND_FUNCTION,
    KIND_ARRAY,
    KIND_POINTER,
    KIND_UNKNOWN
  };
  FortranType(int kind, const ConstString &name, uint64_t bitsize)
      : m_kind(kind), m_bitsize(bitsize), m_type_name(name) {}
  int GetKind() const { return m_kind; }
  uint64_t GetBitSize() const { return m_bitsize; }
  ConstString GetName() const { return m_type_name; }

private:
  int m_kind;
  uint64_t m_bitsize;
  ConstString m_type_name;
};

class FortranFunction : public FortranType {
public:
  FortranFunction(ConstString func_name,
                  const llvm::SmallVectorImpl<CompilerType> &parameters)
      : FortranType(FortranType::KIND_FUNCTION, func_name, 0) {
    m_parameters.assign(parameters.begin(), parameters.end());
  }
  llvm::ArrayRef<CompilerType> GetParameters() const { return m_parameters; }
  size_t GetNumberOfParameters() const { return m_parameters.size(); }

private:
  llvm::SmallVector<CompilerType, 4> m_parameters;
};

class ArrayBound {
public:
  enum class Category { Explicit, Star, Colon };
  ArrayBound() = default;
  ArrayBound(Category category) : m_category{category} {}
  ArrayBound(Category category, int64_t bound)
      : m_category{category}, m_bound(bound), m_is_bound_known(true) {}

  bool IsExplicit() const { return m_category == Category::Explicit; }

  bool IsStar() const { return m_category == Category::Star; }

  bool IsColon() const { return m_category == Category::Colon; }

  void SetCategory(Category c) { m_category = c; }

  bool IsBoundKnown() const { return m_is_bound_known; }

  int64_t GetBound() const {
    assert(m_is_bound_known && "Can't get the bound if it is not explicit");
    return m_bound;
  }

  void SetBound(int64_t bound) {
    m_is_bound_known = true;
    m_bound = bound;
  }

private:
  Category m_category{Category::Explicit};
  int64_t m_bound;
  bool m_is_bound_known = false;
};

class ArrayShape {
public:
  ArrayShape() = default;
  ArrayShape(ArrayBound lb, ArrayBound ub, uint64_t byte_stride)
      : m_lb(lb), m_ub(ub), m_byte_stride(byte_stride) {}

  const ArrayBound &GetLowerBound() const { return m_lb; }
  const ArrayBound &GetUpperBound() const { return m_ub; }
  const uint64_t GetByteStride() const { return m_byte_stride; }
  const uint64_t GetElementCount() const { return m_element_count; }

  int64_t GetNumberOfElements() const {
    return m_ub.GetBound() - m_lb.GetBound() + 1;
  }

  const DWARFExpressionList &GetElementCountExpression() const {
    return m_element_count_exp;
  }
  const DWARFExpressionList &GetUpperBoundExpression() const {
    return m_upper_bound_exp;
  }
  const DWARFExpressionList &GetLowerBoundExpression() const {
    return m_lower_bound_exp;
  }
  const DWARFExpressionList &GetByteStrideExpression() const {
    return m_byte_stride_exp;
  }

  void SetLowerBound(const ArrayBound &lb) { m_lb = lb; }
  void SetUpperBound(const ArrayBound &ub) { m_ub = ub; }
  void SetByteStride(uint64_t byte_stride) { m_byte_stride = byte_stride; }
  void SetElementCount(uint64_t element_count) {
    m_element_count = element_count;
  }

  void SetElementCountExpression(DWARFExpressionList expr) {
    m_element_count_exp = std::move(expr);
  }
  void SetUpperBoundExpression(DWARFExpressionList expr) {
    m_upper_bound_exp = std::move(expr);
  }
  void SetLowerBoundExpression(DWARFExpressionList expr) {
    m_lower_bound_exp = std::move(expr);
  }
  void SetByteStrideExpression(DWARFExpressionList expr) {
    m_byte_stride_exp = std::move(expr);
  }

private:
  ArrayBound m_lb;
  ArrayBound m_ub;
  uint64_t m_byte_stride;
  uint64_t m_element_count;

  DWARFExpressionList m_element_count_exp;
  DWARFExpressionList m_upper_bound_exp;
  DWARFExpressionList m_lower_bound_exp;
  DWARFExpressionList m_byte_stride_exp;
};

class FortranArray : public FortranType {
public:
  FortranArray(CompilerType element_type,
               const llvm::SmallVectorImpl<ArrayShape> &dimensions,
               ConstString array_type_name, uint64_t total_array_size,
               bool is_allocatable, bool is_dynamic, uint64_t total_elements,
               DWARFExpressionList allocated_exp,
               DWARFExpressionList data_location_exp)
      : m_element_type(element_type),
        m_dimensions(dimensions.begin(), dimensions.end()),
        m_is_allocatable(is_allocatable), m_is_dynamic(is_dynamic),
        m_total_elements(total_elements), m_allocated_exp(allocated_exp),
        m_data_location_exp(data_location_exp),
        FortranType(TypeKind::KIND_ARRAY, array_type_name, total_array_size) {}
  // TODO: Add necessary methods here
  CompilerType GetElementType() const { return m_element_type; }
  uint64_t GetTotalElements() const { return m_total_elements; }
  bool IsAllocatable() const { return m_is_allocatable; }
  bool IsDynamic() const { return m_is_dynamic; }
  uint64_t GetElementByteSize() const {
    auto byte_size_or_err = m_element_type.GetByteSize(nullptr);
    // TODO: Change this to returning an error, and change return type to
    // expected<uint64_t>
    if (!byte_size_or_err)
      return 0;
    return *byte_size_or_err;
  }

  size_t GetRank() const { return m_dimensions.size(); }
  llvm::ArrayRef<ArrayShape> GetDimensions() const { return m_dimensions; }

  DWARFExpressionList GetAllocatedExpression() const { return m_allocated_exp; }
  DWARFExpressionList GetDataLocationExpression() const {
    return m_data_location_exp;
  }

private:
  CompilerType m_element_type;
  llvm::SmallVector<ArrayShape, 2> m_dimensions;
  uint64_t m_total_elements;
  bool m_is_allocatable;
  // To know if the array is fully explicit without looping through the shapes
  // every time
  bool m_is_dynamic;
  DWARFExpressionList m_allocated_exp;
  DWARFExpressionList m_data_location_exp;
};

class FortranPointer : public FortranType {
public:
  FortranPointer(FortranType *pointee, ConstString type_name)
      : m_pointee(pointee), FortranType(KIND_POINTER, type_name, 64) {}

  void SetPointee(FortranType *pointee) { m_pointee = pointee; }
  FortranType *GetPointee() const { return m_pointee; }

private:
  FortranType *m_pointee;
};

inline ConstString CreateArrayTypeName(const CompilerType &element_type,
                                       const llvm::ArrayRef<ArrayShape> shapes,
                                       bool is_allocatable, bool is_star) {

  std::string name_buffer;
  llvm::raw_string_ostream name_stream(name_buffer);

  name_stream << element_type.GetTypeName().AsCString(nullptr) << "(";
  size_t rank = shapes.size();

  for (size_t idx = 0; idx < rank; ++idx) {
    if (idx > 0)
      name_stream << ", ";

    const ArrayBound &lb = shapes[idx].GetLowerBound();
    const ArrayBound &ub = shapes[idx].GetUpperBound();

    if (ub.IsStar()) {
      // Assuming you want standard Fortran syntax (e.g., "1:*")
      name_stream << lb.GetBound() << ":*";
    } else if (ub.IsColon()) {
      // Unknown bound elements
      name_stream << ":";
    } else if (ub.IsExplicit()) {
      // Explicit bounds
      if (lb.GetBound() != 1) {
        name_stream << lb.GetBound() << ":";
      }
      name_stream << ub.GetBound();
    }
  }

  name_stream << ")";

  if (is_allocatable) {
    name_stream << ", allocatable";
  }

  name_stream.flush();
  return ConstString(name_buffer.c_str());
}
} // namespace fortran
} // namespace plugin
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H