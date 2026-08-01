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
#include "llvm/ADT/FoldingSet.h"
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
class FortranType : public llvm::FoldingSetNode {
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
  FortranType(int32_t kind, uint64_t bitsize, const ConstString &name)
      : m_kind(kind), m_bitsize(bitsize), m_type_name(name) {}
  virtual ~FortranType() = default;
  int GetKind() const { return m_kind; }
  uint64_t GetBitSize() const { return m_bitsize; }
  ConstString GetName() const { return m_type_name; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, m_kind, m_bitsize);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, int32_t kind,
                      uint64_t bitsize) {
    ID.AddInteger(kind);
    ID.AddInteger(bitsize);
  }

private:
  int32_t m_kind;
  uint64_t m_bitsize;
  ConstString m_type_name;
};

class FortranFunction : public FortranType {
public:
  FortranFunction(ConstString func_name,
                  const llvm::SmallVectorImpl<CompilerType> &parameters)
      : FortranType(FortranType::KIND_FUNCTION, 0, func_name) {
    m_parameters.assign(parameters.begin(), parameters.end());
  }
  llvm::ArrayRef<CompilerType> GetParameters() const { return m_parameters; }
  size_t GetNumberOfParameters() const { return m_parameters.size(); }

  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, GetName(), m_parameters);
  }

  static void Profile(llvm::FoldingSetNodeID &id, ConstString func_name,
                      llvm::ArrayRef<CompilerType> parameters) {
    id.AddString(func_name.GetStringRef());
    id.AddInteger(parameters.size());

    for (const auto &param : parameters)
      id.AddPointer(param.GetOpaqueQualType());
  }

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

  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, m_category, m_is_bound_known, m_bound);
  }

  static void Profile(llvm::FoldingSetNodeID &id, Category category,
                      bool is_bound_known, int64_t bound) {
    id.AddInteger(static_cast<uint32_t>(category));

    id.AddBoolean(is_bound_known);

    if (is_bound_known)
      id.AddInteger(bound);
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
  uint64_t GetByteStride() const { return m_byte_stride; }
  uint64_t GetElementCount() const { return m_element_count; }

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

  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, m_lb, m_ub, m_byte_stride, m_element_count, m_element_count_exp,
            m_upper_bound_exp, m_lower_bound_exp, m_byte_stride_exp);
  }

  static void Profile(llvm::FoldingSetNodeID &id, const ArrayBound &lb,
                      const ArrayBound &ub, const uint64_t byte_stride,
                      const uint64_t element_count,

                      const DWARFExpressionList &element_count_exp,
                      const DWARFExpressionList &upper_bound_exp,
                      const DWARFExpressionList &lower_bound_exp,
                      const DWARFExpressionList &byte_stride_exp) {
    lb.Profile(id);
    ub.Profile(id);
    id.AddInteger(byte_stride);
    id.AddInteger(element_count);
    id.AddBoolean(element_count_exp.IsValid());
    id.AddBoolean(upper_bound_exp.IsValid());
    id.AddBoolean(lower_bound_exp.IsValid());
    id.AddBoolean(byte_stride_exp.IsValid());
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
      : FortranType(TypeKind::KIND_ARRAY, total_array_size, array_type_name),
        m_element_type(element_type),
        m_dimensions(dimensions.begin(), dimensions.end()),
        m_is_allocatable(is_allocatable), m_is_dynamic(is_dynamic),
        m_total_elements(total_elements), m_allocated_exp(allocated_exp),
        m_data_location_exp(data_location_exp) {}
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
  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, m_element_type, m_dimensions, m_is_allocatable, m_is_dynamic,
            m_allocated_exp, m_data_location_exp);
  }

  static void Profile(llvm::FoldingSetNodeID &id, CompilerType element_type,
                      llvm::ArrayRef<ArrayShape> dimensions,
                      bool is_allocatable, bool is_dynamic,
                      const DWARFExpressionList &allocated_exp,
                      const DWARFExpressionList &data_location_exp) {
    id.AddPointer(element_type.GetOpaqueQualType());
    id.AddBoolean(is_allocatable);
    id.AddBoolean(is_dynamic);
    id.AddBoolean(allocated_exp.IsValid());
    id.AddBoolean(data_location_exp.IsValid());
    id.AddInteger(dimensions.size());

    for (const auto &shape : dimensions)
      shape.Profile(id);
  }

private:
  CompilerType m_element_type;
  llvm::SmallVector<ArrayShape, 2> m_dimensions;
  bool m_is_allocatable;
  // To know if the array is fully explicit without looping through the shapes
  // every time
  bool m_is_dynamic;
  uint64_t m_total_elements;
  DWARFExpressionList m_allocated_exp;
  DWARFExpressionList m_data_location_exp;
};

// TODO: Calculate correct pointer size
class FortranPointer : public FortranType {
public:
  FortranPointer(FortranType *pointee, ConstString type_name)
      : FortranType(KIND_POINTER, sizeof(void *), type_name),
        m_pointee(pointee) {}

  void SetPointee(FortranType *pointee) { m_pointee = pointee; }
  FortranType *GetPointee() const { return m_pointee; }

  void Profile(llvm::FoldingSetNodeID &id) const { Profile(id, m_pointee); }
  static void Profile(llvm::FoldingSetNodeID &id,
                      const FortranType *m_pointee) {
    id.AddPointer(m_pointee);
  }

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