//===------------------------- ParameterType.h ---------------------------===//
//
//                              SPIR Tools
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
/*
 * Contributed by: Intel Corporation.
 */

#ifndef SPIRV_MANGLER_PARAMETERTYPE_H
#define SPIRV_MANGLER_PARAMETERTYPE_H

#include "Refcount.h"
#include <string>
#include <vector>

// The Type class hierarchy models the different types in OCL.

namespace SPIR {

// Supported SPIR versions
enum SPIRversion { SPIR12 = 1, SPIR20 = 2 };

// Error Status values
enum MangleError {
  MANGLE_SUCCESS,
  MANGLE_TYPE_NOT_SUPPORTED,
  MANGLE_NULL_FUNC_DESCRIPTOR
};

enum TypePrimitiveEnum {
  PRIMITIVE_FIRST,
  PRIMITIVE_BOOL = PRIMITIVE_FIRST,
  PRIMITIVE_UCHAR,
  PRIMITIVE_CHAR,
  PRIMITIVE_USHORT,
  PRIMITIVE_SHORT,
  PRIMITIVE_UINT,
  PRIMITIVE_INT,
  PRIMITIVE_ULONG,
  PRIMITIVE_LONG,
  PRIMITIVE_HALF,
  PRIMITIVE_FLOAT,
  PRIMITIVE_DOUBLE,
  PRIMITIVE_VOID,
  PRIMITIVE_VAR_ARG,
  PRIMITIVE_STRUCT_FIRST,
  PRIMITIVE_IMAGE1D_RO_T = PRIMITIVE_STRUCT_FIRST,
  PRIMITIVE_IMAGE1D_ARRAY_RO_T,
  PRIMITIVE_IMAGE1D_BUFFER_RO_T,
  PRIMITIVE_IMAGE2D_RO_T,
  PRIMITIVE_IMAGE2D_ARRAY_RO_T,
  PRIMITIVE_IMAGE2D_DEPTH_RO_T,
  PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T,
  PRIMITIVE_IMAGE2D_MSAA_RO_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T,
  PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T,
  PRIMITIVE_IMAGE3D_RO_T,
  PRIMITIVE_IMAGE1D_WO_T,
  PRIMITIVE_IMAGE1D_ARRAY_WO_T,
  PRIMITIVE_IMAGE1D_BUFFER_WO_T,
  PRIMITIVE_IMAGE2D_WO_T,
  PRIMITIVE_IMAGE2D_ARRAY_WO_T,
  PRIMITIVE_IMAGE2D_DEPTH_WO_T,
  PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T,
  PRIMITIVE_IMAGE2D_MSAA_WO_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T,
  PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T,
  PRIMITIVE_IMAGE3D_WO_T,
  PRIMITIVE_IMAGE1D_RW_T,
  PRIMITIVE_IMAGE1D_ARRAY_RW_T,
  PRIMITIVE_IMAGE1D_BUFFER_RW_T,
  PRIMITIVE_IMAGE2D_RW_T,
  PRIMITIVE_IMAGE2D_ARRAY_RW_T,
  PRIMITIVE_IMAGE2D_DEPTH_RW_T,
  PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T,
  PRIMITIVE_IMAGE2D_MSAA_RW_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T,
  PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T,
  PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T,
  PRIMITIVE_IMAGE3D_RW_T,
  PRIMITIVE_EVENT_T,
  PRIMITIVE_PIPE_RO_T,
  PRIMITIVE_PIPE_WO_T,
  PRIMITIVE_RESERVE_ID_T,
  PRIMITIVE_QUEUE_T,
  PRIMITIVE_NDRANGE_T,
  PRIMITIVE_CLK_EVENT_T,
  PRIMITIVE_STRUCT_LAST = PRIMITIVE_CLK_EVENT_T,
  PRIMITIVE_SAMPLER_T,
  PRIMITIVE_KERNEL_ENQUEUE_FLAGS_T,
  PRIMITIVE_CLK_PROFILING_INFO,
  PRIMITIVE_MEMORY_ORDER,
  PRIMITIVE_MEMORY_SCOPE,
  PRIMITIVE_LAST = PRIMITIVE_MEMORY_SCOPE,
  PRIMITIVE_NONE,
  // Keep this at the end.
  PRIMITIVE_NUM = PRIMITIVE_NONE
};

enum TypeEnum {
  TYPE_ID_PRIMITIVE,
  TYPE_ID_POINTER,
  TYPE_ID_VECTOR,
  TYPE_ID_ATOMIC,
  TYPE_ID_BLOCK,
  TYPE_ID_STRUCTURE
};

enum TypeAttributeEnum {
  ATTR_QUALIFIER_FIRST = 0,
  ATTR_RESTRICT = ATTR_QUALIFIER_FIRST,
  ATTR_VOLATILE,
  ATTR_CONST,
  ATTR_QUALIFIER_LAST = ATTR_CONST,
  ATTR_ADDR_SPACE_FIRST,
  ATTR_PRIVATE = ATTR_ADDR_SPACE_FIRST,
  ATTR_GLOBAL,
  ATTR_CONSTANT,
  ATTR_LOCAL,
  ATTR_GENERIC,
  ATTR_ADDR_SPACE_LAST = ATTR_GENERIC,
  ATTR_NONE,
  ATTR_NUM = ATTR_NONE
};

// Forward declaration for abstract structure.
struct ParamType;
typedef RefCount<ParamType> RefParamType;

// Forward declaration for abstract structure.
struct TypeVisitor;

struct ParamType {
  /// @brief Constructor.
  /// @param TypeEnum type id.
  ParamType(TypeEnum TypeId) : TypeId(TypeId){};

  /// @brief Destructor.
  virtual ~ParamType(){};

  /// Abstract Methods ///

  /// @brief Visitor service method. (see TypeVisitor for more details).
  ///        When overridden in subclasses, preform a 'double dispatch' to the
  ///        appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor.
  virtual MangleError accept(TypeVisitor *) const = 0;

  /// @brief Returns a string representation of the underlying type.
  /// @return type as string.
  virtual std::string toString() const = 0;

  /// @brief Returns true if given param type is equal to this type.
  /// @param ParamType given param type.
  /// @return true if given param type is equal to this type and false
  /// otherwise.
  virtual bool equals(const ParamType *) const = 0;

  /// Common Base-Class Methods ///

  /// @brief Returns type id of underlying type.
  /// @return type id.
  TypeEnum getTypeId() const { return TypeId; }

private:
  // @brief Default Constructor.
  ParamType();

protected:
  /// An enumeration to identify the type id of this instance.
  TypeEnum TypeId;
};

struct PrimitiveType : public ParamType {
  /// An enumeration to identify the type id of this class.
  const static TypeEnum EnumTy;

  /// @brief Constructor.
  /// @param TypePrimitiveEnum primitive id.
  PrimitiveType(TypePrimitiveEnum);

  /// Implementation of Abstract Methods ///

  /// @brief Visitor service method. (see TypeVisitor for more details).
  ///        When overridden in subclasses, preform a 'double dispatch' to the
  ///        appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor.
  MangleError accept(TypeVisitor *) const override;

  /// @brief Returns a string representation of the underlying type.
  /// @return type as string.
  std::string toString() const override;

  /// @brief Returns true if given param type is equal to this type.
  /// @param ParamType given param type.
  /// @return true if given param type is equal to this type and false
  /// otherwise.
  bool equals(const ParamType *) const override;

  /// Non-Common Methods ///

  /// @brief Returns the primitive enumeration of the type.
  /// @return primitive type.
  TypePrimitiveEnum getPrimitive() const { return Primitive; }

protected:
  /// An enumeration to identify the primitive type.
  TypePrimitiveEnum Primitive;
};

struct PointerType : public ParamType {
  /// An enumeration to identify the type id of this class.
  const static TypeEnum EnumTy;

  /// @brief Constructor.
  /// @param RefParamType the type of pointee (that the pointer points at).
  PointerType(const RefParamType Type);

  /// Implementation of Abstract Methods ///

  /// @brief Visitor service method. (see TypeVisitor for more details).
  ///        When overridden in subclasses, preform a 'double dispatch' to the
  ///        appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor
  MangleError accept(TypeVisitor *) const override;

  /// @brief Returns a string representation of the underlying type.
  /// @return type as string.
  std::string toString() const override;

  /// @brief Returns true if given param type is equal to this type.
  /// @param ParamType given param type.
  /// @return true if given param type is equal to this type and false
  /// otherwise.
  bool equals(const ParamType *) const override;

  /// Non-Common Methods ///

  /// @brief Returns the type the pointer is pointing at.
  /// @return pointee type.
  const RefParamType &getPointee() const { return PType; }

  /// @brief Sets the address space attribute - default is __private
  /// @param TypeAttributeEnum address space attribute id.
  void setAddressSpace(TypeAttributeEnum Attr);

  /// @brief Returns the pointer's address space.
  /// @return pointer's address space.
  TypeAttributeEnum getAddressSpace() const;

  /// @brief Adds or removes a pointer's qualifier.
  /// @param TypeAttributeEnum qual - qualifier to add/remove.
  /// @param bool enabled - true if qualifier should exist false otherwise.
  ///        default is set to false.
  void setQualifier(TypeAttributeEnum Qual, bool Enabled);

  /// @brief Checks if the pointer has a certain qualifier.
  /// @param TypeAttributeEnum qual - qualifier to check.
  /// @return true if the qualifier exists and false otherwise.
  bool hasQualifier(TypeAttributeEnum Qual) const;

private:
  /// The type this pointer is pointing at.
  RefParamType PType;
  /// Array of the pointer's enabled type qualifiers.
  bool Qualifiers[ATTR_QUALIFIER_LAST - ATTR_QUALIFIER_FIRST + 1];
  /// Pointer's address space.
  TypeAttributeEnum AddressSpace;
};

struct VectorType : public ParamType {
  /// An enumeration to identify the type id of this class.
  const static TypeEnum EnumTy;

  /// @brief Constructor.
  /// @param RefParamType the type of each scalar element in the vector.
  /// @param int the length of the vector.
  VectorType(const RefParamType Type, int Len);

  /// Implementation of Abstract Methods ///

  /// @brief Visitor service method. (see TypeVisitor for more details).
  ///        When overridden in subclasses, preform a 'double dispatch' to the
  ///        appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor.
  MangleError accept(TypeVisitor *) const override;

  /// @brief Returns a string representation of the underlying type.
  /// @return type as string.
  std::string toString() const override;

  /// @brief Returns true if given param type is equal to this type.
  /// @param ParamType given param type.
  /// @return true if given param type is equal to this type and false
  /// otherwise.
  bool equals(const ParamType *) const override;

  /// Non-Common Methods ///

  /// @brief Returns the type the vector is packing.
  /// @return scalar type.
  const RefParamType &getScalarType() const { return PType; }

  /// @brief Returns the length of the vector type.
  /// @return vector type length.
  int getLength() const { return Len; }

private:
  /// The scalar type of this vector type.
  RefParamType PType;
  /// The length of the vector.
  int Len;
};

struct AtomicType : public ParamType {
  /// an enumeration to identify the type id of this class
  const static TypeEnum EnumTy;

  /// @brief Constructor
  /// @param RefParamType the type refernced as atomic.
  AtomicType(const RefParamType Type);

  /// Implementation of Abstract Methods ///

  /// @brief visitor service method. (see TypeVisitor for more details).
  ///       When overridden in subclasses, preform a 'double dispatch' to the
  ///       appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor
  MangleError accept(TypeVisitor *) const override;

  /// @brief returns a string representation of the underlying type.
  /// @return type as string
  std::string toString() const override;

  /// @brief returns true if given param type is equal to this type.
  /// @param ParamType given param type
  /// @return true if given param type is equal to this type and false otherwise
  bool equals(const ParamType *) const override;

  /// Non-Common Methods ///

  /// @brief returns the base type of the atomic parameter.
  /// @return base type
  const RefParamType &getBaseType() const { return PType; }

private:
  /// the type this pointer is pointing at
  RefParamType PType;
};

struct BlockType : public ParamType {
  /// an enumeration to identify the type id of this class
  const static TypeEnum EnumTy;

  ///@brief Constructor
  BlockType();

  /// Implementation of Abstract Methods ///

  /// @brief visitor service method. (see TypeVisitor for more details).
  ///       When overridden in subclasses, preform a 'double dispatch' to the
  ///       appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor
  MangleError accept(TypeVisitor *) const override;

  /// @brief returns a string representation of the underlying type.
  /// @return type as string
  std::string toString() const override;

  /// @brief returns true if given param type is equal to this type.
  /// @param ParamType given param type
  /// @return true if given param type is equal to this type and false otherwise
  bool equals(const ParamType *) const override;

  /// Non-Common Methods ///

  /// @brief returns the number of parameters of the block.
  /// @return parameters count
  unsigned int getNumOfParams() const { return (unsigned int)Params.size(); }

  ///@brief returns the type of parameter "index" of the block.
  // @param index the sequential number of the queried parameter
  ///@return parameter type
  const RefParamType &getParam(unsigned int Index) const {
    assert(Params.size() > Index && "index is OOB");
    return Params[Index];
  }

  ///@brief set the type of parameter "index" of the block.
  // @param index the sequential number of the queried parameter
  // @param type the parameter type
  void setParam(unsigned int Index, RefParamType Type) {
    if (Index < getNumOfParams()) {
      Params[Index] = Type;
    } else if (Index == getNumOfParams()) {
      Params.push_back(Type);
    } else {
      assert(false && "index is OOB");
    }
  }

protected:
  /// an enumeration to identify the primitive type
  std::vector<RefParamType> Params;
};

struct UserDefinedType : public ParamType {
  /// An enumeration to identify the type id of this class.
  const static TypeEnum EnumTy;

  /// @brief Constructor.
  UserDefinedType(const std::string &);

  /// Implementation of Abstract Methods ///

  /// @brief Visitor service method. (see TypeVisitor for more details).
  ///        When overridden in subclasses, preform a 'double dispatch' to the
  ///        appropriate visit method in the given visitor.
  /// @param TypeVisitor type visitor.
  MangleError accept(TypeVisitor *) const override;

  /// @brief Returns a string representation of the underlying type.
  /// @return type as string.
  std::string toString() const override;

  /// @brief Returns true if given param type is equal to this type.
  /// @param ParamType given param type.
  /// @return true if given param type is equal to this type and false
  /// otherwise.
  bool equals(const ParamType *) const override;

protected:
  /// The name of the user defined type.
  std::string Name;
};

/// @brief Can be overridden so an object of static type Type* will
///        dispatch the correct visit method according to its dynamic type.
struct TypeVisitor {
  SPIRversion SpirVer;
  TypeVisitor(SPIRversion Ver) : SpirVer(Ver) {}
  virtual ~TypeVisitor() {}
  virtual MangleError visit(const PrimitiveType *) = 0;
  virtual MangleError visit(const VectorType *) = 0;
  virtual MangleError visit(const PointerType *) = 0;
  virtual MangleError visit(const AtomicType *) = 0;
  virtual MangleError visit(const BlockType *) = 0;
  virtual MangleError visit(const UserDefinedType *) = 0;
};

/// @brief Template dynamic cast function for ParamType derived classes.
/// @param ParamType given param type.
/// @return required casting type if given param type is an instance if
//          that type, NULL otherwise.
template <typename T> T *dynCast(ParamType *PType) {
  assert(PType && "dyn_cast does not support casting of NULL");
  return (T::EnumTy == PType->getTypeId()) ? (T *)PType : NULL;
}

/// @brief Template dynamic cast function for ParamType derived classes
///        (the constant version).
/// @param ParamType given param type.
/// @return required casting type if given param type is an instance if
//          that type, NULL otherwise.
template <typename T> const T *dynCast(const ParamType *PType) {
  assert(PType && "dyn_cast does not support casting of NULL");
  return (T::EnumTy == PType->getTypeId()) ? (const T *)PType : NULL;
}

} // namespace SPIR
#endif // SPIRV_MANGLER_PARAMETERTYPE_H
