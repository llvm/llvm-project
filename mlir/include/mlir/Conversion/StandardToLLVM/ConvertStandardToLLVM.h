//===- ConvertStandardToLLVM.h - Convert to the LLVM dialect ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a dialect conversion targeting the LLVM IR dialect.  By default, it
// converts Standard ops and types and provides hooks for dialect-specific
// extensions to the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace llvm {
class IntegerType;
class LLVMContext;
class Module;
class Type;
} // namespace llvm

namespace mlir {

class LLVMTypeConverter;
class UnrankedMemRefType;

namespace LLVM {
class LLVMDialect;
class LLVMType;
} // namespace LLVM

/// Set of callbacks that allows the customization of LLVMTypeConverter.
struct LLVMTypeConverterCustomization {
  using CustomCallback = std::function<LogicalResult(LLVMTypeConverter &, Type,
                                                     SmallVectorImpl<Type> &)>;

  /// Customize the type conversion of function arguments.
  CustomCallback funcArgConverter;

  /// Initialize customization to default callbacks.
  LLVMTypeConverterCustomization();
};

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult structFuncArgTypeConverter(LLVMTypeConverter &converter,
                                         Type type,
                                         SmallVectorImpl<Type> &result);

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult barePtrFuncArgTypeConverter(LLVMTypeConverter &converter,
                                          Type type,
                                          SmallVectorImpl<Type> &result);

/// Conversion from types in the Standard dialect to the LLVM IR dialect.
class LLVMTypeConverter : public TypeConverter {
  /// Give structFuncArgTypeConverter access to memref-specific functions.
  friend LogicalResult
  structFuncArgTypeConverter(LLVMTypeConverter &converter, Type type,
                             SmallVectorImpl<Type> &result);

public:
  using TypeConverter::convertType;

  /// Create an LLVMTypeConverter using the default
  /// LLVMTypeConverterCustomization.
  LLVMTypeConverter(MLIRContext *ctx);

  /// Create an LLVMTypeConverter using 'custom' customizations.
  LLVMTypeConverter(MLIRContext *ctx,
                    const LLVMTypeConverterCustomization &custom);

  /// Convert a function type.  The arguments and results are converted one by
  /// one and results are packed into a wrapped LLVM IR structure type. `result`
  /// is populated with argument mapping.
  LLVM::LLVMType convertFunctionSignature(FunctionType type, bool isVariadic,
                                          SignatureConversion &result);

  /// Convert a non-empty list of types to be returned from a function into a
  /// supported LLVM IR type.  In particular, if more than one values is
  /// returned, create an LLVM IR structure type with elements that correspond
  /// to each of the MLIR types converted with `convertType`.
  Type packFunctionResults(ArrayRef<Type> types);

  /// Returns the LLVM context.
  llvm::LLVMContext &getLLVMContext();

  /// Returns the LLVM dialect.
  LLVM::LLVMDialect *getDialect() { return llvmDialect; }

  /// Promote the LLVM struct representation of all MemRef descriptors to stack
  /// and use pointers to struct to avoid the complexity of the
  /// platform-specific C/C++ ABI lowering related to struct argument passing.
  SmallVector<Value, 4> promoteMemRefDescriptors(Location loc,
                                                 ValueRange opOperands,
                                                 ValueRange operands,
                                                 OpBuilder &builder);

  /// Promote the LLVM struct representation of one MemRef descriptor to stack
  /// and use pointer to struct to avoid the complexity of the platform-specific
  /// C/C++ ABI lowering related to struct argument passing.
  Value promoteOneMemRefDescriptor(Location loc, Value operand,
                                   OpBuilder &builder);

  /// Converts the function type to a C-compatible format, in particular using
  /// pointers to memref descriptors for arguments.
  LLVM::LLVMType convertFunctionTypeCWrapper(FunctionType type);

  /// Creates descriptor structs from individual values constituting them.
  Operation *materializeConversion(PatternRewriter &rewriter, Type type,
                                   ArrayRef<Value> values,
                                   Location loc) override;

protected:
  /// LLVM IR module used to parse/create types.
  llvm::Module *module;
  LLVM::LLVMDialect *llvmDialect;

private:
  // Convert a function type.  The arguments and results are converted one by
  // one.  Additionally, if the function returns more than one value, pack the
  // results into an LLVM IR structure type so that the converted function type
  // returns at most one result.
  Type convertFunctionType(FunctionType type);

  // Convert the index type.  Uses llvmModule data layout to create an integer
  // of the pointer bitwidth.
  Type convertIndexType(IndexType type);

  // Convert an integer type `i*` to `!llvm<"i*">`.
  Type convertIntegerType(IntegerType type);

  // Convert a floating point type: `f16` to `!llvm.half`, `f32` to
  // `!llvm.float` and `f64` to `!llvm.double`.  `bf16` is not supported
  // by LLVM.
  Type convertFloatType(FloatType type);

  /// Convert a memref type into an LLVM type that captures the relevant data.
  Type convertMemRefType(MemRefType type);

  /// Convert a memref type into a list of non-aggregate LLVM IR types that
  /// contain all the relevant data. In particular, the list will contain:
  /// - two pointers to the memref element type, followed by
  /// - an integer offset, followed by
  /// - one integer size per dimension of the memref, followed by
  /// - one integer stride per dimension of the memref.
  /// For example, memref<?x?xf32> is converted to the following list:
  /// - `!llvm<"float*">` (allocated pointer),
  /// - `!llvm<"float*">` (aligned pointer),
  /// - `!llvm.i64` (offset),
  /// - `!llvm.i64`, `!llvm.i64` (sizes),
  /// - `!llvm.i64`, `!llvm.i64` (strides).
  /// These types can be recomposed to a memref descriptor struct.
  SmallVector<Type, 5> convertMemRefSignature(MemRefType type);

  /// Convert an unranked memref type into a list of non-aggregate LLVM IR types
  /// that contain all the relevant data. In particular, this list contains:
  /// - an integer rank, followed by
  /// - a pointer to the memref descriptor struct.
  /// For example, memref<*xf32> is converted to the following list:
  /// !llvm.i64 (rank)
  /// !llvm<"i8*"> (type-erased pointer).
  /// These types can be recomposed to a unranked memref descriptor struct.
  SmallVector<Type, 2> convertUnrankedMemRefSignature();

  // Convert an unranked memref type to an LLVM type that captures the
  // runtime rank and a pointer to the static ranked memref desc
  Type convertUnrankedMemRefType(UnrankedMemRefType type);

  // Convert a 1D vector type into an LLVM vector type.
  Type convertVectorType(VectorType type);

  // Get the LLVM representation of the index type based on the bitwidth of the
  // pointer as defined by the data layout of the module.
  LLVM::LLVMType getIndexType();

  /// Callbacks for customizing the type conversion.
  LLVMTypeConverterCustomization customizations;
};

/// Helper class to produce LLVM dialect operations extracting or inserting
/// values to a struct.
class StructBuilder {
public:
  /// Construct a helper for the given value.
  explicit StructBuilder(Value v);
  /// Builds IR creating an `undef` value of the descriptor type.
  static StructBuilder undef(OpBuilder &builder, Location loc,
                             Type descriptorType);

  /*implicit*/ operator Value() { return value; }

protected:
  // LLVM value
  Value value;
  // Cached struct type.
  Type structType;

protected:
  /// Builds IR to extract a value from the struct at position pos
  Value extractPtr(OpBuilder &builder, Location loc, unsigned pos);
  /// Builds IR to set a value in the struct at position pos
  void setPtr(OpBuilder &builder, Location loc, unsigned pos, Value ptr);
};

/// Helper class to produce LLVM dialect operations extracting or inserting
/// elements of a MemRef descriptor. Wraps a Value pointing to the descriptor.
/// The Value may be null, in which case none of the operations are valid.
class MemRefDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit MemRefDescriptor(Value descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static MemRefDescriptor undef(OpBuilder &builder, Location loc,
                                Type descriptorType);
  /// Builds IR creating a MemRef descriptor that represents `type` and
  /// populates it with static shape and stride information extracted from the
  /// type.
  static MemRefDescriptor fromStaticShape(OpBuilder &builder, Location loc,
                                          LLVMTypeConverter &typeConverter,
                                          MemRefType type, Value memory);

  /// Builds IR extracting the allocated pointer from the descriptor.
  Value allocatedPtr(OpBuilder &builder, Location loc);
  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &builder, Location loc, Value ptr);

  /// Builds IR extracting the aligned pointer from the descriptor.
  Value alignedPtr(OpBuilder &builder, Location loc);

  /// Builds IR inserting the aligned pointer into the descriptor.
  void setAlignedPtr(OpBuilder &builder, Location loc, Value ptr);

  /// Builds IR extracting the offset from the descriptor.
  Value offset(OpBuilder &builder, Location loc);

  /// Builds IR inserting the offset into the descriptor.
  void setOffset(OpBuilder &builder, Location loc, Value offset);
  void setConstantOffset(OpBuilder &builder, Location loc, uint64_t offset);

  /// Builds IR extracting the pos-th size from the descriptor.
  Value size(OpBuilder &builder, Location loc, unsigned pos);

  /// Builds IR inserting the pos-th size into the descriptor
  void setSize(OpBuilder &builder, Location loc, unsigned pos, Value size);
  void setConstantSize(OpBuilder &builder, Location loc, unsigned pos,
                       uint64_t size);

  /// Builds IR extracting the pos-th size from the descriptor.
  Value stride(OpBuilder &builder, Location loc, unsigned pos);

  /// Builds IR inserting the pos-th stride into the descriptor
  void setStride(OpBuilder &builder, Location loc, unsigned pos, Value stride);
  void setConstantStride(OpBuilder &builder, Location loc, unsigned pos,
                         uint64_t stride);

  /// Returns the (LLVM) type this descriptor points to.
  LLVM::LLVMType getElementType();

  /// Builds IR populating a MemRef descriptor structure from a list of
  /// individual values composing that descriptor, in the following order:
  /// - allocated pointer;
  /// - aligned pointer;
  /// - offset;
  /// - <rank> sizes;
  /// - <rank> shapes;
  /// where <rank> is the MemRef rank as provided in `type`.
  static Value pack(OpBuilder &builder, Location loc,
                    LLVMTypeConverter &converter, MemRefType type,
                    ValueRange values);

  /// Builds IR extracting individual elements of a MemRef descriptor structure
  /// and returning them as `results` list.
  static void unpack(OpBuilder &builder, Location loc, Value packed,
                     MemRefType type, SmallVectorImpl<Value> &results);

  /// Returns the number of non-aggregate values that would be produced by
  /// `unpack`.
  static unsigned getNumUnpackedValues(MemRefType type);

private:
  // Cached index type.
  Type indexType;
};

/// Helper class allowing the user to access a range of Values that correspond
/// to an unpacked memref descriptor using named accessors. This does not own
/// the values.
class MemRefDescriptorView {
public:
  /// Constructs the view from a range of values. Infers the rank from the size
  /// of the range.
  explicit MemRefDescriptorView(ValueRange range);

  /// Returns the allocated pointer Value.
  Value allocatedPtr();

  /// Returns the aligned pointer Value.
  Value alignedPtr();

  /// Returns the offset Value.
  Value offset();

  /// Returns the pos-th size Value.
  Value size(unsigned pos);

  /// Returns the pos-th stride Value.
  Value stride(unsigned pos);

private:
  /// Rank of the memref the descriptor is pointing to.
  int rank;
  /// Underlying range of Values.
  ValueRange elements;
};

class UnrankedMemRefDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit UnrankedMemRefDescriptor(Value descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static UnrankedMemRefDescriptor undef(OpBuilder &builder, Location loc,
                                        Type descriptorType);

  /// Builds IR extracting the rank from the descriptor
  Value rank(OpBuilder &builder, Location loc);
  /// Builds IR setting the rank in the descriptor
  void setRank(OpBuilder &builder, Location loc, Value value);
  /// Builds IR extracting ranked memref descriptor ptr
  Value memRefDescPtr(OpBuilder &builder, Location loc);
  /// Builds IR setting ranked memref descriptor ptr
  void setMemRefDescPtr(OpBuilder &builder, Location loc, Value value);

  /// Builds IR populating an unranked MemRef descriptor structure from a list
  /// of individual constituent values in the following order:
  /// - rank of the memref;
  /// - pointer to the memref descriptor.
  static Value pack(OpBuilder &builder, Location loc,
                    LLVMTypeConverter &converter, UnrankedMemRefType type,
                    ValueRange values);

  /// Builds IR extracting individual elements that compose an unranked memref
  /// descriptor and returns them as `results` list.
  static void unpack(OpBuilder &builder, Location loc, Value packed,
                     SmallVectorImpl<Value> &results);

  /// Returns the number of non-aggregate values that would be produced by
  /// `unpack`.
  static unsigned getNumUnpackedValues() { return 2; }
};
/// Base class for operation conversions targeting the LLVM IR dialect. Provides
/// conversion patterns with access to an LLVMTypeConverter.
class ConvertToLLVMPattern : public ConversionPattern {
public:
  ConvertToLLVMPattern(StringRef rootOpName, MLIRContext *context,
                       LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1);

protected:
  /// Reference to the type converter, with potential extensions.
  LLVMTypeConverter &typeConverter;
};

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
