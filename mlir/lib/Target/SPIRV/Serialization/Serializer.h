//===- Serializer.h - MLIR SPIR-V Serializer ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MLIR SPIR-V module to SPIR-V binary serializer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_SPIRV_SERIALIZATION_SERIALIZER_H
#define MLIR_LIB_TARGET_SPIRV_SERIALIZATION_SERIALIZER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace spirv {

void encodeInstructionInto(SmallVectorImpl<uint32_t> &binary, spirv::Opcode op,
                           ArrayRef<uint32_t> operands);

/// A SPIR-V module serializer.
///
/// A SPIR-V binary module is a single linear stream of instructions; each
/// instruction is composed of 32-bit words with the layout:
///
///   | <word-count>|<opcode> |  <operand>   |  <operand>   | ... |
///   | <------ word -------> | <-- word --> | <-- word --> | ... |
///
/// For the first word, the 16 high-order bits are the word count of the
/// instruction, the 16 low-order bits are the opcode enumerant. The
/// instructions then belong to different sections, which must be laid out in
/// the particular order as specified in "2.4 Logical Layout of a Module" of
/// the SPIR-V spec.
class Serializer {
public:
  /// Creates a serializer for the given SPIR-V `module`.
  explicit Serializer(spirv::ModuleOp module,
                      const SerializationOptions &options);

  /// Serializes the remembered SPIR-V module.
  LogicalResult serialize();

  /// Collects the final SPIR-V `binary`.
  void collect(SmallVectorImpl<uint32_t> &binary);

#ifndef NDEBUG
  /// (For debugging) prints each value and its corresponding result <id>.
  void printValueIDMap(raw_ostream &os);
#endif

private:
  // Note that there are two main categories of methods in this class:
  // * process*() methods are meant to fully serialize a SPIR-V module entity
  //   (header, type, op, etc.). They update internal vectors containing
  //   different binary sections. They are not meant to be called except the
  //   top-level serialization loop.
  // * prepare*() methods are meant to be helpers that prepare for serializing
  //   certain entity. They may or may not update internal vectors containing
  //   different binary sections. They are meant to be called among themselves
  //   or by other process*() methods for subtasks.

  //===--------------------------------------------------------------------===//
  // <id>
  //===--------------------------------------------------------------------===//

  // Note that it is illegal to use id <0> in SPIR-V binary module. Various
  // methods in this class, if using SPIR-V word (uint32_t) as interface,
  // check or return id <0> to indicate error in processing.

  /// Consumes the next unused <id>. This method will never return 0.
  uint32_t getNextID() { return nextID++; }

  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  uint32_t getSpecConstID(StringRef constName) const {
    return specConstIDMap.lookup(constName);
  }

  uint32_t getVariableID(StringRef varName) const {
    return globalVarIDMap.lookup(varName);
  }

  uint32_t getFunctionID(StringRef fnName) const {
    return funcIDMap.lookup(fnName);
  }

  /// Gets the <id> for the function with the given name. Assigns the next
  /// available <id> if the function haven't been deserialized.
  uint32_t getOrCreateFunctionID(StringRef fnName);

  void processCapability();

  void processDebugInfo();

  void processExtension();

  void processMemoryModel();

  LogicalResult processConstantOp(spirv::ConstantOp op);

  LogicalResult processSpecConstantOp(spirv::SpecConstantOp op);

  LogicalResult
  processSpecConstantCompositeOp(spirv::SpecConstantCompositeOp op);

  LogicalResult
  processSpecConstantOperationOp(spirv::SpecConstantOperationOp op);

  /// SPIR-V dialect supports OpUndef using spv.UndefOp that produces a SSA
  /// value to use with other operations. The SPIR-V spec recommends that
  /// OpUndef be generated at module level. The serialization generates an
  /// OpUndef for each type needed at module level.
  LogicalResult processUndefOp(spirv::UndefOp op);

  /// Emit OpName for the given `resultID`.
  LogicalResult processName(uint32_t resultID, StringRef name);

  /// Processes a SPIR-V function op.
  LogicalResult processFuncOp(spirv::FuncOp op);

  LogicalResult processVariableOp(spirv::VariableOp op);

  /// Process a SPIR-V GlobalVariableOp
  LogicalResult processGlobalVariableOp(spirv::GlobalVariableOp varOp);

  /// Process attributes that translate to decorations on the result <id>
  LogicalResult processDecoration(Location loc, uint32_t resultID,
                                  NamedAttribute attr);

  template <typename DType>
  LogicalResult processTypeDecoration(Location loc, DType type,
                                      uint32_t resultId) {
    return emitError(loc, "unhandled decoration for type:") << type;
  }

  /// Process member decoration
  LogicalResult processMemberDecoration(
      uint32_t structID,
      const spirv::StructType::MemberDecorationInfo &memberDecorationInfo);

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  uint32_t getTypeID(Type type) const { return typeIDMap.lookup(type); }

  Type getVoidType() { return mlirBuilder.getNoneType(); }

  bool isVoidType(Type type) const { return type.isa<NoneType>(); }

  /// Returns true if the given type is a pointer type to a struct in some
  /// interface storage class.
  bool isInterfaceStructPtrType(Type type) const;

  /// Main dispatch method for serializing a type. The result <id> of the
  /// serialized type will be returned as `typeID`.
  LogicalResult processType(Location loc, Type type, uint32_t &typeID);
  LogicalResult processTypeImpl(Location loc, Type type, uint32_t &typeID,
                                SetVector<StringRef> &serializationCtx);

  /// Method for preparing basic SPIR-V type serialization. Returns the type's
  /// opcode and operands for the instruction via `typeEnum` and `operands`.
  LogicalResult prepareBasicType(Location loc, Type type, uint32_t resultID,
                                 spirv::Opcode &typeEnum,
                                 SmallVectorImpl<uint32_t> &operands,
                                 bool &deferSerialization,
                                 SetVector<StringRef> &serializationCtx);

  LogicalResult prepareFunctionType(Location loc, FunctionType type,
                                    spirv::Opcode &typeEnum,
                                    SmallVectorImpl<uint32_t> &operands);

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  uint32_t getConstantID(Attribute value) const {
    return constIDMap.lookup(value);
  }

  /// Main dispatch method for processing a constant with the given `constType`
  /// and `valueAttr`. `constType` is needed here because we can interpret the
  /// `valueAttr` as a different type than the type of `valueAttr` itself; for
  /// example, ArrayAttr, whose type is NoneType, is used for spirv::ArrayType
  /// constants.
  uint32_t prepareConstant(Location loc, Type constType, Attribute valueAttr);

  /// Prepares array attribute serialization. This method emits corresponding
  /// OpConstant* and returns the result <id> associated with it. Returns 0 if
  /// failed.
  uint32_t prepareArrayConstant(Location loc, Type constType, ArrayAttr attr);

  /// Prepares bool/int/float DenseElementsAttr serialization. This method
  /// iterates the DenseElementsAttr to construct the constant array, and
  /// returns the result <id>  associated with it. Returns 0 if failed. Note
  /// that the size of `index` must match the rank.
  /// TODO: Consider to enhance splat elements cases. For splat cases,
  /// we don't need to loop over all elements, especially when the splat value
  /// is zero. We can use OpConstantNull when the value is zero.
  uint32_t prepareDenseElementsConstant(Location loc, Type constType,
                                        DenseElementsAttr valueAttr, int dim,
                                        MutableArrayRef<uint64_t> index);

  /// Prepares scalar attribute serialization. This method emits corresponding
  /// OpConstant* and returns the result <id> associated with it. Returns 0 if
  /// the attribute is not for a scalar bool/integer/float value. If `isSpec` is
  /// true, then the constant will be serialized as a specialization constant.
  uint32_t prepareConstantScalar(Location loc, Attribute valueAttr,
                                 bool isSpec = false);

  uint32_t prepareConstantBool(Location loc, BoolAttr boolAttr,
                               bool isSpec = false);

  uint32_t prepareConstantInt(Location loc, IntegerAttr intAttr,
                              bool isSpec = false);

  uint32_t prepareConstantFp(Location loc, FloatAttr floatAttr,
                             bool isSpec = false);

  //===--------------------------------------------------------------------===//
  // Control flow
  //===--------------------------------------------------------------------===//

  /// Returns the result <id> for the given block.
  uint32_t getBlockID(Block *block) const { return blockIDMap.lookup(block); }

  /// Returns the result <id> for the given block. If no <id> has been assigned,
  /// assigns the next available <id>
  uint32_t getOrCreateBlockID(Block *block);

#ifndef NDEBUG
  /// (For debugging) prints the block with its result <id>.
  void printBlock(Block *block, raw_ostream &os);
#endif

  /// Processes the given `block` and emits SPIR-V instructions for all ops
  /// inside. Does not emit OpLabel for this block if `omitLabel` is true.
  /// `emitMerge` is a callback that will be invoked before handling the
  /// terminator op to inject the Op*Merge instruction if this is a SPIR-V
  /// selection/loop header block.
  LogicalResult processBlock(Block *block, bool omitLabel = false,
                             function_ref<LogicalResult()> emitMerge = nullptr);

  /// Emits OpPhi instructions for the given block if it has block arguments.
  LogicalResult emitPhiForBlockArguments(Block *block);

  LogicalResult processSelectionOp(spirv::SelectionOp selectionOp);

  LogicalResult processLoopOp(spirv::LoopOp loopOp);

  LogicalResult processBranchConditionalOp(spirv::BranchConditionalOp);

  LogicalResult processBranchOp(spirv::BranchOp branchOp);

  //===--------------------------------------------------------------------===//
  // Operations
  //===--------------------------------------------------------------------===//

  LogicalResult encodeExtensionInstruction(Operation *op,
                                           StringRef extensionSetName,
                                           uint32_t opcode,
                                           ArrayRef<uint32_t> operands);

  uint32_t getValueID(Value val) const { return valueIDMap.lookup(val); }

  LogicalResult processAddressOfOp(spirv::AddressOfOp addressOfOp);

  LogicalResult processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp);

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op);

  /// Serializes an operation `op` as core instruction with `opcode` if
  /// `extInstSet` is empty. Otherwise serializes it as an extended instruction
  /// with `opcode` from `extInstSet`.
  /// This method is a generic one for dispatching any SPIR-V ops that has no
  /// variadic operands and attributes in TableGen definitions.
  LogicalResult processOpWithoutGrammarAttr(Operation *op, StringRef extInstSet,
                                            uint32_t opcode);

  /// Dispatches to the serialization function for an operation in SPIR-V
  /// dialect that is a mirror of an instruction in the SPIR-V spec. This is
  /// auto-generated from ODS. Dispatch is handled for all operations in SPIR-V
  /// dialect that have hasOpcode == 1.
  LogicalResult dispatchToAutogenSerialization(Operation *op);

  /// Serializes an operation in the SPIR-V dialect that is a mirror of an
  /// instruction in the SPIR-V spec. This is auto generated if hasOpcode == 1
  /// and autogenSerialization == 1 in ODS.
  template <typename OpTy> LogicalResult processOp(OpTy op) {
    return op.emitError("unsupported op serialization");
  }

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Emits an OpDecorate instruction to decorate the given `target` with the
  /// given `decoration`.
  LogicalResult emitDecoration(uint32_t target, spirv::Decoration decoration,
                               ArrayRef<uint32_t> params = {});

  /// Emits an OpLine instruction with the given `loc` location information into
  /// the given `binary` vector.
  LogicalResult emitDebugLine(SmallVectorImpl<uint32_t> &binary, Location loc);

private:
  /// The SPIR-V module to be serialized.
  spirv::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;

  /// Serialization options.
  SerializationOptions options;

  /// A flag which indicates if the last processed instruction was a merge
  /// instruction.
  /// According to SPIR-V spec: "If a branch merge instruction is used, the last
  /// OpLine in the block must be before its merge instruction".
  bool lastProcessedWasMergeInst = false;

  /// The <id> of the OpString instruction, which specifies a file name, for
  /// use by other debug instructions.
  uint32_t fileID = 0;

  /// The next available result <id>.
  uint32_t nextID = 1;

  // The following are for different SPIR-V instruction sections. They follow
  // the logical layout of a SPIR-V module.

  SmallVector<uint32_t, 4> capabilities;
  SmallVector<uint32_t, 0> extensions;
  SmallVector<uint32_t, 0> extendedSets;
  SmallVector<uint32_t, 3> memoryModel;
  SmallVector<uint32_t, 0> entryPoints;
  SmallVector<uint32_t, 4> executionModes;
  SmallVector<uint32_t, 0> debug;
  SmallVector<uint32_t, 0> names;
  SmallVector<uint32_t, 0> decorations;
  SmallVector<uint32_t, 0> typesGlobalValues;
  SmallVector<uint32_t, 0> functions;

  /// Recursive struct references are serialized as OpTypePointer instructions
  /// to the recursive struct type. However, the OpTypePointer instruction
  /// cannot be emitted before the recursive struct's OpTypeStruct.
  /// RecursiveStructPointerInfo stores the data needed to emit such
  /// OpTypePointer instructions after forward references to such types.
  struct RecursiveStructPointerInfo {
    uint32_t pointerTypeID;
    spirv::StorageClass storageClass;
  };

  // Maps spirv::StructType to its recursive reference member info.
  DenseMap<Type, SmallVector<RecursiveStructPointerInfo, 0>>
      recursiveStructInfos;

  /// `functionHeader` contains all the instructions that must be in the first
  /// block in the function, and `functionBody` contains the rest. After
  /// processing FuncOp, the encoded instructions of a function are appended to
  /// `functions`. An example of instructions in `functionHeader` in order:
  /// OpFunction ...
  /// OpFunctionParameter ...
  /// OpFunctionParameter ...
  /// OpLabel ...
  /// OpVariable ...
  /// OpVariable ...
  SmallVector<uint32_t, 0> functionHeader;
  SmallVector<uint32_t, 0> functionBody;

  /// Map from type used in SPIR-V module to their <id>s.
  DenseMap<Type, uint32_t> typeIDMap;

  /// Map from constant values to their <id>s.
  DenseMap<Attribute, uint32_t> constIDMap;

  /// Map from specialization constant names to their <id>s.
  llvm::StringMap<uint32_t> specConstIDMap;

  /// Map from GlobalVariableOps name to <id>s.
  llvm::StringMap<uint32_t> globalVarIDMap;

  /// Map from FuncOps name to <id>s.
  llvm::StringMap<uint32_t> funcIDMap;

  /// Map from blocks to their <id>s.
  DenseMap<Block *, uint32_t> blockIDMap;

  /// Map from the Type to the <id> that represents undef value of that type.
  DenseMap<Type, uint32_t> undefValIDMap;

  /// Map from results of normal operations to their <id>s.
  DenseMap<Value, uint32_t> valueIDMap;

  /// Map from extended instruction set name to <id>s.
  llvm::StringMap<uint32_t> extendedInstSetIDMap;

  /// Map from values used in OpPhi instructions to their offset in the
  /// `functions` section.
  ///
  /// When processing a block with arguments, we need to emit OpPhi
  /// instructions to record the predecessor block <id>s and the values they
  /// send to the block in question. But it's not guaranteed all values are
  /// visited and thus assigned result <id>s. So we need this list to capture
  /// the offsets into `functions` where a value is used so that we can fix it
  /// up later after processing all the blocks in a function.
  ///
  /// More concretely, say if we are visiting the following blocks:
  ///
  /// ```mlir
  /// ^phi(%arg0: i32):
  ///   ...
  /// ^parent1:
  ///   ...
  ///   spv.Branch ^phi(%val0: i32)
  /// ^parent2:
  ///   ...
  ///   spv.Branch ^phi(%val1: i32)
  /// ```
  ///
  /// When we are serializing the `^phi` block, we need to emit at the beginning
  /// of the block OpPhi instructions which has the following parameters:
  ///
  /// OpPhi id-for-i32 id-for-%arg0 id-for-%val0 id-for-^parent1
  ///                               id-for-%val1 id-for-^parent2
  ///
  /// But we don't know the <id> for %val0 and %val1 yet. One way is to visit
  /// all the blocks twice and use the first visit to assign an <id> to each
  /// value. But it's paying the overheads just for OpPhi emission. Instead,
  /// we still visit the blocks once for emission. When we emit the OpPhi
  /// instructions, we use 0 as a placeholder for the <id>s for %val0 and %val1.
  /// At the same time, we record their offsets in the emitted binary (which is
  /// placed inside `functions`) here. And then after emitting all blocks, we
  /// replace the dummy <id> 0 with the real result <id> by overwriting
  /// `functions[offset]`.
  DenseMap<Value, SmallVector<size_t, 1>> deferredPhiValues;
};
} // namespace spirv
} // namespace mlir

#endif // MLIR_LIB_TARGET_SPIRV_SERIALIZATION_SERIALIZER_H
