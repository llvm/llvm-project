//===- Deserializer.h - MLIR SPIR-V Deserializer ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SPIR-V binary to MLIR SPIR-V module deserializer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_DESERIALIZER_H
#define MLIR_TARGET_SPIRV_DESERIALIZER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace spirv {

//===----------------------------------------------------------------------===//
// Utility Definitions
//===----------------------------------------------------------------------===//

/// A struct for containing a header block's merge and continue targets.
///
/// This struct is used to track original structured control flow info from
/// SPIR-V blob. This info will be used to create
/// spirv.mlir.selection/spirv.mlir.loop later.
struct BlockMergeInfo {
  Block *mergeBlock;
  Block *continueBlock; // nullptr for spirv.mlir.selection
  Location loc;
  uint32_t control; // Selection/loop control

  BlockMergeInfo(Location location, uint32_t control)
      : mergeBlock(nullptr), continueBlock(nullptr), loc(location),
        control(control) {}
  BlockMergeInfo(Location location, uint32_t control, Block *m,
                 Block *c = nullptr)
      : mergeBlock(m), continueBlock(c), loc(location), control(control) {}
};

/// A struct for containing OpLine instruction information.
struct DebugLine {
  uint32_t fileID;
  uint32_t line;
  uint32_t column;
};

/// Map from a selection/loop's header block to its merge (and continue) target.
using BlockMergeInfoMap = DenseMap<Block *, BlockMergeInfo>;

/// A "deferred struct type" is a struct type with one or more member types not
/// known when the Deserializer first encounters the struct. This happens, for
/// example, with recursive structs where a pointer to the struct type is
/// forward declared through OpTypeForwardPointer in the SPIR-V module before
/// the struct declaration; the actual pointer to struct type should be defined
/// later through an OpTypePointer. For example, the following C struct:
///
/// struct A {
///   A* next;
/// };
///
/// would be represented in the SPIR-V module as:
///
/// OpName %A "A"
/// OpTypeForwardPointer %APtr Generic
/// %A = OpTypeStruct %APtr
/// %APtr = OpTypePointer Generic %A
///
/// This means that the spirv::StructType cannot be fully constructed directly
/// when the Deserializer encounters it. Instead we create a
/// DeferredStructTypeInfo that contains all the information we know about the
/// spirv::StructType. Once all forward references for the struct are resolved,
/// the struct's body is set with all member info.
struct DeferredStructTypeInfo {
  spirv::StructType deferredStructType;

  // A list of all unresolved member types for the struct. First element of each
  // item is operand ID, second element is member index in the struct.
  SmallVector<std::pair<uint32_t, unsigned>, 0> unresolvedMemberTypes;

  // The list of member types. For unresolved members, this list contains
  // place-holder empty types that will be updated later.
  SmallVector<Type, 4> memberTypes;
  SmallVector<spirv::StructType::OffsetInfo, 0> offsetInfo;
  SmallVector<spirv::StructType::MemberDecorationInfo, 0> memberDecorationsInfo;
};

/// A struct that collects the info needed to materialize/emit a
/// SpecConstantOperation op.
struct SpecConstOperationMaterializationInfo {
  spirv::Opcode enclodesOpcode;
  uint32_t resultTypeID;
  SmallVector<uint32_t> enclosedOpOperands;
};

//===----------------------------------------------------------------------===//
// Deserializer Declaration
//===----------------------------------------------------------------------===//

/// A SPIR-V module serializer.
///
/// A SPIR-V binary module is a single linear stream of instructions; each
/// instruction is composed of 32-bit words. The first word of an instruction
/// records the total number of words of that instruction using the 16
/// higher-order bits. So this deserializer uses that to get instruction
/// boundary and parse instructions and build a SPIR-V ModuleOp gradually.
///
// TODO: clean up created ops on errors
class Deserializer {
public:
  /// Creates a deserializer for the given SPIR-V `binary` module.
  /// The SPIR-V ModuleOp will be created into `context.
  explicit Deserializer(ArrayRef<uint32_t> binary, MLIRContext *context);

  /// Deserializes the remembered SPIR-V binary module.
  LogicalResult deserialize();

  /// Collects the final SPIR-V ModuleOp.
  OwningOpRef<spirv::ModuleOp> collect();

private:
  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  /// Initializes the `module` ModuleOp in this deserializer instance.
  OwningOpRef<spirv::ModuleOp> createModuleOp();

  /// Processes SPIR-V module header in `binary`.
  LogicalResult processHeader();

  /// Processes the SPIR-V OpCapability with `operands` and updates bookkeeping
  /// in the deserializer.
  LogicalResult processCapability(ArrayRef<uint32_t> operands);

  /// Processes the SPIR-V OpExtension with `operands` and updates bookkeeping
  /// in the deserializer.
  LogicalResult processExtension(ArrayRef<uint32_t> words);

  /// Processes the SPIR-V OpExtInstImport with `operands` and updates
  /// bookkeeping in the deserializer.
  LogicalResult processExtInstImport(ArrayRef<uint32_t> words);

  /// Attaches (version, capabilities, extensions) triple to `module` as an
  /// attribute.
  void attachVCETriple();

  /// Processes the SPIR-V OpMemoryModel with `operands` and updates `module`.
  LogicalResult processMemoryModel(ArrayRef<uint32_t> operands);

  /// Process SPIR-V OpName with `operands`.
  LogicalResult processName(ArrayRef<uint32_t> operands);

  /// Processes an OpDecorate instruction.
  LogicalResult processDecoration(ArrayRef<uint32_t> words);

  // Processes an OpMemberDecorate instruction.
  LogicalResult processMemberDecoration(ArrayRef<uint32_t> words);

  /// Processes an OpMemberName instruction.
  LogicalResult processMemberName(ArrayRef<uint32_t> words);

  /// Gets the function op associated with a result <id> of OpFunction.
  spirv::FuncOp getFunction(uint32_t id) { return funcMap.lookup(id); }

  /// Processes the SPIR-V function at the current `offset` into `binary`.
  /// The operands to the OpFunction instruction is passed in as ``operands`.
  /// This method processes each instruction inside the function and dispatches
  /// them to their handler method accordingly.
  LogicalResult processFunction(ArrayRef<uint32_t> operands);

  /// Processes OpFunctionEnd and finalizes function. This wires up block
  /// argument created from OpPhi instructions and also structurizes control
  /// flow.
  LogicalResult processFunctionEnd(ArrayRef<uint32_t> operands);

  /// Gets the constant's attribute and type associated with the given <id>.
  std::optional<std::pair<Attribute, Type>> getConstant(uint32_t id);

  /// Gets the info needed to materialize the spec constant operation op
  /// associated with the given <id>.
  std::optional<SpecConstOperationMaterializationInfo>
  getSpecConstantOperation(uint32_t id);

  /// Gets the constant's integer attribute with the given <id>. Returns a
  /// null IntegerAttr if the given is not registered or does not correspond
  /// to an integer constant.
  IntegerAttr getConstantInt(uint32_t id);

  /// Returns a symbol to be used for the function name with the given
  /// result <id>. This tries to use the function's OpName if
  /// exists; otherwise creates one based on the <id>.
  std::string getFunctionSymbol(uint32_t id);

  /// Returns a symbol to be used for the specialization constant with the given
  /// result <id>. This tries to use the specialization constant's OpName if
  /// exists; otherwise creates one based on the <id>.
  std::string getSpecConstantSymbol(uint32_t id);

  /// Gets the specialization constant with the given result <id>.
  spirv::SpecConstantOp getSpecConstant(uint32_t id) {
    return specConstMap.lookup(id);
  }

  /// Gets the composite specialization constant with the given result <id>.
  spirv::SpecConstantCompositeOp getSpecConstantComposite(uint32_t id) {
    return specConstCompositeMap.lookup(id);
  }

  /// Creates a spirv::SpecConstantOp.
  spirv::SpecConstantOp createSpecConstant(Location loc, uint32_t resultID,
                                           TypedAttr defaultValue);

  /// Processes the OpVariable instructions at current `offset` into `binary`.
  /// It is expected that this method is used for variables that are to be
  /// defined at module scope and will be deserialized into a
  /// spirv.GlobalVariable instruction.
  LogicalResult processGlobalVariable(ArrayRef<uint32_t> operands);

  /// Gets the global variable associated with a result <id> of OpVariable.
  spirv::GlobalVariableOp getGlobalVariable(uint32_t id) {
    return globalVariableMap.lookup(id);
  }

  //===--------------------------------------------------------------------===//
  // Type
  //===--------------------------------------------------------------------===//

  /// Gets type for a given result <id>.
  Type getType(uint32_t id) { return typeMap.lookup(id); }

  /// Get the type associated with the result <id> of an OpUndef.
  Type getUndefType(uint32_t id) { return undefMap.lookup(id); }

  /// Returns true if the given `type` is for SPIR-V void type.
  bool isVoidType(Type type) const { return isa<NoneType>(type); }

  /// Processes a SPIR-V type instruction with given `opcode` and `operands` and
  /// registers the type into `module`.
  LogicalResult processType(spirv::Opcode opcode, ArrayRef<uint32_t> operands);

  LogicalResult processOpTypePointer(ArrayRef<uint32_t> operands);

  LogicalResult processArrayType(ArrayRef<uint32_t> operands);

  LogicalResult processCooperativeMatrixTypeKHR(ArrayRef<uint32_t> operands);

  LogicalResult processCooperativeMatrixTypeNV(ArrayRef<uint32_t> operands);

  LogicalResult processFunctionType(ArrayRef<uint32_t> operands);

  LogicalResult processJointMatrixType(ArrayRef<uint32_t> operands);

  LogicalResult processImageType(ArrayRef<uint32_t> operands);

  LogicalResult processSampledImageType(ArrayRef<uint32_t> operands);

  LogicalResult processRuntimeArrayType(ArrayRef<uint32_t> operands);

  LogicalResult processStructType(ArrayRef<uint32_t> operands);

  LogicalResult processMatrixType(ArrayRef<uint32_t> operands);

  LogicalResult processTypeForwardPointer(ArrayRef<uint32_t> operands);

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  /// Processes a SPIR-V Op{|Spec}Constant instruction with the given
  /// `operands`. `isSpec` indicates whether this is a specialization constant.
  LogicalResult processConstant(ArrayRef<uint32_t> operands, bool isSpec);

  /// Processes a SPIR-V Op{|Spec}Constant{True|False} instruction with the
  /// given `operands`. `isSpec` indicates whether this is a specialization
  /// constant.
  LogicalResult processConstantBool(bool isTrue, ArrayRef<uint32_t> operands,
                                    bool isSpec);

  /// Processes a SPIR-V OpConstantComposite instruction with the given
  /// `operands`.
  LogicalResult processConstantComposite(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpSpecConstantComposite instruction with the given
  /// `operands`.
  LogicalResult processSpecConstantComposite(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpSpecConstantOp instruction with the given
  /// `operands`.
  LogicalResult processSpecConstantOperation(ArrayRef<uint32_t> operands);

  /// Materializes/emits an OpSpecConstantOp instruction.
  Value materializeSpecConstantOperation(uint32_t resultID,
                                         spirv::Opcode enclosedOpcode,
                                         uint32_t resultTypeID,
                                         ArrayRef<uint32_t> enclosedOpOperands);

  /// Processes a SPIR-V OpConstantNull instruction with the given `operands`.
  LogicalResult processConstantNull(ArrayRef<uint32_t> operands);

  //===--------------------------------------------------------------------===//
  // Debug
  //===--------------------------------------------------------------------===//

  /// Discontinues any source-level location information that might be active
  /// from a previous OpLine instruction.
  void clearDebugLine();

  /// Creates a FileLineColLoc with the OpLine location information.
  Location createFileLineColLoc(OpBuilder opBuilder);

  /// Processes a SPIR-V OpLine instruction with the given `operands`.
  LogicalResult processDebugLine(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpString instruction with the given `operands`.
  LogicalResult processDebugString(ArrayRef<uint32_t> operands);

  //===--------------------------------------------------------------------===//
  // Control flow
  //===--------------------------------------------------------------------===//

  /// Returns the block for the given label <id>.
  Block *getBlock(uint32_t id) const { return blockMap.lookup(id); }

  // In SPIR-V, structured control flow is explicitly declared using merge
  // instructions (OpSelectionMerge and OpLoopMerge). In the SPIR-V dialect,
  // we use spirv.mlir.selection and spirv.mlir.loop to group structured control
  // flow. The deserializer need to turn structured control flow marked with
  // merge instructions into using spirv.mlir.selection/spirv.mlir.loop ops.
  //
  // Because structured control flow can nest and the basic block order have
  // flexibility, we cannot isolate a structured selection/loop without
  // deserializing all the blocks. So we use the following approach:
  //
  // 1. Deserialize all basic blocks in a function and create MLIR blocks for
  //    them into the function's region. In the meanwhile, keep a map between
  //    selection/loop header blocks to their corresponding merge (and continue)
  //    target blocks.
  // 2. For each selection/loop header block, recursively get all basic blocks
  //    reachable (except the merge block) and put them in a newly created
  //    spirv.mlir.selection/spirv.mlir.loop's region. Structured control flow
  //    guarantees that we enter and exit in structured ways and the construct
  //    is nestable.
  // 3. Put the new spirv.mlir.selection/spirv.mlir.loop op at the beginning of
  // the
  //    old merge block and redirect all branches to the old header block to the
  //    old merge block (which contains the spirv.mlir.selection/spirv.mlir.loop
  //    op now).

  /// For OpPhi instructions, we use block arguments to represent them. OpPhi
  /// encodes a list of (value, predecessor) pairs. At the time of handling the
  /// block containing an OpPhi instruction, the predecessor block might not be
  /// processed yet, also the value sent by it. So we need to defer handling
  /// the block argument from the predecessors. We use the following approach:
  ///
  /// 1. For each OpPhi instruction, add a block argument to the current block
  ///    in construction. Record the block argument in `valueMap` so its uses
  ///    can be resolved. For the list of (value, predecessor) pairs, update
  ///    `blockPhiInfo` for bookkeeping.
  /// 2. After processing all blocks, loop over `blockPhiInfo` to fix up each
  ///    block recorded there to create the proper block arguments on their
  ///    terminators.

  /// A data structure for containing a SPIR-V block's phi info. It will be
  /// represented as block argument in SPIR-V dialect.
  using BlockPhiInfo =
      SmallVector<uint32_t, 2>; // The result <id> of the values sent

  /// Gets or creates the block corresponding to the given label <id>. The newly
  /// created block will always be placed at the end of the current function.
  Block *getOrCreateBlock(uint32_t id);

  LogicalResult processBranch(ArrayRef<uint32_t> operands);

  LogicalResult processBranchConditional(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpLabel instruction with the given `operands`.
  LogicalResult processLabel(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpSelectionMerge instruction with the given `operands`.
  LogicalResult processSelectionMerge(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpLoopMerge instruction with the given `operands`.
  LogicalResult processLoopMerge(ArrayRef<uint32_t> operands);

  /// Processes a SPIR-V OpPhi instruction with the given `operands`.
  LogicalResult processPhi(ArrayRef<uint32_t> operands);

  /// Creates block arguments on predecessors previously recorded when handling
  /// OpPhi instructions.
  LogicalResult wireUpBlockArgument();

  /// Extracts blocks belonging to a structured selection/loop into a
  /// spirv.mlir.selection/spirv.mlir.loop op. This method iterates until all
  /// blocks declared as selection/loop headers are handled.
  LogicalResult structurizeControlFlow();

  //===--------------------------------------------------------------------===//
  // Instruction
  //===--------------------------------------------------------------------===//

  /// Get the Value associated with a result <id>.
  ///
  /// This method materializes normal constants and inserts "casting" ops
  /// (`spirv.mlir.addressof` and `spirv.mlir.referenceof`) to turn an symbol
  /// into a SSA value for handling uses of module scope constants/variables in
  /// functions.
  Value getValue(uint32_t id);

  /// Slices the first instruction out of `binary` and returns its opcode and
  /// operands via `opcode` and `operands` respectively. Returns failure if
  /// there is no more remaining instructions (`expectedOpcode` will be used to
  /// compose the error message) or the next instruction is malformed.
  LogicalResult
  sliceInstruction(spirv::Opcode &opcode, ArrayRef<uint32_t> &operands,
                   std::optional<spirv::Opcode> expectedOpcode = std::nullopt);

  /// Processes a SPIR-V instruction with the given `opcode` and `operands`.
  /// This method is the main entrance for handling SPIR-V instruction; it
  /// checks the instruction opcode and dispatches to the corresponding handler.
  /// Processing of Some instructions (like OpEntryPoint and OpExecutionMode)
  /// might need to be deferred, since they contain forward references to <id>s
  /// in the deserialized binary, but module in SPIR-V dialect expects these to
  /// be ssa-uses.
  LogicalResult processInstruction(spirv::Opcode opcode,
                                   ArrayRef<uint32_t> operands,
                                   bool deferInstructions = true);

  /// Processes a SPIR-V instruction from the given `operands`. It should
  /// deserialize into an op with the given `opName` and `numOperands`.
  /// This method is a generic one for dispatching any SPIR-V ops without
  /// variadic operands and attributes in TableGen definitions.
  LogicalResult processOpWithoutGrammarAttr(ArrayRef<uint32_t> words,
                                            StringRef opName, bool hasResult,
                                            unsigned numOperands);

  /// Processes a OpUndef instruction. Adds a spirv.Undef operation at the
  /// current insertion point.
  LogicalResult processUndef(ArrayRef<uint32_t> operands);

  /// Method to dispatch to the specialized deserialization function for an
  /// operation in SPIR-V dialect that is a mirror of an instruction in the
  /// SPIR-V spec. This is auto-generated from ODS. Dispatch is handled for
  /// all operations in SPIR-V dialect that have hasOpcode == 1.
  LogicalResult dispatchToAutogenDeserialization(spirv::Opcode opcode,
                                                 ArrayRef<uint32_t> words);

  /// Processes a SPIR-V OpExtInst with given `operands`. This slices the
  /// entries of `operands` that specify the extended instruction set <id> and
  /// the instruction opcode. The op deserializer is then invoked using the
  /// other entries.
  LogicalResult processExtInst(ArrayRef<uint32_t> operands);

  /// Dispatches the deserialization of extended instruction set operation based
  /// on the extended instruction set name, and instruction opcode. This is
  /// autogenerated from ODS.
  LogicalResult
  dispatchToExtensionSetAutogenDeserialization(StringRef extensionSetName,
                                               uint32_t instructionID,
                                               ArrayRef<uint32_t> words);

  /// Method to deserialize an operation in the SPIR-V dialect that is a mirror
  /// of an instruction in the SPIR-V spec. This is auto generated if hasOpcode
  /// == 1 and autogenSerialization == 1 in ODS.
  template <typename OpTy>
  LogicalResult processOp(ArrayRef<uint32_t> words) {
    return emitError(unknownLoc, "unsupported deserialization for ")
           << OpTy::getOperationName() << " op";
  }

private:
  /// The SPIR-V binary module.
  ArrayRef<uint32_t> binary;

  /// Contains the data of the OpLine instruction which precedes the current
  /// processing instruction.
  std::optional<DebugLine> debugLine;

  /// The current word offset into the binary module.
  unsigned curOffset = 0;

  /// MLIRContext to create SPIR-V ModuleOp into.
  MLIRContext *context;

  // TODO: create Location subclass for binary blob
  Location unknownLoc;

  /// The SPIR-V ModuleOp.
  OwningOpRef<spirv::ModuleOp> module;

  /// The current function under construction.
  std::optional<spirv::FuncOp> curFunction;

  /// The current block under construction.
  Block *curBlock = nullptr;

  OpBuilder opBuilder;

  spirv::Version version = spirv::Version::V_1_0;

  /// The list of capabilities used by the module.
  llvm::SmallSetVector<spirv::Capability, 4> capabilities;

  /// The list of extensions used by the module.
  llvm::SmallSetVector<spirv::Extension, 2> extensions;

  // Result <id> to type mapping.
  DenseMap<uint32_t, Type> typeMap;

  // Result <id> to constant attribute and type mapping.
  ///
  /// In the SPIR-V binary format, all constants are placed in the module and
  /// shared by instructions at module level and in subsequent functions. But in
  /// the SPIR-V dialect, we materialize the constant to where it's used in the
  /// function. So when seeing a constant instruction in the binary format, we
  /// don't immediately emit a constant op into the module, we keep its value
  /// (and type) here. Later when it's used, we materialize the constant.
  DenseMap<uint32_t, std::pair<Attribute, Type>> constantMap;

  // Result <id> to spec constant mapping.
  DenseMap<uint32_t, spirv::SpecConstantOp> specConstMap;

  // Result <id> to composite spec constant mapping.
  DenseMap<uint32_t, spirv::SpecConstantCompositeOp> specConstCompositeMap;

  /// Result <id> to info needed to materialize an OpSpecConstantOp
  /// mapping.
  DenseMap<uint32_t, SpecConstOperationMaterializationInfo>
      specConstOperationMap;

  // Result <id> to variable mapping.
  DenseMap<uint32_t, spirv::GlobalVariableOp> globalVariableMap;

  // Result <id> to function mapping.
  DenseMap<uint32_t, spirv::FuncOp> funcMap;

  // Result <id> to block mapping.
  DenseMap<uint32_t, Block *> blockMap;

  // Header block to its merge (and continue) target mapping.
  BlockMergeInfoMap blockMergeInfo;

  // For each pair of {predecessor, target} blocks, maps the pair of blocks to
  // the list of phi arguments passed from predecessor to target.
  DenseMap<std::pair<Block * /*predecessor*/, Block * /*target*/>, BlockPhiInfo>
      blockPhiInfo;

  // Result <id> to value mapping.
  DenseMap<uint32_t, Value> valueMap;

  // Mapping from result <id> to undef value of a type.
  DenseMap<uint32_t, Type> undefMap;

  // Result <id> to name mapping.
  DenseMap<uint32_t, StringRef> nameMap;

  // Result <id> to debug info mapping.
  DenseMap<uint32_t, StringRef> debugInfoMap;

  // Result <id> to decorations mapping.
  DenseMap<uint32_t, NamedAttrList> decorations;

  // Result <id> to type decorations.
  DenseMap<uint32_t, uint32_t> typeDecorations;

  // Result <id> to member decorations.
  // decorated-struct-type-<id> ->
  //    (struct-member-index -> (decoration -> decoration-operands))
  DenseMap<uint32_t,
           DenseMap<uint32_t, DenseMap<spirv::Decoration, ArrayRef<uint32_t>>>>
      memberDecorationMap;

  // Result <id> to member name.
  // struct-type-<id> -> (struct-member-index -> name)
  DenseMap<uint32_t, DenseMap<uint32_t, StringRef>> memberNameMap;

  // Result <id> to extended instruction set name.
  DenseMap<uint32_t, StringRef> extendedInstSets;

  // List of instructions that are processed in a deferred fashion (after an
  // initial processing of the entire binary). Some operations like
  // OpEntryPoint, and OpExecutionMode use forward references to function
  // <id>s. In SPIR-V dialect the corresponding operations (spirv.EntryPoint and
  // spirv.ExecutionMode) need these references resolved. So these instructions
  // are deserialized and stored for processing once the entire binary is
  // processed.
  SmallVector<std::pair<spirv::Opcode, ArrayRef<uint32_t>>, 4>
      deferredInstructions;

  /// A list of IDs for all types forward-declared through OpTypeForwardPointer
  /// instructions.
  SetVector<uint32_t> typeForwardPointerIDs;

  /// A list of all structs which have unresolved member types.
  SmallVector<DeferredStructTypeInfo, 0> deferredStructTypesInfos;

#ifndef NDEBUG
  /// A logger used to emit information during the deserialzation process.
  llvm::ScopedPrinter logger;
#endif
};

} // namespace spirv
} // namespace mlir

#endif // MLIR_TARGET_SPIRV_DESERIALIZER_H
