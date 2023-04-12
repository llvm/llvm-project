//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/KokkosCpp/KokkosCppEmitter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <utility>

#include <iostream>
//#include <unistd.h>

#define DEBUG_TYPE "translate-to-kokkos-cpp"

using namespace mlir;
using namespace mlir::emitc;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace {
struct KokkosCppEmitter {
  explicit KokkosCppEmitter(raw_ostream &os, bool declareVariablesAtTop);
  explicit KokkosCppEmitter(raw_ostream &os, raw_ostream& py_os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits the functions kokkos_mlir_initialize() and kokkos_mlir_finalize()
  /// These are responsible for init/finalize of Kokkos, and allocation/initialization/deallocation
  /// of global Kokkos::Views.
  LogicalResult emitInitAndFinalize();

  LogicalResult emitPythonBoilerplate();

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(Value result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(Value result,
                                        bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the name of a previously declared Value, or a literal constant.
  LogicalResult emitValue(Value val);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(KokkosCppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    KokkosCppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the C++ output stream.
  raw_indented_ostream &ostream() { return os; };

  bool emittingPython() {return py_os.get() != nullptr;}

  /// Returns the Python output stream.
  raw_indented_ostream &py_ostream() { return *py_os.get(); };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

  //This lets the emitter act like a stream (writes to the C++ file)
  template<typename T>
  raw_indented_ostream& operator<<(const T& t)
  {
    os << t;
    return os;
  }

  //Tell the emitter that it is now inside device code.
  LogicalResult enterDeviceCode()
  {
    if(insideDeviceCode)
    {
      //Shouldn't ever call this if already inside device code!
      return failure();
    }
    insideDeviceCode = true;
    return success();
  }

  LogicalResult exitDeviceCode()
  {
    if(!insideDeviceCode)
    {
      //Shouldn't ever call this if already outside device code!
      return failure();
    }
    insideDeviceCode = false;
    return success();
  }

  //Is this Value a strided memref generated by memref.subview?
  bool isStridedSubview(Value v) const
  {
    return stridedSubviews.find(v) != stridedSubviews.end();
  }

  // (Precondition: isStridedSubview(v))
  // Get the SubViewOp that generated the Value v (a memref).
  const memref::SubViewOp& getStridedSubview(Value v)
  {
    return stridedSubviews[v];
  }

  // Record a memref.subview operation that generated a strided subview (in any scope).
  void registerStridedSubview(Value result, memref::SubViewOp op)
  {
    stridedSubviews[result] = op;
  }

  void registerGlobalView(memref::GlobalOp op)
  {
    globalViews.push_back(op);
  }

  //Is v a scalar constant?
  bool isScalarConstant(Value v) const
  {
    return scalarConstants.find(v) != scalarConstants.end();
  }

  //val is how other ops reference the value.
  //attr actually describes the type and data of the literal.
  //The original arith::ConstantOp is not needed.
  void registerScalarConstant(Value val, arith::ConstantOp op)
  {
    scalarConstants[val] = op;
  }

  arith::ConstantOp getScalarConstantOp(Value v) const
  {
    return scalarConstants[v];
  }

  //Get the total number of elements (aka span, since it's contiguous) of a
  //statically sized MemRefType.
  static int64_t getMemrefSpan(MemRefType memrefType)
  {
    int64_t span = 1;
    for(auto extent : memrefType.getShape())
    {
      span *= extent;
    }
    return span;
  }

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// C++ output stream to emit to.
  raw_indented_ostream os;

  /// Python output stream to emit to.
  std::shared_ptr<raw_indented_ostream> py_os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
  
  /// Whether we are currently emitting the body of a device function 
  bool insideDeviceCode = false;

  //Bookkeeping for a memref.SubViewOp (strided subview).
  //This is more general than Kokkos::subview - each dimension can have an arbitrary stride,
  //equivalent to multiplying the index by some value. Kokkos::subview can only take contiguous slices in each dimension.
  //
  //This just maps the SubViewOp result (the variable name) back to the SubViewOp,
  //so that the index multiplications can be generated when the subview is accessed by memref.load/memref.store.
  llvm::DenseMap<Value, memref::SubViewOp> stridedSubviews;

  //Bookeeping for scalar constants (individual integer and floating-point values)
  mutable llvm::DenseMap<Value, arith::ConstantOp> scalarConstants;

  //Bookeeping for Kokkos::Views in global scope.
  //Each element has a name, element type, total size and whether it is intialized.
  //If initialized, ${name}_initial is assumed to be a global 1D host array with the data.
  std::vector<memref::GlobalOp> globalViews;
};
} // namespace

static LogicalResult printConstantOp(KokkosCppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Skip the assignment if the emitc.constant has no value.
    if (auto oAttr = value.dyn_cast<emitc::OpaqueAttr>()) {
      if (oAttr.getValue().empty())
        return success();
    }

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration for an emitc.constant op without value.
  if (auto oAttr = value.dyn_cast<emitc::OpaqueAttr>()) {
    if (oAttr.getValue().empty())
      // The semicolon gets printed by the emitOperation function.
      return emitter.emitVariableDeclaration(result,
                                             /*trailingSemicolon=*/false);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::GlobalOp op) {
  auto maybeValue = op.getInitialValue();
  //Emit the View declaration first. 
  //NOTE: using the GlobalOp's symbol name instead of a name generated for the current scope,
  //because GlobalOp does not produce a Result.
  if(failed(emitter.emitType(op.getLoc(), op.getType())))
    return failure();
  emitter << ' ' << op.getSymName() << ";\n";
  //Note: module-wide initialization will be responsible for allocating and copying the initializing data (if any).
  //Then module-wide finalization will deallocate (to avoid Kokkos warning about dealloc after finalize).
  if(maybeValue)
  {
    auto memrefType = op.getType();
    //For constants (initialized views), keep the actual data in a 1D array (with a related name).
    if(failed(emitter.emitType(op.getLoc(), memrefType.getElementType())))
      return failure();
    int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
    emitter << ' ' << op.getSymName() << "_initial" << "[" << span << "] = ";
    //Emit the 1D array literal
    if (failed(emitter.emitAttribute(op.getLoc(), maybeValue.value())))
      return failure();
    emitter << ";\n";
  }
  //Register this in list of global views
  emitter.registerGlobalView(op);
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::GetGlobalOp op) {
  Operation *operation = op.getOperation();
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    //Emit just the name of the global symbol (shallow copy)
    emitter << op.getName();
    return success();
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  //Emit just the name of the global symbol (shallow copy)
  emitter << op.getName();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::AllocOp op) {
  Operation *operation = op.getOperation();
  OpResult result = operation->getResult(0);
  MemRefType type = op.getType();

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    //Emit a Kokkos::View constructor call. Use the variable name as label.
    if (failed(emitter.emitType(op.getLoc(), type)))
      return failure();
    emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
    return success();
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (failed(emitter.emitType(op.getLoc(), type)))
    return failure();
  emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::AllocaOp op) {
  Operation *operation = op.getOperation();
  OpResult result = operation->getResult(0);
  MemRefType type = op.getType();

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    //Emit a Kokkos::View constructor call. Use the variable name as label.
    if (failed(emitter.emitType(op.getLoc(), type)))
      return failure();
    emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
    return success();
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (failed(emitter.emitType(op.getLoc(), type)))
    return failure();
  emitter << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << emitter.getOrCreateName(result) << "\"))";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::StoreOp op) {
  //TODO: if in host code, use a mirror view?
  emitter << emitter.getOrCreateName(op.getMemref()) << "(";
  for(auto iter = op.getIndices().begin(); iter != op.getIndices().end(); iter++)
  {
    if(iter != op.getIndices().begin())
      emitter << ", ";
    if(failed(emitter.emitValue(*iter)))
      return failure();
  }
  emitter << ") = ";
  if(failed(emitter.emitValue(op.getValue())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::LoadOp op) {
  //TODO: if in host code, use a mirror view?
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return op.emitError("Failed to emit LoadOp result type");
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getMemRef())))
    return op.emitError("Failed to emit the LoadOp's memref value");
  emitter << "(";
  for(auto iter = op.getIndices().begin(); iter != op.getIndices().end(); iter++)
  {
    if(iter != op.getIndices().begin())
      emitter << ", ";
    if(failed(emitter.emitValue(*iter)))
      return op.emitError("Failed to emit a LoadOp index");
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CopyOp op) {
  //TODO: if in device code, use a sequential loop or error out
  //(not clear if this is possible in IR from linalg)
  //
  //TODO: if source and/or target are strided subviews, must write a parallel loop and generate the strided accesses.
  //If neither are strided subviews, then Kokkos::deep_copy will be valid (may change layout, but will be within same memspace).
  if(emitter.isStridedSubview(op.getTarget()) || emitter.isStridedSubview(op.getSource()))
  {
    return op.emitError("ERROR: strided subviews not supported yet in memref.copy.");
  }
  emitter << "Kokkos::deep_copy(exec_space(), ";
  if(failed(emitter.emitValue(op.getTarget())))
    return failure();
  emitter << ", ";
  if(failed(emitter.emitValue(op.getSource())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::SubViewOp op) {
  Value result = op.getResult();
  if(emitter.isStridedSubview(op.getSource()))
  {
    puts("NOT SUPPORTED YET: strided subview of strided subview. Would need to figure out how to get extents correct");
    return failure();
  }
  emitter << "auto " << emitter.getOrCreateName(result) << " = Kokkos::subview(";
  if(failed(emitter.emitValue(op.getSource())))
    return failure();
  if(op.getOffsets().size())
  {
    //Subview has dynamic sizes/offsets/strides.
    //NOTE: if the offsets (non-static) are populated, we assume that the sizes and strides are also non-static.
    if(op.getSizes().size() != op.getOffsets().size())
    {
      puts("ERROR: sizes of SubViewOp don't have same size as offsets");
      return failure();
    }
    if(op.getStrides().size() != op.getOffsets().size())
    {
      puts("ERROR: strides of SubViewOp don't have same size as offsets");
      return failure();
    }
    //The subview in each dimension starts at the offset and goes to offset + size * stride.
    for(auto dim : llvm::zip(op.getOffsets(), op.getSizes(), op.getStrides()))
    {
      emitter << ", Kokkos::make_pair<int64_t, int64_t>(";
      if(failed(emitter.emitValue(std::get<0>(dim))))
        return failure();
      emitter << ", ";
      if(failed(emitter.emitValue(std::get<0>(dim))))
        return failure();
      emitter << " + ";
      if(failed(emitter.emitValue(std::get<1>(dim))))
        return failure();
      emitter << " * ";
      if(failed(emitter.emitValue(std::get<2>(dim))))
        return failure();
      emitter << ")";
    }
    emitter << ")";
    emitter.registerStridedSubview(result, op);
  }
  else if(op.getStaticOffsets().size())
  {
    //Subview has static sizes/offsets/strides.
    //NOTE: if the static offsets are populated, we assume that the sizes and strides are also static.
    if(op.getStaticSizes().size() != op.getStaticOffsets().size())
    {
      puts("ERROR: static_sizes of SubViewOp don't have same size as static_offsets");
      return failure();
    }
    if(op.getStaticStrides().size() != op.getStaticOffsets().size())
    {
      puts("ERROR: static_strides of SubViewOp don't have same size as static_offsets");
      return failure();
    }
    //If all strides are 1, this doesn't need to be registered as a strided subview. Kokkos::subview is enough.
    //Either way, the subview in each dimension starts at the offset and goes to offset + size * stride.
    for(auto dim : llvm::zip(op.getStaticOffsets(), op.getStaticSizes(), op.getStaticStrides()))
    {
      emitter << ", Kokkos::make_pair<int64_t, int64_t>(";
      emitter << std::get<0>(dim) << ", ";
      emitter << std::get<0>(dim) << " + ";
      emitter << std::get<1>(dim) << " * ";
      emitter << std::get<2>(dim);
      emitter << ")";
    }
    emitter << ")";
    bool isStrided = false;
    for(auto stride : op.getStaticStrides())
    {
      if(stride != 1)
        isStrided = true;
    }
    if(isStrided)
      emitter.registerStridedSubview(result, op);
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CollapseShapeOp op) {
  //CollapseShape flattens a subset of dimensions. The op semantics require that this can alias the existing data without copying/shuffling.
  //TODO: expressing this with unmanaged view, so need to manage the lifetime of the owning view so it doesn't end up with a dangling pointer.
  //MemRefType srcType = op.src().getType().cast<MemRefType>();
  MemRefType dstType = op.getResult().getType().cast<MemRefType>();
  if(failed(emitter.emitType(op.getLoc(), dstType)))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult());
  //TODO: handle dynamic dimensions here
  emitter << '(';
  if(failed(emitter.emitValue(op.getSrc())))
    return failure();
  emitter << ".data())";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    memref::CastOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getDest().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getDest()) << "(";
  if(failed(emitter.emitValue(op.getSource())))
    return failure();
  emitter << ".data()";
  if(auto dstType = op.getDest().getType().dyn_cast<MemRefType>())
  {
    //Static-extent Kokkos views need no other ctor arguments than the pointer.
    //Dynamic-extent need each extent.
    if(!dstType.hasStaticShape()) {
      auto srcType = op.getSource().getType().dyn_cast<MemRefType>();
      if(!srcType.hasStaticShape())
        return op.emitError("memref.cast: cast from one dynamic-shape memref to another not supported");
      for(auto extent : srcType.getShape()) {
        emitter << ", " << extent;
      }
    }
  }
  else if(auto dstType = op.getDest().getType().dyn_cast<UnrankedMemRefType>())
  {
    //Dst is unranked, so it is represented as a rank-1 runtime sized View.
    MemRefType srcType = op.getSource().getType().dyn_cast<MemRefType>();
    if(!srcType)
      return op.emitError("memref.cast: if result is an unranked memref, we assume that src is a (ranked) memref, but it isn't.");
    int64_t span = KokkosCppEmitter::getMemrefSpan(srcType);
    emitter << ", " << span;
  }
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  //Register the constant with the emitter so that it can replace usage of this variable with 
  //an equivalent literal. Don't need to declare the actual SSA variable.
  emitter.registerScalarConstant(operation->getResult(0), constantOp);
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::FPToUIOp op) {
  //In C, float->unsigned conversion when input is negative is implementation defined, but MLIR says it should convert to the nearest value (0)
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getOut()) << " = (";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  emitter << " <= 0.f) ?";
  emitter << "0U : (";
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ") ";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    func::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();
  return printConstantOp(emitter, operation, value);
}

/*
static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    cf::BranchOp branchOp) {
  raw_ostream &os = emitter.ostream();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  os.indent();

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os.unindent() << "} else {\n";
  os.indent();
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os.unindent() << "}";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    cf::AssertOp op) {
  emitter << "if(!" << emitter.getOrCreateName(op.getArg()) << ") Kokkos::abort(";
  if(failed(emitter.emitAttribute(op.getLoc(), op.getMsgAttr())))
    return failure();
  emitter << ")";
  return success();
}
*/

static LogicalResult printOperation(KokkosCppEmitter &emitter, func::CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getIterOperands();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if(failed(emitter.emitValue(std::get<1>(pair))))
      return failure();
    os << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if(failed(emitter.emitValue(forOp.getLowerBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  if(failed(emitter.emitValue(forOp.getUpperBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if(failed(emitter.emitValue(forOp.getStep())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = ";
    if(failed(emitter.emitValue(operand)))
      return failure();
    emitter << ";\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = ";
    if(failed(emitter.emitValue(iterArg)))
      return failure();
    emitter << ";";
  }

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::WhileOp whileOp) {
  //Declare the before args, after args, and results.
  for (auto pair : llvm::zip(whileOp.getBeforeArguments(), whileOp.getInits())) {
  //for (OpResult beforeArg : whileOp.getBeforeArguments()) {
    // Before args are initialized to the whileOp's "inits"
    if(failed(emitter.emitType(whileOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    emitter << ' ' << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    if(failed(emitter.emitValue(std::get<1>(pair))))
      return failure();
  }
  for (auto afterArg : whileOp.getAfterArguments()) {
    if (failed(emitter.emitVariableDeclaration(afterArg, /*trailingSemicolon=*/true)))
      return failure();
  }
  for (OpResult result : whileOp.getResults()) {
    if (failed(emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
      return failure();
  }

  emitter << "/*\n";
  emitter << "Hello from while op.\n";
  emitter << "Inits:\n";
  for(auto a : whileOp.getInits())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "Before block args:\n";
  for(auto a : whileOp.getBeforeArguments())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "After block args:\n";
  for(auto a : whileOp.getAfterArguments())
    emitter << "  " << emitter.getOrCreateName(a) << "\n";
  emitter << "*/\n";

  emitter << "while(true) {\n";
  emitter.ostream().indent();

  //Emit the "before" block(s)
  for (auto& beforeOp : whileOp.getBefore().getOps()) {
    if (failed(emitter.emitOperation(beforeOp, /*trailingSemicolon=*/true)))
      return failure();
  }

  //Emit the "after" block(s)
  for (auto& afterOp : whileOp.getAfter().getOps()) {
    if (failed(emitter.emitOperation(afterOp, /*trailingSemicolon=*/true)))
      return failure();
  }
  emitter.ostream().unindent();
  emitter << "}\n";

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ConditionOp condOp) {
  //The condition value should already be in scope. Just break out of loop if it's falsey.
  emitter << "if(";
  if(failed(emitter.emitValue(condOp.getCondition())))
    return failure();
  emitter << ") {\n";
  emitter << "}\n";
  emitter << "else {\n";
  //Condition false: breaking out of loop
  emitter << "break\n";
  emitter << "}\n";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::ParallelOp op) {
  OperandRange lowerBounds = op.getLowerBound();
  OperandRange upperBounds = op.getUpperBound();
  OperandRange step = op.getStep();
  //OperandRange initVals = op.getInitVals();
  ValueRange inductionVars = op.getInductionVars();
  //Note: results mean there is a reduction
  ResultRange results = op.getResults();
  bool isReduction = results.size() > size_t(0);

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result, true)))
        return failure();
    }
  }

  //TODO: handle common simplifying cases:
  //  - if step for a dimension is the constant 1, don't need to shift/scale the induction variable.
  //  - if iter range for a dimension is a single value, remove it and simply declare
  //    that induction variable as a constant (lowerBound).
  int rank = inductionVars.size();
  emitter << "Kokkos::parallel_";
  if(isReduction)
    emitter << "reduce";
  else
    emitter << "for";
  if(rank == 0)
    return op.emitError("Rank-0 (single element) parallel iteration space not supported");
  //Kokkos policies don't support step size other than 1.
  //To express an iteration space with arbitrary (step, lower, upper) correctly:
  //  iterate from [0, (upper - lower + step - 1) / step)
  //  and then take lower + i * step to get the value of the induction variable's value
  if(rank == 1)
  {
    emitter << "(Kokkos::RangePolicy<exec_space>(0, (";
    if(failed(emitter.emitValue(upperBounds[0])))
      return failure();
    emitter << " - ";
    if(failed(emitter.emitValue(lowerBounds[0])))
      return failure();
    emitter << " + ";
    if(failed(emitter.emitValue(step[0])))
      return failure();
    emitter << " - 1) / ";
    if(failed(emitter.emitValue(step[0])))
      return failure();
    emitter << "),\n";
  }
  else
  {
    emitter << "(Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<" << rank << ">>(";
    emitter << "{";
    for(int i = 0; i < rank; i++)
    {
      if(i != 0)
        emitter << ", ";
      emitter << "0";
    }
    emitter << "}, {";
    for(int i = 0; i < rank; i++)
    {
      if(i != 0)
        emitter << ", ";
      emitter << '(';
      if(failed(emitter.emitValue(upperBounds[i])))
        return failure();
      emitter << " - ";
      if(failed(emitter.emitValue(lowerBounds[i])))
        return failure();
      emitter << " + ";
      if(failed(emitter.emitValue(step[i])))
        return failure();
      emitter << " - 1) / ";
      if(failed(emitter.emitValue(step[i])))
        return failure();
    }
    emitter << "}),\n";
  }
  emitter << "KOKKOS_LAMBDA(";
  for(int i = 0; i < rank; i++)
  {
    if(i > 0)
      emitter << ", ";
    //note: the MDRangePolicy only iterates with unit step in each dimension. This variable needs to be
    //shifted and scaled to match the actual range.
    emitter << "int64_t unit_" << emitter.getOrCreateName(inductionVars[i]);
  }
  //TODO: declare local update vars for each reduction here
  emitter << ")\n{\n";
  emitter.ostream().indent();
  // Declare the actual induction variables, with the correct bounds and step
  for(int i = 0; i < rank; i++)
  {
    emitter << "int64_t " << emitter.getOrCreateName(inductionVars[i]) << " = ";
    if(failed(emitter.emitValue(lowerBounds[i])))
      return failure();
    emitter << " + unit_" << emitter.getOrCreateName(inductionVars[i]) << " * ";
    if(failed(emitter.emitValue(step[i])))
      return failure();
    emitter << ";\n";
  }
  //Now add the parallel body
  Region& body = op.getRegion();
  for (auto& op : body.getOps())
  {
    if (failed(emitter.emitOperation(op, true)))
      return failure();
  }
  emitter.ostream().unindent();
  //TODO: add Kokkos reducers here. Then join with the 'init' values,
  // since Kokkos always assumes the init is identity.
  emitter << "})\n";
  return success();
  /*

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  os << emitter.getOrCreateName(forOp.getLowerBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  os << emitter.getOrCreateName(forOp.getUpperBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  os << emitter.getOrCreateName(forOp.getStep());
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";";
  }

  return success();
  */
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, ModuleOp moduleOp) {
  KokkosCppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, func::FuncOp functionOp) {
  llvm::StringRef functionOpName = functionOp.getName();
  if (functionOpName == "main") {
    functionOpName = "mymain";
  }
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }
  raw_indented_ostream &os = emitter.ostream();
  // Handle function declarations (empty body). Don't need to give parameters names either.
  if(functionOp.getBody().empty())
  {
    if (failed(emitter.emitTypes(functionOp.getLoc(), functionOp.getFunctionType().getResults())))
      return failure();
    os << ' ' << functionOpName << '(';
    if (failed(interleaveCommaWithError(functionOp.getArgumentTypes(), os,
      [&](Type argType) -> LogicalResult
      {
        if (failed(emitter.emitType(functionOp.getLoc(), argType)))
          return failure();
        return success();
      })))
      return failure();
    os << ");\n";
    return success();
  }
  // Otherwise, it's a function definition with body.
  KokkosCppEmitter::Scope scope(emitter);
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getResultTypes())))
    return failure();
  os << ' ' << functionOpName;
  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";
  os.indent();
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    Block &block = *it;
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or cf.cond_br op no semicolon needs
      // to be printed after the closing brace.
      // When generating code for an scf.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<scf::IfOp, scf::ForOp, cf::CondBranchOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n\n";
  // If just emitting C++, this is all we have to do
  if(!emitter.emittingPython())
    return success();
  // Emit a wrapper function without Kokkos::View for Python to call
  os << "extern \"C\" void " << "py_" << functionOpName << '(';
  // Put the results first: primitives and memrefs are both passed by pointer.
  // Python interface will enforce LayoutRight on all numpy arrays.
  //
  FunctionType ftype = functionOp.getFunctionType();
  size_t numResults = ftype.getNumResults();
  size_t numParams = ftype.getNumInputs();
  for(size_t i = 0; i < numResults; i++)
  {
    if(i != 0)
      os << ", ";
    auto retType = ftype.getResult(i);
    if(auto memrefType = retType.dyn_cast<MemRefType>())
    {
      //This is represented using a pointer to the element type
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "* ret" << i;
    }
    else
    {
      //Assuming it is a scalar primitive
      if(failed(emitter.emitType(functionOp.getLoc(), retType)))
        return failure();
      os << "* ret" << i;
    }
  }
  // Now emit the parameters - primitives passed by value
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0 || numResults)
      os << ", ";
    auto paramType = ftype.getInput(i);
    if(auto memrefType = paramType.dyn_cast<MemRefType>())
    {
      //This is represented using a pointer to the element type
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "* param" << i;
    }
    else
    {
      //Assuming it is a scalar primitive
      if(failed(emitter.emitType(functionOp.getLoc(), paramType)))
        return failure();
      os << " param" << i;
    }
  }
  os << ")\n";
  os << "{\n";
  os.indent();
  //Create/allocate device Kokkos::Views for the memref inputs.
  //TODO: if executing on on host, we might as well use the NumPy buffers directly.
  for(size_t i = 0; i < numParams; i++)
  {
    auto paramType = ftype.getInput(i);
    if(auto memrefType = paramType.dyn_cast<MemRefType>())
    {
      int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
      os << "Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*> param" << i << "_buf(\"param buffer " << i << "\", ";
      os << span << ");\n";
      //Copy from the NumPy buffer
      os << "Kokkos::deep_copy(exec_space(), param" << i << "_buf, Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*, Kokkos::HostSpace>(param" << i << ", " << span << "));\n";
    }
  }
  os << "auto results = " << functionOpName << "(";
  //Construct a Kokkos::View for each memref input, from raw pointer.
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      os << ", ";
    auto paramType = ftype.getInput(i);
    if(auto memrefType = paramType.dyn_cast<MemRefType>())
    {
      //TODO: handle dynamic sized and unranked memrefs here
      if(failed(emitter.emitType(functionOp.getLoc(), paramType)))
        return failure();
      os << "(param" << i << "_buf.data())";
    }
    else
    {
      os << "param" << i;
    }
  }
  os << ");\n";
  //Now, unpack the results (if any) to the return values.
  //If there are multiple results, 'results' will be a std::tuple.
  //Need to deep_copy memref returns back to the NumPy buffers.
  for(size_t i = 0; i < numResults; i++)
  {
    if(i != 0)
      os << ", ";
    auto retType = ftype.getResult(i);
    if(auto memrefType = retType.dyn_cast<MemRefType>())
    {
      int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
      os << "Kokkos::deep_copy(exec_space(), Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*, Kokkos::HostSpace>(ret" << i << ", " << span << "), Kokkos::View<";
      if(failed(emitter.emitType(functionOp.getLoc(), memrefType.getElementType())))
        return failure();
      os << "*>(";
      if(numResults == size_t(1))
        os << "results";
      else
        os << "std::get<" << i << ">(results)";
      os << ".data(), " << span << "));\n";
    }
    else
    {
      os << "*ret" << i << " = ";
      if(numResults == size_t(1))
        os << "results;\n";
      else
        os << "std::get<" << i << ">(results);\n";
    }
  }
  os.unindent();
  os << "}\n";
  // Now that the native function (name: "py_" + functionOpName)
  // exists, generate the Python function to call it.
  //
  // Get the NumPy type corresponding to MLIR primitive.
  auto getNumpyType = [](Type t) -> std::string
  {
    if(t.isIndex())
      return "numpy.uint64";
    //Note: treating MLIR "signless" integer types as equivalent to unsigned NumPy integers.
    if(t.isSignlessInteger(8) || t.isUnsignedInteger(8))
      return "numpy.uint8";
    if(t.isSignlessInteger(16) || t.isUnsignedInteger(16))
      return "numpy.uint16";
    if(t.isSignlessInteger(32) || t.isUnsignedInteger(32))
      return "numpy.uint32";
    if(t.isSignlessInteger(64) || t.isUnsignedInteger(64))
      return "numpy.uint64";
    if(t.isSignedInteger(8))
      return "numpy.int8";
    if(t.isSignedInteger(16))
      return "numpy.int16";
    if(t.isSignedInteger(32))
      return "numpy.int32";
    if(t.isSignedInteger(64))
      return "numpy.int64";
    if(t.isF16())
      return "numpy.float16";
    if(t.isF32())
      return "numpy.float32";
    if(t.isF64())
      return "numpy.float64";
    return "";
  };

  // Use this to enforce LayoutRight for input memrefs (zero cost if it already is):
  // arr = numpy.require(arr, dtype=TYPE, requirements=['C'])
  // NOTE: numpy.zeros(shape, dtype=...) already defaults to LayoutRight, so that is ok.
  //
  // TODO: need to pass and return runtime extents here.
  // For return values, need 3 phases:
  //  - compute results in Kokkos and hold them temporarily.
  //  - get the extents back to Python and allocate
  //  - copy out the results to Python and then free the Kokkos temporaries.
  auto& py_os = emitter.py_ostream();
  //NOTE: this function is a member of the module's class, but py_os is already indented to write methods.
  py_os << "def " << functionOpName << "(self, ";
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0)
      py_os << ", ";
    py_os << "param" << i;
  }
  py_os << "):\n";
  py_os.indent();
  // Enforce types on all inputs
  for(size_t i = 0; i < numParams; i++)
  {
    py_os << "param" << i << " = ";
    auto paramType = ftype.getInput(i);
    if(auto memrefType = paramType.dyn_cast<MemRefType>())
    {
      std::string numpyDType = getNumpyType(memrefType.getElementType());
      if(!numpyDType.size())
        return failure();
      py_os << "numpy.require(param" << i << ", dtype=" << numpyDType << ", requirements=['C'])\n";
    }
    else
    {
      //Wrap scalar primitives in 1D NumPy array.
      //This gives it the correct type, and lets us use the same ndarray CTypes API as memrefs.
      std::string numpyDType = getNumpyType(paramType);
      if(!numpyDType.size())
        return failure();
      py_os << "numpy.array(param" << i << ", dtype=" << numpyDType << ", ndmin=1)\n";
    }
  }
  // Construct outputs
  // Note: by default, numpy.zeros uses LayoutRight
  for(size_t i = 0; i < numResults; i++)
  {
    auto retType = ftype.getResult(i);
    if(auto memrefType = retType.dyn_cast<MemRefType>())
    {
      std::string numpyDType = getNumpyType(memrefType.getElementType());
      if(!numpyDType.size())
        return failure();
      py_os << "ret" << i << " = numpy.zeros((";
      //TODO: this assumes static dimensions
      for(size_t j = 0; j < memrefType.getShape().size(); j++)
      {
        if(j != 0)
          py_os << ", ";
        py_os << memrefType.getShape()[j];
      }
      py_os << "), dtype=" << numpyDType << ")\n";
    }
    else
    {
      //For scalars, construct a single-element numpy ndarray so that we can use its CTypes API
      std::string numpyDType = getNumpyType(retType);
      if(!numpyDType.size())
        return failure();
      py_os << "ret" << i << " = numpy.zeros(1, dtype=" << numpyDType << ")\n";
    }
  }
  // Generate the native call. It always returns void.
  py_os << "self.libHandle.py_" << functionOpName << "(";
  // Outputs go first
  for(size_t i = 0; i < numResults; i++)
  {
    if(i != 0)
      py_os << ", ";
    py_os << "ret" << i << ".ctypes.data_as(ctypes.c_void_p)";
  }
  for(size_t i = 0; i < numParams; i++)
  {
    if(i != 0 || numResults != size_t(0))
    {
      py_os << ", ";
    }
    py_os << "param" << i << ".ctypes.data_as(ctypes.c_void_p)";
  }
  py_os << ")\n";
  // Finally, generate the return. Note that in Python, a 1-elem tuple is equivalent to scalar.
  if(numResults)
  {
    py_os << "return (";
    for(size_t i = 0; i < numResults; i++)
    {
      if(i != 0)
        py_os << ", ";
      auto retType = ftype.getResult(i);
      if(auto memrefType = retType.dyn_cast<MemRefType>())
      {
        //Return the whole memref
        py_os << "ret" << i;
      }
      else
      {
        //Return just the single element
        py_os << "ret" << i << "[0]";
      }
    }
    py_os << ")\n";
  }
  py_os.unindent();
  return success();
}

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& os, bool declareVariablesAtTop)
    : os(os), py_os(nullptr), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

KokkosCppEmitter::KokkosCppEmitter(raw_ostream& os, raw_ostream& py_os_, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  this->py_os = std::make_shared<raw_indented_ostream>(py_os_); 
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef KokkosCppEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef KokkosCppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool KokkosCppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool KokkosCppEmitter::hasValueInScope(Value val) { return valueMapper.count(val) || isScalarConstant(val); }

bool KokkosCppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult KokkosCppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      os << strValue;
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "f";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        //no suffix for double literal
        break;
      default:
        puts("WARNING: literal printing only supports float and double now!");
        break;
      };
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = attr.dyn_cast<FloatAttr>()) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = attr.dyn_cast<DenseFPElementsAttr>()) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = attr.dyn_cast<IntegerAttr>()) {
    if (auto iType = iAttr.getType().dyn_cast<IntegerType>()) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = iAttr.getType().dyn_cast<IndexType>()) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = attr.dyn_cast<DenseIntElementsAttr>()) {
    if (auto iType = dense.getType()
                         .cast<TensorType>()
                         .getElementType()
                         .dyn_cast<IntegerType>()) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dense.getType()
                         .cast<TensorType>()
                         .getElementType()
                         .dyn_cast<IndexType>()) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = attr.dyn_cast<SymbolRefAttr>()) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print string attribute (including quotes). Using hex of each character so that special characters don't need escaping.
  if (auto strAttr = attr.dyn_cast<StringAttr>())
  {
    os << '"';
    auto val = strAttr.strref();
    for(char c : val)
    {
      char buf[4];
      sprintf(buf, "%02x", (unsigned) c);
      os << "\\x" << buf;
    }
    os << '"';
    return success();
  }

  // Print type attributes.
  if (auto type = attr.dyn_cast<TypeAttr>())
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute of unsupported type");
}

LogicalResult KokkosCppEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    if(failed(emitValue(result)))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
KokkosCppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult
KokkosCppEmitter::emitValue(Value val)
{
  if(isScalarConstant(val))
  {
    arith::ConstantOp op = getScalarConstantOp(val);
    Attribute value = op.getValue();
    return emitAttribute(op.getLoc(), value);
  }
  else
  {
    //If calling this, the value should have already been declared
    if (!valueMapper.count(val))
      return failure();
    os << *valueMapper.begin(val);
    return success();
  }
}

LogicalResult KokkosCppEmitter::emitVariableAssignment(Value result) {
  if (!hasValueInScope(result)) {
    auto op = result.getDefiningOp();
    if(op) {
      return op->emitOpError(
          "result variable for the operation has not been declared");
    }
    else {
      return failure();
    }
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult KokkosCppEmitter::emitVariableDeclaration(Value result,
                                                  bool trailingSemicolon) {
  auto op = result.getDefiningOp();
  if (hasValueInScope(result)) {
    if(op) {
      return op->emitError(
          "result variable for the operation already declared");
    }
    else {
      return failure();
    }
  }
  Location loc = op ? op->getLoc() : Location(LocationAttr());
  if (failed(emitType(loc, result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult KokkosCppEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult KokkosCppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::NegFOp op) {
  if (failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = -" << emitter.getOrCreateName(op.getOperand());
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::CmpFOp op) {
  //Note: see ArithmeticOpsEnums.h.inc for values of arith::CmpFPredicate
  //2 types of float comparisons in MLIR: ordered and unordered. Ordered is like C.
  //Unordered is true if the ordered version would be true, OR if neither a<=b nor a>=b is true.
  //The second case applies for example when a and/or b is NaN.
  emitter << "bool " << emitter.getOrCreateName(op.getResult()) << " = ";
  //Handle two easy cases first - always true or always false.
  if(op.getPredicate() == arith::CmpFPredicate::AlwaysFalse)
  {
    emitter << "false";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::AlwaysTrue)
  {
    emitter << "true";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::ORD)
  {
    emitter << "!(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << "))";
    return success();
  }
  if(op.getPredicate() == arith::CmpFPredicate::UNO)
  {
    emitter << "(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << "))";
    return success();
  }
  //CmpFOp predicate is an enum, 0..15 inclusive. 1..6 are ordered comparisons (== > >= < <= !=), and 8..13 are corresponding unordered comparisons.
  int rawPred = (int) op.getPredicate();
  bool isUnordered = rawPred >= 8 && rawPred < 15;
  //Now, can convert unordered predicates to equivalent ordered.
  if(isUnordered)
    rawPred -= 7;
  if(isUnordered)
  {
    emitter << "(Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getLhs())))
      return failure();
    emitter << ") || Kokkos::isnan(";
    if(failed(emitter.emitValue(op.getRhs())))
      return failure();
    emitter << ")) || ";
  }
  emitter << "(";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ';
  switch(rawPred)
  {
    case 1:
      emitter << "=="; break;
    case 2:
      emitter << ">"; break;
    case 3:
      emitter << ">="; break;
    case 4:
      emitter << "<"; break;
    case 5:
      emitter << "<="; break;
    case 6:
      emitter << "!="; break;
    default:
      return op.emitError("CmpFOp: should never get here");
  }
  emitter << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::CmpIOp op) {
  //Note: see ArithmeticOpsEnums.h.inc for values of arith::CmpIPredicate
  int rawPred = (int) op.getPredicate();
  bool needsCast = rawPred > 1;
  bool castToUnsigned = rawPred >= 6;
  emitter << "bool " << emitter.getOrCreateName(op.getResult()) << " = ";
  //Emit a value, but cast to signed/unsigned depending on needsCast and castToUnsigned.
  auto emitValueWithSignednessCast = [&](Value v)
  {
    if(needsCast)
    {
      if(castToUnsigned)
      {
        emitter << "static_cast<std::make_unsigned_t<";
        if(failed(emitter.emitType(op.getLoc(), v.getType())))
          return failure();
        emitter << ">>(";
      }
      else
      {
        emitter << "static_cast<std::make_signed_t<";
        if(failed(emitter.emitType(op.getLoc(), v.getType())))
          return failure();
        emitter << ">>(";
      }
    }
    emitter << emitter.getOrCreateName(v);
    if(needsCast)
      emitter << ')';
    return success();
  };
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << ' ';
  switch(op.getPredicate())
  {
    case arith::CmpIPredicate::eq:
      emitter << "=="; break;
    case arith::CmpIPredicate::ne:
      emitter << "!="; break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      emitter << "<"; break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      emitter << "<="; break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      emitter << ">"; break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      emitter << ">="; break;
    default:
      puts("Should never get here.");
      return failure();
  }
  emitter << ' ';
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    arith::SelectOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getCondition())))
    return failure();
  emitter << "? ";
  if(failed(emitter.emitValue(op.getTrueValue())))
    return failure();
  emitter << " : ";
  if(failed(emitter.emitValue(op.getFalseValue())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printFloatMinMax(KokkosCppEmitter &emitter, T op, const char* selectOperator) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << '(';
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ' << selectOperator << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  emitter << ')';
  emitter << " ? ";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << " : ";
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printIntMinMax(KokkosCppEmitter &emitter, T op, const char* selectOperator, bool castToUnsigned) {
  auto emitValueWithSignednessCast = [&](Value v)
  {
    if(castToUnsigned)
    {
      emitter << "static_cast<std::make_unsigned_t<";
      if(failed(emitter.emitType(op.getLoc(), v.getType())))
        return failure();
      emitter << ">>(";
    }
    else
    {
      emitter << "static_cast<std::make_signed_t<";
      if(failed(emitter.emitType(op.getLoc(), v.getType())))
        return failure();
      emitter << ">>(";
    }
    if(failed(emitter.emitValue(v)))
      return failure();
    emitter << ')';
    return success();
  };
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << '(';
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << ' ' << selectOperator << ' ';
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  emitter << ')';
  emitter << " ? ";
  if(failed(emitValueWithSignednessCast(op.getLhs())))
    return failure();
  emitter << " : ";
  if(failed(emitValueWithSignednessCast(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
struct ArithBinaryInfixOperator
{
  static std::string get() {return std::string("/* ERROR: binary infix operator for ") + T::getOperationName() + " is not registered */";}
};

template<>
struct ArithBinaryInfixOperator<arith::AddFOp>
{
  static std::string get() {return "+";}
};

template<>
struct ArithBinaryInfixOperator<arith::AddIOp>
{
  static std::string get() {return "+";}
};

template<>
struct ArithBinaryInfixOperator<arith::SubFOp>
{
  static std::string get() {return "-";}
};

template<>
struct ArithBinaryInfixOperator<arith::SubIOp>
{
  static std::string get() {return "-";}
};

template<>
struct ArithBinaryInfixOperator<arith::MulFOp>
{
  static std::string get() {return "*";}
};

template<>
struct ArithBinaryInfixOperator<arith::MulIOp>
{
  static std::string get() {return "*";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivFOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivSIOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::DivUIOp>
{
  static std::string get() {return "/";}
};

template<>
struct ArithBinaryInfixOperator<arith::AndIOp>
{
  static std::string get() {return "&";}
};

template<>
struct ArithBinaryInfixOperator<arith::OrIOp>
{
  static std::string get() {return "|";}
};

template<>
struct ArithBinaryInfixOperator<arith::XOrIOp>
{
  static std::string get() {return "^";}
};

template<typename T>
static LogicalResult printBinaryInfixOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitValue(op.getLhs())))
    return failure();
  emitter << ' ' << ArithBinaryInfixOperator<T>::get() << ' ';
  if(failed(emitter.emitValue(op.getRhs())))
    return failure();
  return success();
}

template<typename T>
static LogicalResult printImplicitConversionOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getOut().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getOut()) << " = ";
  if(failed(emitter.emitValue(op.getIn())))
    return failure();
  return success();
}

template<typename T>
struct MathFunction
{
  static std::string get() {return std::string("/* ERROR: math function for operator ") + T::getOperationName() + " is not registered */";}
};

template<>
struct MathFunction<math::SqrtOp>
{
  static std::string get() {return "Kokkos::sqrt";}
};

template<>
struct MathFunction<math::AbsIOp>
{
  static std::string get() {return "Kokkos::abs";}
};

template<>
struct MathFunction<math::AbsFOp>
{
  static std::string get() {return "Kokkos::abs";}
};

template<>
struct MathFunction<math::ExpOp>
{
  static std::string get() {return "Kokkos::exp";}
};

template<>
struct MathFunction<math::Exp2Op>
{
  static std::string get() {return "Kokkos::exp2";}
};

template<>
struct MathFunction<math::SinOp>
{
  static std::string get() {return "Kokkos::sin";}
};

template<>
struct MathFunction<math::CosOp>
{
  static std::string get() {return "Kokkos::cos";}
};

template<>
struct MathFunction<math::AtanOp>
{
  static std::string get() {return "Kokkos::atan";}
};

template<>
struct MathFunction<math::TanhOp>
{
  static std::string get() {return "Kokkos::tanh";}
};

template<>
struct MathFunction<math::ErfOp>
{
  static std::string get() {return "Kokkos::erf";}
};

template<>
struct MathFunction<math::LogOp>
{
  static std::string get() {return "Kokkos::log";}
};

template<>
struct MathFunction<math::Log2Op>
{
  static std::string get() {return "Kokkos::log2";}
};

template<typename T>
static LogicalResult printMathOperation(KokkosCppEmitter &emitter, T op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  emitter << MathFunction<T>::get() << "(";
  if(failed(emitter.emitValue(op.getOperand())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter, math::RsqrtOp op) {
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = ";
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << "(1.0f) / Kokkos::sqrt(";
  if(failed(emitter.emitValue(op.getOperand())))
    return failure();
  emitter << ")";
  return success();
}

static LogicalResult printOperation(KokkosCppEmitter &emitter,
                                    LLVM::NullOp op) {
  // Make sure the result type is a pointer (raw, not a memref).
  // Emit this is a raw null pointer, not an unallocated Kokkos::View
  if (!op.getResult().getType().isa<LLVM::LLVMPointerType>()) {
    return op.emitOpError("LLVM::NullOp has a non-pointer result type");
  }
  if(failed(emitter.emitType(op.getLoc(), op.getResult().getType())))
    return failure();
  emitter << ' ' << emitter.getOrCreateName(op.getResult()) << " = nullptr";
  return success();
}

LogicalResult KokkosCppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  if(auto constantOp = dyn_cast<arith::ConstantOp>(&op)) {
    //arith.constant is not directly emitted in the code, so always skip the
    // "// arith.constant" comment and trailing semicolon.
    return printOperation(*this, constantOp);
  }
  os << "// " << op.getName() << '\n';
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<func::FuncOp, ModuleOp>(
              [&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<cf::BranchOp, cf::CondBranchOp, cf::AssertOp>(
              [&](auto op) { /* Do nothing */ return success(); })
          // Func ops.
          .Case<func::CallOp, func::ConstantOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // SCF ops.
          .Case<scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp, scf::ConditionOp, scf::ParallelOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops: general
          .Case<arith::FPToUIOp, arith::NegFOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops: standard binary infix operators. All have the same syntax "result = lhs <operator> rhs;".
          // ArithBinaryInfixOperator<Op>::get() will provide the <operator>.
          .Case<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(
              [&](auto op) { return printBinaryInfixOperation(*this, op); })
          // Arithmetic ops: type casting that C++ compiler can handle automatically with implicit conversion: "result = operand;"
          .Case<arith::UIToFPOp, arith::FPToSIOp, arith::TruncFOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp>(
              [&](auto op) { return printImplicitConversionOperation(*this, op); })
          // Arithmetic ops: min/max expressed using ternary operator.
          .Case<arith::MinFOp>(
              [&](auto op) { return printFloatMinMax(*this, op, "<"); })
          .Case<arith::MaxFOp>(
              [&](auto op) { return printFloatMinMax(*this, op, ">"); })
          .Case<arith::MinSIOp>(
              [&](auto op) { return printIntMinMax(*this, op, "<", false); })
          .Case<arith::MaxSIOp>(
              [&](auto op) { return printIntMinMax(*this, op, ">", false); })
          .Case<arith::MinUIOp>(
              [&](auto op) { return printIntMinMax(*this, op, "<", true); })
          .Case<arith::MaxUIOp>(
              [&](auto op) { return printIntMinMax(*this, op, ">", true); })
          // Math ops: general
          .Case<math::RsqrtOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Math ops: unary functions supported by Kokkos
          .Case<math::SqrtOp, math::AbsIOp, math::AbsFOp, math::ExpOp, math::Exp2Op, math::SinOp, math::CosOp, math::AtanOp, math::TanhOp, math::ErfOp, math::LogOp, math::Log2Op>(
              [&](auto op) { return printMathOperation(*this, op); })
          // Memref ops.
          .Case<memref::GlobalOp, memref::GetGlobalOp, memref::AllocOp, memref::AllocaOp, memref::StoreOp, memref::LoadOp, memref::CopyOp, memref::SubViewOp, memref::CollapseShapeOp, memref::CastOp>(
              [&](auto op) { return printOperation(*this, op); })
          // LLVM ops.
          .Case<LLVM::NullOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Other operations are unknown/unsupported.
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();
  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult KokkosCppEmitter::emitInitAndFinalize()
{
  os << "extern \"C\" void kokkos_mlir_initialize()\n";
  os << "{\n";
  os.indent();
  os << "Kokkos::initialize();\n";
  //For each global view: allocate it, and if there is initializing data, copy it
  for(auto& op : globalViews)
  {
    os << "{\n";
    os.indent();
    os << op.getSymName() << " = ";
    if(failed(emitType(op.getLoc(), op.getType())))
      return failure();
    //TODO: handle dynamic sized view here. This assumes all compile time sizes.
    MemRefType type = op.getType().cast<MemRefType>();
    os << "(Kokkos::view_alloc(Kokkos::WithoutInitializing, \"" << op.getSymName() << "\"));\n";
    auto maybeValue = op.getInitialValue();
    if(maybeValue)
    {
      MemRefType memrefType = op.getType();
      int64_t span = KokkosCppEmitter::getMemrefSpan(memrefType);
      //This view has intializing data (span elements)
      os << "Kokkos::View<";
      if(failed(emitType(op.getLoc(), type.getElementType())))
        return failure();
      os << "*> tempDst(" << op.getSymName() << ".data(), " << span << ");\n";
      os << "Kokkos::View<";
      if(failed(emitType(op.getLoc(), type.getElementType())))
        return failure();
      os << "*, Kokkos::HostSpace> tempSrc(" << op.getSymName() << "_initial, " << span << ");\n";
      os << "Kokkos::deep_copy(exec_space(), tempDst, tempSrc);\n";
    }
    os.unindent();
    os << "}\n";
  }
  os.unindent();
  os << "}\n\n";
  os << "extern \"C\" void kokkos_mlir_finalize()\n";
  os << "{\n";
  os.indent();
  for(auto& op : globalViews)
  {
    os << op.getSymName() << " = ";
    if(failed(emitType(op.getLoc(), op.getType())))
      return failure();
    os << "();\n";
  }
  os << "Kokkos::finalize();\n";
  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult KokkosCppEmitter::emitPythonBoilerplate()
{
  *py_os << "import ctypes\n";
  *py_os << "import numpy\n";
  *py_os << "class MLIRKokkosModule:\n";
  *py_os << "  def __init__(self, libPath):\n";
  *py_os << "    print('Hello from MLIRKokkosModule.__init__!')\n";
  *py_os << "    self.libHandle = ctypes.CDLL(libPath)\n";
  // Do all initialization immediately
  *py_os << "    print('Initializing module.')\n";
  *py_os << "    self.libHandle.kokkos_mlir_initialize()\n";
  *py_os << "    print('Done initializing module.')\n";
  *py_os << "  def __del__(self):\n";
  *py_os << "    print('Finalizing module.')\n";
  *py_os << "    self.libHandle.kokkos_mlir_finalize()\n";
  //From here, just function wrappers are emitted as class members.
  //Indent now for all of them.
  py_os->indent();
  return success();
}

LogicalResult KokkosCppEmitter::emitType(Location loc, Type type) {
  if (auto iType = type.dyn_cast<IntegerType>()) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = type.dyn_cast<FloatType>()) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = type.dyn_cast<IndexType>())
    return (os << "size_t"), success();
  if (auto tType = type.dyn_cast<TensorType>()) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = type.dyn_cast<TupleType>())
    return emitTupleType(loc, tType.getTypes());
  if (auto mrType = type.dyn_cast<MemRefType>()) {
    os << "Kokkos::View<";
    if (failed(emitType(loc, mrType.getElementType())))
      return failure();
    for(auto extent : mrType.getShape()) {
      if(mrType.hasStaticShape()) {
          os << '[' << extent << ']';
      }
      else {
          os << '*';
      }
    }
    os << ", Kokkos::LayoutRight>";
    return success();
  }
  if (auto mrType = type.dyn_cast<UnrankedMemRefType>()) {
    os << "Kokkos::View<";
    if (failed(emitType(loc, mrType.getElementType())))
      return failure();
    os << "*>";
    return success();
  }
  if (auto mrType = type.dyn_cast<LLVM::LLVMPointerType>()) {
    if (failed(emitType(loc, mrType.getElementType())))
      return failure();
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type << "\n";
}

LogicalResult KokkosCppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult KokkosCppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

//Version for when we are just emitting C++
LogicalResult emitc::translateToKokkosCpp(Operation *op, raw_ostream &os, bool declareVariablesAtTop) {
  /*
  std::cout << "Hello from K emitter: proc " << getpid() << '\n';
  std::cout << "Enter anything to proceed: ";
  std::string asdf;
  std::cin >> asdf;
  */
  KokkosCppEmitter emitter(os, declareVariablesAtTop);
  //Global preamble.
  emitter << "#include <Kokkos_Core.hpp>\n";
  emitter << "using exec_space = Kokkos::DefaultExecutionSpace;\n\n";
  //Emit the actual module (global variables and functions)
  if(failed(emitter.emitOperation(*op, /*trailingSemicolon=*/false)))
    return failure();
  return success();
}

//Version for when we are emitting both C++ and Python wrappers
LogicalResult emitc::translateToKokkosCpp(Operation *op, raw_ostream &os, raw_ostream &py_os, bool declareVariablesAtTop) {
  /*
  std::cout << "Hello from K emitter: proc " << getpid() << '\n';
  std::cout << "Enter anything to proceed: ";
  std::string asdf;
  std::cin >> asdf;
  */
  KokkosCppEmitter emitter(os, py_os, declareVariablesAtTop);
  //Emit the Python ctypes boilerplate first - function wrappers need to come after this.
  if(failed(emitter.emitPythonBoilerplate()))
      return failure();
  //Global preamble.
  emitter << "#include <Kokkos_Core.hpp>\n";
  emitter << "using exec_space = Kokkos::DefaultExecutionSpace;\n\n";
  emitter << "extern \"C\" void kokkos_mlir_initialize();\n";
  emitter << "extern \"C\" void kokkos_mlir_finalize();\n\n";
  //Emit the actual module (global variables and functions)
  if(failed(emitter.emitOperation(*op, /*trailingSemicolon=*/false)))
    return failure();
  //Emit the init and finalize function definitions.
  if(failed(emitter.emitInitAndFinalize()))
    return failure();
  return success();
}

