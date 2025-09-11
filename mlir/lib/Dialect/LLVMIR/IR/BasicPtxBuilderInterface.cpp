//===- BasicPtxBuilderInterface.td - PTX builder interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to build PTX (Parallel Thread Execution) from NVVM Ops
// automatically. It is used by NVVM to LLVM pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "ptx-builder"

//===----------------------------------------------------------------------===//
// BasicPtxBuilderInterface
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.cpp.inc"

using namespace mlir;
using namespace NVVM;

static constexpr int64_t kSharedMemorySpace = 3;

static FailureOr<char> getRegisterType(Type type, Location loc) {
  MLIRContext *ctx = type.getContext();
  auto i16 = IntegerType::get(ctx, 16);
  auto i32 = IntegerType::get(ctx, 32);
  auto f32 = Float32Type::get(ctx);

  auto getRegisterTypeForScalar = [&](Type type) -> FailureOr<char> {
    if (type.isInteger(1))
      return 'b';
    if (type.isInteger(16))
      return 'h';
    if (type.isInteger(32))
      return 'r';
    if (type.isInteger(64))
      return 'l';
    if (type.isF32())
      return 'f';
    if (type.isF64())
      return 'd';
    if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(type)) {
      // Shared address spaces is addressed with 32-bit pointers.
      if (ptr.getAddressSpace() == kSharedMemorySpace) {
        return 'r';
      }
      return 'l';
    }
    // register type for struct is not supported.
    mlir::emitError(
        loc, "The register type could not be deduced from MLIR type. The ")
        << type
        << " is not supported. Supported types are:"
           "i1, i16, i32, i64, f32, f64,"
           "pointers.\nPlease use llvm.bitcast if you have different type. "
           "\nSee the constraints from here: "
           "https://docs.nvidia.com/cuda/inline-ptx-assembly/"
           "index.html#constraints";
    return failure();
  };

  // Packed registers
  if (auto v = dyn_cast<VectorType>(type)) {
    assert(v.getNumDynamicDims() == 0 && "Dynamic vectors are not supported");

    int64_t lanes = v.getNumElements();
    Type elem = v.getElementType();

    // Case 1. Single vector
    if (lanes <= 1)
      return getRegisterTypeForScalar(elem);

    // Case 2. Packed registers
    Type widened = elem;
    switch (lanes) {

    case 2:
      if (elem.isF16() || elem.isBF16()) // vector<2xf16>
        widened = f32;
      else if (elem.isFloat(8)) // vector<2xf8>
        widened = i16;
      break;
    case 4:
      if (elem.isInteger(8)) // vector<i8x4>
        widened = i32;
      else if (elem.isFloat(8)) // vector<f8x4>
        widened = f32;
      else if (elem.isFloat(4)) // vector<f4x4>
        widened = i16;
      break;
      // Other packing is not supported
    default:
      break;
    }
    return getRegisterTypeForScalar(widened);
  }

  return getRegisterTypeForScalar(type);
}

static FailureOr<char> getRegisterType(Value v, Location loc) {
  if (v.getDefiningOp<LLVM::ConstantOp>())
    return 'n';
  return getRegisterType(v.getType(), loc);
}

/// Extract every element of a struct value.
static SmallVector<Value> extractStructElements(PatternRewriter &rewriter,
                                                Location loc, Value structVal) {
  auto structTy = dyn_cast<LLVM::LLVMStructType>(structVal.getType());
  assert(structTy && "expected LLVM struct");

  SmallVector<Value> elems;
  for (unsigned i : llvm::seq<unsigned>(0, structTy.getBody().size()))
    elems.push_back(LLVM::ExtractValueOp::create(rewriter, loc, structVal, i));

  return elems;
}

LogicalResult PtxBuilder::insertValue(Value v, PTXRegisterMod itype) {
  LDBG() << v << "\t Modifier : " << itype << "\n";
  registerModifiers.push_back(itype);

  Location loc = interfaceOp->getLoc();
  auto getModifier = [&]() -> const char * {
    switch (itype) {
    case PTXRegisterMod::Read:
      return "";
    case PTXRegisterMod::Write:
      return "=";
    case PTXRegisterMod::ReadWrite:
      // "Read-Write modifier is not actually supported
      // Interface will change it to "=" later and add integer mapping
      return "+";
    }
    llvm_unreachable("Unknown PTX register modifier");
  };

  auto addValue = [&](Value v) {
    if (itype == PTXRegisterMod::Read) {
      ptxOperands.push_back(v);
      return;
    }
    if (itype == PTXRegisterMod::ReadWrite)
      ptxOperands.push_back(v);
    hasResult = true;
  };

  llvm::raw_string_ostream ss(registerConstraints);
  // Handle Structs
  if (auto stype = dyn_cast<LLVM::LLVMStructType>(v.getType())) {
    if (itype == PTXRegisterMod::Write) {
      addValue(v);
    }
    for (auto [idx, t] : llvm::enumerate(stype.getBody())) {
      if (itype != PTXRegisterMod::Write) {
        Value extractValue =
            LLVM::ExtractValueOp::create(rewriter, loc, v, idx);
        addValue(extractValue);
      }
      if (itype == PTXRegisterMod::ReadWrite) {
        ss << idx << ",";
      } else {
        FailureOr<char> regType = getRegisterType(t, loc);
        if (failed(regType))
          return rewriter.notifyMatchFailure(loc,
                                             "failed to get register type");
        ss << getModifier() << regType.value() << ",";
      }
    }
    return success();
  }
  // Handle Scalars
  addValue(v);
  FailureOr<char> regType = getRegisterType(v, loc);
  if (failed(regType))
    return rewriter.notifyMatchFailure(loc, "failed to get register type");
  ss << getModifier() << regType.value() << ",";
  return success();
}

/// Check if the operation needs to pack and unpack results.
static bool
needsPackUnpack(BasicPtxBuilderInterface interfaceOp,
                bool needsManualRegisterMapping,
                SmallVectorImpl<PTXRegisterMod> &registerModifiers) {
  if (needsManualRegisterMapping)
    return false;
  const unsigned writeOnlyVals = interfaceOp->getNumResults();
  const unsigned readWriteVals =
      llvm::count_if(registerModifiers, [](PTXRegisterMod m) {
        return m == PTXRegisterMod::ReadWrite;
      });
  return (writeOnlyVals + readWriteVals) > 1;
}

/// Pack the result types of the interface operation.
/// If the operation has multiple results, it packs them into a struct
/// type. Otherwise, it returns the original result types.
static SmallVector<Type>
packResultTypes(BasicPtxBuilderInterface interfaceOp,
                bool needsManualRegisterMapping,
                SmallVectorImpl<PTXRegisterMod> &registerModifiers,
                SmallVectorImpl<Value> &ptxOperands) {
  MLIRContext *ctx = interfaceOp->getContext();
  TypeRange resultRange = interfaceOp->getResultTypes();

  if (!needsPackUnpack(interfaceOp, needsManualRegisterMapping,
                       registerModifiers)) {
    // Single value path:
    if (interfaceOp->getResults().size() == 1)
      return SmallVector<Type>{resultRange.front()};

    // No declared results: if there is an RW, forward its type.
    for (auto [m, v] : llvm::zip(registerModifiers, ptxOperands))
      if (m == PTXRegisterMod::ReadWrite)
        return SmallVector<Type>{v.getType()};
  }

  SmallVector<Type> packed;
  for (auto [m, v] : llvm::zip(registerModifiers, ptxOperands))
    if (m == PTXRegisterMod::ReadWrite)
      packed.push_back(v.getType());
  for (Type t : resultRange)
    packed.push_back(t);

  if (packed.empty())
    return {};

  auto sTy = LLVM::LLVMStructType::getLiteral(ctx, packed, /*isPacked=*/false);
  return SmallVector<Type>{sTy};
}

/// Canonicalize the register constraints:
///  - Turn every "+X" into "=X"
///  - Append (at the very end) the 0-based indices of tokens that were "+X"
/// Examples:
///  "+f,+f,+r,=r,=r,r,r" -> "=f,=f,=r,=r,=r,r,r,0,1,2"
///  "+f,+f,+r,=r,=r"     -> "=f,=f,=r,=r,=r,0,1,2"
static std::string canonicalizeRegisterConstraints(llvm::StringRef csv) {
  SmallVector<llvm::StringRef> toks;
  SmallVector<std::string> out;
  SmallVector<unsigned> plusIdx;

  csv.split(toks, ',');
  out.reserve(toks.size() + 8);

  for (unsigned i = 0, e = toks.size(); i < e; ++i) {
    StringRef t = toks[i].trim();
    if (t.consume_front("+")) {
      plusIdx.push_back(i);
      out.push_back(("=" + t).str());
    } else {
      out.push_back(t.str());
    }
  }

  // Append indices of original "+X" tokens.
  for (unsigned idx : plusIdx)
    out.push_back(std::to_string(idx));

  // Join back to CSV.
  std::string result;
  result.reserve(csv.size() + plusIdx.size() * 2);
  llvm::raw_string_ostream os(result);
  for (size_t i = 0; i < out.size(); ++i) {
    if (i)
      os << ',';
    os << out[i];
  }
  return os.str();
}

constexpr llvm::StringLiteral kReadWritePrefix{"rw"};
constexpr llvm::StringLiteral kWriteOnlyPrefix{"w"};
constexpr llvm::StringLiteral kReadOnlyPrefix{"r"};

/// Returns a regex that matches {$rwN}, {$wN}, {$rN}
static llvm::Regex getPredicateMappingRegex() {
  llvm::Regex rx(llvm::formatv(R"(\{\$({0}|{1}|{2})([0-9]+)\})",
                               kReadWritePrefix, kWriteOnlyPrefix,
                               kReadOnlyPrefix)
                     .str());
  return rx;
}

void mlir::NVVM::countPlaceholderNumbers(
    StringRef ptxCode, llvm::SmallDenseSet<unsigned int> &seenRW,
    llvm::SmallDenseSet<unsigned int> &seenW,
    llvm::SmallDenseSet<unsigned int> &seenR,
    llvm::SmallVectorImpl<unsigned int> &rwNums,
    llvm::SmallVectorImpl<unsigned int> &wNums,
    llvm::SmallVectorImpl<unsigned int> &rNums) {

  llvm::Regex rx = getPredicateMappingRegex();
  StringRef rest = ptxCode;

  SmallVector<StringRef, 3> m; // 0: full, 1: kind, 2: number
  while (!rest.empty() && rx.match(rest, &m)) {
    unsigned num = 0;
    (void)m[2].getAsInteger(10, num);
    // Insert it into the vector only the first time we see this number
    if (m[1].equals_insensitive(kReadWritePrefix)) {
      if (seenRW.insert(num).second)
        rwNums.push_back(num);
    } else if (m[1].equals_insensitive(kWriteOnlyPrefix)) {
      if (seenW.insert(num).second)
        wNums.push_back(num);
    } else {
      if (seenR.insert(num).second)
        rNums.push_back(num);
    }

    const size_t advance = (size_t)(m[0].data() - rest.data()) + m[0].size();
    rest = rest.drop_front(advance);
  }
}

/// Rewrites `{$rwN}`, `{$wN}`, and `{$rN}` placeholders in `ptxCode` into
/// compact `$K` indices:
///   - All `rw*` first (sorted by N),
///   - Then `w*`,
///   - Then `r*`.
/// If there a predicate, it comes always in the end.
/// Each number is assigned once; duplicates are ignored.
///
/// Example Input:
/// "{
///       reg .pred p;
///       setp.ge.s32 p,   {$r0}, {$r1};"
///       selp.s32 {$rw0}, {$r0}, {$r1}, p;
///       selp.s32 {$rw1}, {$r0}, {$r1}, p;
///       selp.s32 {$w0},  {$r0}, {$r1}, p;
///       selp.s32 {$w1},  {$r0}, {$r1}, p;
/// }\n"
/// Example Output:
/// "{
///       reg .pred p;
///       setp.ge.s32 p, $4, $5;"
///       selp.s32 $0,   $4, $5, p;
///       selp.s32 $1,   $4, $5, p;
///       selp.s32 $2,   $4, $5, p;
///       selp.s32 $3,   $4, $5, p;
/// }\n"
static std::string rewriteAsmPlaceholders(llvm::StringRef ptxCode) {
  llvm::SmallDenseSet<unsigned> seenRW, seenW, seenR;
  llvm::SmallVector<unsigned> rwNums, wNums, rNums;

  // Step 1. Count Register Placeholder numbers
  countPlaceholderNumbers(ptxCode, seenRW, seenW, seenR, rwNums, wNums, rNums);

  // Step 2. Sort the Register Placeholder numbers
  llvm::sort(rwNums);
  llvm::sort(wNums);
  llvm::sort(rNums);

  // Step 3. Create mapping from original to new IDs
  llvm::DenseMap<unsigned, unsigned> rwMap, wMap, rMap;
  unsigned nextId = 0;
  for (unsigned n : rwNums)
    rwMap[n] = nextId++;
  for (unsigned n : wNums)
    wMap[n] = nextId++;
  for (unsigned n : rNums)
    rMap[n] = nextId++;

  // Step 4. Rewrite the PTX code with new IDs
  std::string out;
  out.reserve(ptxCode.size());
  size_t prev = 0;
  StringRef rest = ptxCode;
  SmallVector<StringRef, 3> matches;
  llvm::Regex rx = getPredicateMappingRegex();
  while (!rest.empty() && rx.match(rest, &matches)) {
    // Compute absolute match bounds in the original buffer.
    size_t absStart = (size_t)(matches[0].data() - ptxCode.data());
    size_t absEnd = absStart + matches[0].size();

    // Emit text before the match.
    out.append(ptxCode.data() + prev, ptxCode.data() + absStart);

    // Emit compact $K
    unsigned num = 0;
    (void)matches[2].getAsInteger(10, num);
    unsigned id = 0;
    if (matches[1].equals_insensitive(kReadWritePrefix))
      id = rwMap.lookup(num);
    else if (matches[1].equals_insensitive(kWriteOnlyPrefix))
      id = wMap.lookup(num);
    else
      id = rMap.lookup(num);

    out.push_back('$');
    out += std::to_string(id);

    prev = absEnd;

    const size_t advance =
        (size_t)(matches[0].data() - rest.data()) + matches[0].size();
    rest = rest.drop_front(advance);
  }

  // Step 5. Tail.
  out.append(ptxCode.data() + prev, ptxCode.data() + ptxCode.size());
  return out;
}

LLVM::InlineAsmOp PtxBuilder::build() {
  auto asmDialectAttr = LLVM::AsmDialectAttr::get(interfaceOp->getContext(),
                                                  LLVM::AsmDialect::AD_ATT);

  SmallVector<Type> resultTypes = packResultTypes(
      interfaceOp, needsManualRegisterMapping, registerModifiers, ptxOperands);

  // Remove the last comma from the constraints string.
  if (!registerConstraints.empty() &&
      registerConstraints[registerConstraints.size() - 1] == ',')
    registerConstraints.pop_back();
  registerConstraints = canonicalizeRegisterConstraints(registerConstraints);

  std::string ptxInstruction = interfaceOp.getPtx();
  if (!needsManualRegisterMapping)
    ptxInstruction = rewriteAsmPlaceholders(ptxInstruction);

  // Add the predicate to the asm string.
  if (interfaceOp.getPredicate().has_value() &&
      interfaceOp.getPredicate().value()) {
    std::string predicateStr = "@%";
    predicateStr += std::to_string((ptxOperands.size() - 1));
    ptxInstruction = predicateStr + " " + ptxInstruction;
  }

  // Tablegen doesn't accept $, so we use %, but inline assembly uses $.
  // Replace all % with $
  llvm::replace(ptxInstruction, '%', '$');

  return LLVM::InlineAsmOp::create(
      rewriter, interfaceOp->getLoc(),
      /*result types=*/resultTypes,
      /*operands=*/ptxOperands,
      /*asm_string=*/ptxInstruction,
      /*constraints=*/registerConstraints.data(),
      /*has_side_effects=*/interfaceOp.hasSideEffect(),
      /*is_align_stack=*/false, LLVM::TailCallKind::None,
      /*asm_dialect=*/asmDialectAttr,
      /*operand_attrs=*/ArrayAttr());
}

void PtxBuilder::buildAndReplaceOp() {
  LLVM::InlineAsmOp inlineAsmOp = build();
  LDBG() << "\n Generated PTX \n\t" << inlineAsmOp;

  // Case 0: no result at all → just erase wrapper op.
  if (!hasResult) {
    rewriter.eraseOp(interfaceOp);
    return;
  }

  if (needsManualRegisterMapping) {
    rewriter.replaceOp(interfaceOp, inlineAsmOp->getResults());
    return;
  }

  // Case 1: Simple path, return single scalar
  if (!needsPackUnpack(interfaceOp, needsManualRegisterMapping,
                       registerModifiers)) {
    if (inlineAsmOp->getNumResults() > 0) {
      rewriter.replaceOp(interfaceOp, inlineAsmOp->getResults());
    } else {
      // RW-only case with no declared results: forward the RW value.
      SmallVector<Value> results;
      for (auto [m, v] : llvm::zip(registerModifiers, ptxOperands))
        if (m == PTXRegisterMod::ReadWrite) {
          results.push_back(v);
          break;
        }
      rewriter.replaceOp(interfaceOp, results);
    }
    return;
  }

  const bool hasRW = llvm::any_of(registerModifiers, [](PTXRegisterMod m) {
    return m == PTXRegisterMod::ReadWrite;
  });

  // All multi-value paths produce a single struct result we need to unpack.
  assert(LLVM::LLVMStructType::classof(inlineAsmOp.getResultTypes().front()) &&
         "expected struct return for multi-result inline asm");
  Value structVal = inlineAsmOp.getResult(0);
  SmallVector<Value> unpacked =
      extractStructElements(rewriter, interfaceOp->getLoc(), structVal);

  // Case 2: only declared results (no RW): replace the op with all unpacked.
  if (!hasRW && interfaceOp->getResults().size() > 0) {
    rewriter.replaceOp(interfaceOp, unpacked);
    return;
  }

  // Case 3: RW-only (no declared results): update RW uses and erase wrapper.
  if (hasRW && interfaceOp->getResults().size() == 0) {
    unsigned idx = 0;
    for (auto [m, v] : llvm::zip(registerModifiers, ptxOperands)) {
      if (m != PTXRegisterMod::ReadWrite)
        continue;
      Value repl = unpacked[idx++];
      v.replaceUsesWithIf(repl, [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        return owner != interfaceOp && owner != inlineAsmOp;
      });
    }
    rewriter.eraseOp(interfaceOp);
    return;
  }

  // Case 4: mixed (RW + declared results).
  {
    // First rewrite RW operands in place.
    unsigned idx = 0;
    for (auto [m, v] : llvm::zip(registerModifiers, ptxOperands)) {
      if (m != PTXRegisterMod::ReadWrite)
        continue;
      Value repl = unpacked[idx++];
      v.replaceUsesWithIf(repl, [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        return owner != interfaceOp && owner != inlineAsmOp;
      });
    }
    // The remaining unpacked values correspond to the declared results.
    SmallVector<Value> tail;
    tail.reserve(unpacked.size() - idx);
    for (unsigned i = idx, e = unpacked.size(); i < e; ++i)
      tail.push_back(unpacked[i]);

    rewriter.replaceOp(interfaceOp, tail);
  }
}
