//===- Var.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Var.h"
#include "DimLvlMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::sparse_tensor::ir_detail;

//===----------------------------------------------------------------------===//
// `VarKind` helpers.
//===----------------------------------------------------------------------===//

/// For use in foreach loops.
static constexpr const VarKind everyVarKind[] = {
    VarKind::Dimension, VarKind::Symbol, VarKind::Level};

//===----------------------------------------------------------------------===//
// `Var` implementation.
//===----------------------------------------------------------------------===//

std::string Var::str() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

void Var::print(AsmPrinter &printer) const { print(printer.getStream()); }

void Var::print(llvm::raw_ostream &os) const {
  os << toChar(getKind()) << getNum();
}

void Var::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

//===----------------------------------------------------------------------===//
// `Ranks` implementation.
//===----------------------------------------------------------------------===//

bool Ranks::operator==(Ranks const &other) const {
  for (const auto vk : everyVarKind)
    if (getRank(vk) != other.getRank(vk))
      return false;
  return true;
}

bool Ranks::isValid(DimLvlExpr expr) const {
  assert(expr);
  // Compute the maximum identifiers for symbol-vars and dim/lvl-vars
  // (each `DimLvlExpr` only allows one kind of non-symbol variable).
  int64_t maxSym = -1, maxVar = -1;
  mlir::getMaxDimAndSymbol<ArrayRef<AffineExpr>>({{expr.getAffineExpr()}},
                                                 maxVar, maxSym);
  // TODO(wrengr): We may want to add a call to `LLVM_DEBUG` like
  // `willBeValidAffineMap` does.  And/or should return `InFlightDiagnostic`
  // instead of bool.
  return maxSym < getSymRank() && maxVar < getRank(expr.getAllowedVarKind());
}

//===----------------------------------------------------------------------===//
// `VarSet` implementation.
//===----------------------------------------------------------------------===//

VarSet::VarSet(Ranks const &ranks) {
  // NOTE: We must not use `reserve` here, since that doesn't change
  // the `size` of the bitvectors and therefore will result in unexpected
  // OOB errors.  Either `resize` or copy/move-ctor work; we opt for the
  // move-ctor since it should be (marginally) more efficient.
  for (const auto vk : everyVarKind)
    impl[vk] = llvm::SmallBitVector(ranks.getRank(vk));
  assert(getRanks() == ranks);
}

bool VarSet::contains(Var var) const {
  // NOTE: We make sure to return false on OOB, for consistency with
  // the `anyCommon` implementation of `VarSet::occursIn(VarSet)`.
  // However beware that, as always with silencing OOB, this can hide
  // bugs in client code.
  const llvm::SmallBitVector &bits = impl[var.getKind()];
  const auto num = var.getNum();
  return num < bits.size() && bits[num];
}

bool VarSet::occursIn(VarSet const &other) const {
  for (const auto vk : everyVarKind)
    if (impl[vk].anyCommon(other.impl[vk]))
      return true;
  return false;
}

bool VarSet::occursIn(DimLvlExpr expr) const {
  if (!expr)
    return false;
  switch (expr.getAffineKind()) {
  case AffineExprKind::Constant:
    return false;
  case AffineExprKind::SymbolId:
    return contains(expr.castSymVar());
  case AffineExprKind::DimId:
    return contains(expr.castDimLvlVar());
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    const auto [lhs, op, rhs] = expr.unpackBinop();
    (void)op;
    return occursIn(lhs) || occursIn(rhs);
  }
  }
  llvm_unreachable("unknown AffineExprKind");
}

void VarSet::add(Var var) {
  // NOTE: `SmallBitVector::operator[]` will raise assertion errors for OOB.
  impl[var.getKind()][var.getNum()] = true;
}

void VarSet::add(VarSet const &other) {
  // NOTE: `SmallBitVector::operator&=` will implicitly resize
  // the bitvector (unlike `BitVector::operator&=`), so we add an
  // assertion against OOB for consistency with the implementation
  // of `VarSet::add(Var)`.
  for (const auto vk : everyVarKind) {
    assert(impl[vk].size() >= other.impl[vk].size());
    impl[vk] &= other.impl[vk];
  }
}

void VarSet::add(DimLvlExpr expr) {
  if (!expr)
    return;
  switch (expr.getAffineKind()) {
  case AffineExprKind::Constant:
    return;
  case AffineExprKind::SymbolId:
    add(expr.castSymVar());
    return;
  case AffineExprKind::DimId:
    add(expr.castDimLvlVar());
    return;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    const auto [lhs, op, rhs] = expr.unpackBinop();
    (void)op;
    add(lhs);
    add(rhs);
    return;
  }
  }
  llvm_unreachable("unknown AffineExprKind");
}

//===----------------------------------------------------------------------===//
// `VarInfo` implementation.
//===----------------------------------------------------------------------===//

void VarInfo::setNum(Var::Num n) {
  assert(!hasNum() && "Var::Num is already set");
  assert(Var::isWF_Num(n) && "Var::Num is too large");
  num = n;
}

//===----------------------------------------------------------------------===//
// `VarEnv` implementation.
//===----------------------------------------------------------------------===//

/// Helper function for `assertUsageConsistency` to better handle SMLoc
/// mismatches.
// TODO(wrengr): If we switch to the `LocatedVar` design, then there's
// no need for anything like `minSMLoc` since `assertUsageConsistency`
// won't need to do anything about locations.
LLVM_ATTRIBUTE_UNUSED static llvm::SMLoc
minSMLoc(AsmParser &parser, llvm::SMLoc sm1, llvm::SMLoc sm2) {
  const auto loc1 = parser.getEncodedSourceLoc(sm1).dyn_cast<FileLineColLoc>();
  assert(loc1 && "Could not get `FileLineColLoc` for first `SMLoc`");
  const auto loc2 = parser.getEncodedSourceLoc(sm2).dyn_cast<FileLineColLoc>();
  assert(loc2 && "Could not get `FileLineColLoc` for second `SMLoc`");
  if (loc1.getFilename() != loc2.getFilename())
    return SMLoc();
  const auto pair1 = std::make_pair(loc1.getLine(), loc1.getColumn());
  const auto pair2 = std::make_pair(loc2.getLine(), loc2.getColumn());
  return pair1 <= pair2 ? sm1 : sm2;
}

bool isInternalConsistent(VarEnv const &env, VarInfo::ID id, StringRef name) {
  const auto &var = env.access(id);
  return (var.getName() == name && var.getID() == id);
}

// NOTE(wrengr): if we can actually obtain an `AsmParser` for `minSMLoc`
// (or find some other way to convert SMLoc to FileLineColLoc), then this
// would no longer be `const VarEnv` (and couldn't be a free-function either).
bool isUsageConsistent(VarEnv const &env, VarInfo::ID id, llvm::SMLoc loc,
                       VarKind vk) {
  const auto &var = env.access(id);
  // Since the same variable can occur at several locations,
  // it would not be appropriate to do `assert(var.getLoc() == loc)`.
  /* TODO(wrengr):
  const auto minLoc = minSMLoc(_, var.getLoc(), loc);
  assert(minLoc && "Location mismatch/incompatibility");
  var.loc = minLoc;
  // */
  return var.getKind() == vk;
}

std::optional<VarInfo::ID> VarEnv::lookup(StringRef name) const {
  // NOTE: `StringMap::lookup` will return a default-constructed value if
  // the key isn't found; which for enums means zero, and therefore makes
  // it impossible to distinguish between actual zero-VarInfo::ID vs not-found.
  // Whereas `StringMap::at` asserts that the key is found, which we don't
  // want either.
  const auto iter = ids.find(name);
  if (iter == ids.end())
    return std::nullopt;
  const auto id = iter->second;
  if (!isInternalConsistent(*this, id, name))
    return std::nullopt;
  return id;
}

std::optional<std::pair<VarInfo::ID, bool>>
VarEnv::create(StringRef name, llvm::SMLoc loc, VarKind vk, bool verifyUsage) {
  const auto &[iter, didInsert] = ids.try_emplace(name, nextID());
  const auto id = iter->second;
  if (didInsert) {
    vars.emplace_back(id, name, loc, vk);
  } else {
  if (!isInternalConsistent(*this, id, name))
    return std::nullopt;
  if (verifyUsage)
    if (!isUsageConsistent(*this, id, loc, vk))
      return std::nullopt;
  }
  return std::make_pair(id, didInsert);
}

std::optional<std::pair<VarInfo::ID, bool>>
VarEnv::lookupOrCreate(Policy creationPolicy, StringRef name, llvm::SMLoc loc,
                       VarKind vk) {
  switch (creationPolicy) {
  case Policy::MustNot: {
    const auto oid = lookup(name);
    if (!oid)
      return std::nullopt;  // Doesn't exist, but must not create.
    if (!isUsageConsistent(*this, *oid, loc, vk))
      return std::nullopt;
    return std::make_pair(*oid, false);
  }
  case Policy::May:
    return create(name, loc, vk, /*verifyUsage=*/true);
  case Policy::Must: {
    const auto res = create(name, loc, vk, /*verifyUsage=*/false);
    const auto didCreate = res->second;
    if (!didCreate)
      return std::nullopt;  // Already exists, but must create.
    return res;
  }
  }
  llvm_unreachable("unknown Policy");
}

Var VarEnv::bindUnusedVar(VarKind vk) { return Var(vk, nextNum[vk]++); }
Var VarEnv::bindVar(VarInfo::ID id) {
  auto &info = access(id);
  const auto var = bindUnusedVar(info.getKind());
  // NOTE: `setNum` already checks wellformedness of the `Var::Num`.
  info.setNum(var.getNum());
  return var;
}

// TODO(wrengr): Alternatively there's `mlir::emitError(Location, Twine const&)`
// which is what `Operation::emitError` uses; though I'm not sure if
// that's appropriate to use here...  But if it is, then that means
// we can have `VarInfo` store `Location` rather than `SMLoc`, which
// means we can use `FusedLoc` to handle the combination issue in
// `VarEnv::lookupOrCreate`.
//
// TODO(wrengr): is there any way to combine multiple IFDs, so that
// we can report all unbound variables instead of just the first one
// encountered?
//
InFlightDiagnostic VarEnv::emitErrorIfAnyUnbound(AsmParser &parser) const {
  for (const auto &var : vars)
    if (!var.hasNum())
      return parser.emitError(var.getLoc(),
                              "Unbound variable: " + var.getName());
  return {};
}

//===----------------------------------------------------------------------===//
