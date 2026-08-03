//===-- PISATargetStreamer.h - PISA Target Streamer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISATARGETSTREAMER_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISATARGETSTREAMER_H

#include "PISADefines.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include <tuple>
#include <utility>
#include <variant>

namespace llvm {

namespace PISA {

enum LinkageTy { DEFAULT, EXPORT, IMPORT };
StringRef getLinkageTyName(LinkageTy Linkage);

enum class StorageSpace : unsigned {
  GENERIC,
  GLOBAL,
  CONSTANT,
  SHARED,
  PRIVATE,
};
StorageSpace mapAddrSpaceToStorageSpace(unsigned AS);

enum class KernelAttributeType {
  REQD_WORK_GROUP_SIZE,
  VEC_TYPE_HINT,
};

struct KernelAttribute {
  KernelAttributeType KernelAttrType;
  std::variant<std::vector<uint32_t>, std::string, std::vector<std::string>,
               uint32_t>
      KernelAttrValues;
};

// PISA function and directive name
struct FunctionDirectiveAndName {
  // Calling convention.
  CallingConv::ID CC;
  // External linkage.
  LinkageTy Linkage;
  // Function name.
  std::string Name;
  // An optional return LLT for non-kernel functions.
  LLT RetLLT;
  // An optional list of kernel attributes.
  SmallVector<KernelAttribute> KernelAttrs;
};

struct FunctionDeclParam {
  LLT Ty;
};

struct FunctionDeclaration {
  FunctionDirectiveAndName DN;
  SmallVector<FunctionDeclParam> FunctionParams;
};

struct FunctionParameter {
  LLT Ty;
  std::string Prefix;
  unsigned Idx;
};

struct KernelParameter {
  unsigned Size = 0;
  unsigned Align = 0;
  unsigned PtrAlign = 0;
  unsigned AS = ~0;
  std::string ArgName;
  std::string TypeName;
  std::string TypeQualifier;

  bool hasAlign() const { return Align != 0; }
  bool hasPtrAlign() const { return PtrAlign != 0; }
  bool hasAS() const { return AS != (unsigned)~0; }
};
// PISA header info
struct HeaderDcl {
  unsigned Version;
  SmallString<16> Target;
};
// PISA function signature
struct FunctionSignature {
  FunctionDirectiveAndName DN;
  // FIXME: Need to unify kernel and function parameters.
  SmallVector<FunctionParameter> FunctionParams;
  SmallVector<KernelParameter> KernelParams;
  unsigned SourceLine = 0;
};
// PISA function register declarations
struct RegDcls {
  MapVector<std::tuple</*NumElts=*/unsigned, /*BitWidth=*/unsigned,
                       /*Type=*/unsigned>,
            std::vector<std::pair</*Prefix=*/std::string, /*Id=*/unsigned>>>
      Regs;
};

// PISA register declaration types
struct DataType {
  unsigned NumElts;
  unsigned EltSize;
  unsigned RegType;
  DataType(unsigned N, unsigned B, unsigned R)
      : NumElts(N), EltSize(B), RegType(R) {}
};
struct TypeInfo {
  unsigned RegStart;
  unsigned RegCounter;
  LLT Ty;
  std::string Prefix;
};
class DataTypes {
private:
  MapVector<std::tuple</*NumElts=*/unsigned, /*BitWidth=*/unsigned,
                       /*Type=*/unsigned>,
            TypeInfo>
      TypeInfos;

  std::tuple<unsigned, unsigned, unsigned> tupleDT(DataType DT) {
    return std::make_tuple(DT.NumElts, DT.EltSize, DT.RegType);
  }

public:
  static DataType getTypeFromLLT(LLT Ty);
  static LLT getLLTFromType(const DataType &Ty);
  static std::string getPrefixFromLLT(LLT Ty);
  static DataType getTypeFromPrefix(std::string Prefix);

  auto begin() const { return TypeInfos.begin(); }
  auto end() const { return TypeInfos.end(); }

  TypeInfo &getInfo(LLT Ty) { return getInfo(getTypeFromLLT(Ty)); }
  TypeInfo &getInfo(unsigned NumElts, unsigned EltSize, unsigned RegType) {
    return getInfo(DataType(NumElts, EltSize, RegType));
  }
  TypeInfo &getInfo(DataType DT) {
    auto *It = TypeInfos.find(tupleDT(DT));
    if (It == TypeInfos.end())
      llvm_unreachable("Expect that requested DataType is already present in "
                       "the TypeInfos map");
    return It->second;
  }

  TypeInfo &emplaceInfo(unsigned NumElts, unsigned EltSize, unsigned RegType) {
    return emplaceInfo(DataType(NumElts, EltSize, RegType));
  }
  TypeInfo &emplaceInfo(LLT Ty) { return emplaceInfo(getTypeFromLLT(Ty)); };
  TypeInfo &emplaceInfo(DataType DT) {
    LLT Ty = getLLTFromType(DT);
    auto [It, Inserted] = TypeInfos.try_emplace(
        tupleDT(DT), TypeInfo{0, 0, Ty, getPrefixFromLLT(Ty)});
    return It->second;
  }

  void insertInfo(unsigned NumElts, unsigned EltSize, unsigned RegType,
                  TypeInfo TI) {
    insertInfo(DataType(NumElts, EltSize, RegType), std::move(TI));
  }
  void insertInfo(DataType DT, TypeInfo TI) {
    auto [It, Inserted] = TypeInfos.try_emplace(tupleDT(DT), std::move(TI));
    if (!Inserted)
      llvm_unreachable("Expect that inserted DataType is not already present "
                       "in the TypeInfos map");
  }

  // Once all function parameter declarations have been processed, call this
  // to set all RegStart to the register after the last parameter of that Type
  // Ex. if last parameter of Type Double-Word is %d3, then RegStart for the
  // Double-Word TypeInfo should be 4. Note that RegCounter is not changed
  // (in this example, it should still be 4 for d0-d3).
  void finalizeFuncParams() {
    for (auto &[_, TI] : TypeInfos)
      TI.RegStart = TI.RegCounter;
  }
};

struct VariableDcl {
  // External linkage.
  LinkageTy Linkage;
  StorageSpace SS;
  LLT Type;
  Align Alignment;
  // Variable name.
  std::string Name;
  uint64_t Size = 0;
  int StackIndex = -1;
  std::string Section;
  std::string HostAccessName;
};

struct VariableInit {
  struct InitElement {
    LLT Type;
    uint64_t Value;
  };
  SmallVector<InitElement> Initializer;
  struct GlobalExpr {
    // e.g., "@f+8"
    std::string Name;
    int64_t Offset = 0;
  };
  struct Zeros {
    // e.g., .zeros 7
    uint64_t N = 0;
  };
  using SpecialEntry = std::variant<GlobalExpr, Zeros>;
  // This notes that the index in the initializer is actually the given
  // global or ".zero" directive rather than the immediate value there.
  DenseMap<uint64_t, SpecialEntry> Exprs;
};

// Variable declaration and initialization
struct VariableDclInit {
  VariableDcl Dcl;
  VariableInit Init;
};

// Local variable declaration in private and shared space
using LocalVariableDcl = VariableDcl;
struct LocalVariableDcls {
  SmallVector<LocalVariableDcl> Vars;
};

// Global variable declaration in global and const space
using GlobalVariableDcl = VariableDclInit;
struct GlobalVariableDcls {
  SmallVector<GlobalVariableDcl> Vars;
};

} // namespace PISA

class PISATargetStreamer : public MCTargetStreamer {
public:
  PISATargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}
  ~PISATargetStreamer() override;

  /// Emit header directives (.version, .target). Returns the resolved target
  /// CPU string, which may differ from the initial CPU if a .target directive
  /// was present in the input.
  virtual StringRef emitHeader(const PISA::HeaderDcl &) = 0;
  virtual void emitGlobalVariable(const PISA::GlobalVariableDcl &) = 0;
  virtual void emitFunctionSignature(const PISA::FunctionSignature &) = 0;
  virtual void emitFunctionDeclaration(const PISA::FunctionDeclaration &) = 0;
  virtual void emitRegDcls(const PISA::RegDcls &, const PISA::DataTypes &) = 0;
  virtual void emitLocalVariableDcls(const PISA::LocalVariableDcls &) = 0;
  virtual void emitFuncBodyStart() = 0;
  virtual void emitFuncBodyEnd() = 0;

  // Register names stored as <Prefix, Id>
  using RegName = std::tuple<std::string, unsigned>;

  // Array of register names for immutable APIs.
  using RegNamesRange = ArrayRef<RegName>;

  // If MustAdd is true, addRegDcl would assert if This reg is already added.
  virtual void addRegDcl(unsigned NumElts, unsigned EltSize, unsigned RegType,
                         RegNamesRange RegNames, bool MustAdd = true) = 0;
  virtual void addLocalVariableDecl(const PISA::LocalVariableDcl &) = 0;

  void changeSection(const MCSection *CurSection, MCSection *Section,
                     uint32_t SubSection, raw_ostream &OS) override {}

  // Only Asm stream subclass returns true
  virtual bool isAsmStreamer() const { return false; }
  virtual int getCurLine() const { return -1; }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISATARGETSTREAMER_H
