//===-- InstrumentorStubPrinter.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The implementation of a generator of Instrumentor's runtime stubs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <system_error>

namespace llvm {
namespace instrumentor {

/// Get the string representation of an argument with type \p Ty. Two strings
/// are returned: one for direct arguments and another for indirect arguments.
/// The flags in \p Flags describe the properties of the argument. See
/// IRTArg::IRArgFlagTy.
static std::pair<std::string, std::string> getAsCType(Type *Ty,
                                                      unsigned Flags) {
  if (Ty->isIntegerTy()) {
    auto BW = Ty->getIntegerBitWidth();
    if (BW == 1)
      return {"bool ", "bool *"};
    auto S = "int" + std::to_string(BW) + "_t ";
    return {S, S + "*"};
  }
  if (Ty->isPointerTy())
    return {Flags & IRTArg::STRING ? "char *" : "void *", "void **"};
  if (Ty->isFloatTy())
    return {"float ", "float *"};
  if (Ty->isDoubleTy())
    return {"double ", "double *"};
  return {"<>", "<>"};
}

/// Get the string representation of the C printf format of an argument with
/// type \p Ty. The flags in \p Flags describe the properties of the argument.
/// See IRTArg::IRArgFlagTy.
static std::string getPrintfFormatString(Type *Ty, unsigned Flags) {
  if (Ty->isIntegerTy()) {
    if (Ty->getIntegerBitWidth() > 32) {
      assert(Ty->getIntegerBitWidth() == 64);
      return "%\" PRId64 \"";
    }
    return "%\" PRId32 \"";
  }
  if (Ty->isPointerTy())
    return Flags & IRTArg::STRING ? "%s" : "%p";
  if (Ty->isFloatTy())
    return "%f";
  if (Ty->isDoubleTy())
    return "%lf";
  return "<>";
}

std::pair<std::string, std::string> IRTCallDescription::createCBodies() const {
  std::string DirectFormat = "printf(\"" + IO.getName().str() +
                             (IO.IP.isPRE() ? " pre" : " post") + " -- ";
  std::string IndirectFormat = DirectFormat;
  std::string DirectArg, IndirectArg, DirectReturnValue, IndirectReturnValue;

  auto AddToFormats = [&](Twine S) {
    DirectFormat += S.str();
    IndirectFormat += S.str();
  };
  auto AddToArgs = [&](Twine S) {
    DirectArg += S.str();
    IndirectArg += S.str();
  };
  bool First = true;
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    if (!First)
      AddToFormats(", ");
    First = false;
    AddToArgs(", " + IRArg.Name);
    AddToFormats(IRArg.Name + ": ");
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectReturnValue = IRArg.Name;
      if (!isPotentiallyIndirect(IRArg))
        IndirectReturnValue = IRArg.Name;
    }

    // Handle value pack arguments specially
    if (IRArg.Flags & IRTArg::VALUE_PACK) {
      DirectFormat += "[value pack at %p]";
      IndirectFormat += "[value pack at %p]";
      continue;
    }

    if (!isPotentiallyIndirect(IRArg)) {
      AddToFormats(getPrintfFormatString(IRArg.Ty, IRArg.Flags));
    } else {
      DirectFormat += getPrintfFormatString(IRArg.Ty, IRArg.Flags);
      IndirectFormat += "%p";
      IndirectArg += "_ptr";
      // Add the indirect argument size
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE)) {
        IndirectFormat += ", " + IRArg.Name.str() + "_size: %\" PRId32 \"";
        IndirectArg += ", " + IRArg.Name.str() + "_size";
      }
    }
  }

  std::string DirectBody = DirectFormat + "\\n\"" + DirectArg + ");\n";
  std::string IndirectBody = IndirectFormat + "\\n\"" + IndirectArg + ");\n";

  // Add value pack element printing
  for (size_t ArgIdx = 0; ArgIdx < IO.IRTArgs.size(); ++ArgIdx) {
    auto &IRArg = IO.IRTArgs[ArgIdx];
    if (!IRArg.Enabled || !(IRArg.Flags & IRTArg::VALUE_PACK))
      continue;

    // Find the count parameter - it should be the previous enabled argument
    std::string CountParam;
    for (int PrevIdx = ArgIdx - 1; PrevIdx >= 0; --PrevIdx) {
      if (IO.IRTArgs[PrevIdx].Enabled &&
          IO.IRTArgs[PrevIdx].Name.equals_insensitive(
              ("num_" + IRArg.Name).str())) {
        CountParam = IO.IRTArgs[PrevIdx].Name.str();
        break;
      }
    }

    // If no count parameter found, use 0 (will skip iteration)
    if (CountParam.empty())
      CountParam = "0 /* count not enabled! */";

    auto AddToBodies = [&](Twine T) {
      DirectBody += T.str();
      IndirectBody += T.str();
    };

    // Direct version: iterate through the value pack at the pointer
    AddToBodies("  ValuePackIterator iter_" + IRArg.Name.str() + ";\n");
    AddToBodies("  initValuePackIterator(&iter_" + IRArg.Name.str() + ", " +
                IRArg.Name.str() + ", " + CountParam + ");\n");
    AddToBodies("  while (iter_" + IRArg.Name.str() + ".index < iter_" +
                IRArg.Name.str() + ".count) {\n");
    AddToBodies("    ValuePackHeader header_" + IRArg.Name.str() +
                " = getValuePackHeader(&iter_" + IRArg.Name.str() + ");\n");
    AddToBodies("    const void *data_" + IRArg.Name.str() +
                " = getValuePackData(&iter_" + IRArg.Name.str() + ");\n");
    AddToBodies("    printf(\"  [%" PRIu32 "] type=%s size=%" PRIu32
                " data=%p\\n\", iter_" +
                IRArg.Name.str() + ".index, getLLVMTypeIDName(header_" +
                IRArg.Name.str() + ".type_id), header_" + IRArg.Name.str() +
                ".size, data_" + IRArg.Name.str() + ");\n");
    AddToBodies("    nextValuePack(&iter_" + IRArg.Name.str() + ");\n");
    AddToBodies("  }\n");
  }

  if (RetTy)
    IndirectReturnValue = DirectReturnValue = "0";
  if (!DirectReturnValue.empty())
    DirectBody += "  return " + DirectReturnValue + ";\n";
  if (!IndirectReturnValue.empty())
    IndirectBody += "  return " + IndirectReturnValue + ";\n";
  return {DirectBody, IndirectBody};
}

std::pair<std::string, std::string>
IRTCallDescription::createCSignature(const InstrumentationConfig &IConf) const {
  SmallVector<std::string> DirectArgs, IndirectArgs;
  std::string DirectRetTy = "void ", IndirectRetTy = "void ";
  for (auto &IRArg : IO.IRTArgs) {
    if (!IRArg.Enabled)
      continue;
    const auto &[DirectArgTy, IndirectArgTy] =
        getAsCType(IRArg.Ty, IRArg.Flags);
    std::string DirectArg = DirectArgTy + IRArg.Name.str();
    std::string IndirectArg = IndirectArgTy + IRArg.Name.str() + "_ptr";
    std::string IndirectArgSize = "int32_t " + IRArg.Name.str() + "_size";
    DirectArgs.push_back(DirectArg);
    if (NumReplaceableArgs == 1 && (IRArg.Flags & IRTArg::REPLACABLE)) {
      DirectRetTy = DirectArgTy;
      if (!isPotentiallyIndirect(IRArg))
        IndirectRetTy = DirectArgTy;
    }
    if (!isPotentiallyIndirect(IRArg)) {
      IndirectArgs.push_back(DirectArg);
    } else {
      IndirectArgs.push_back(IndirectArg);
      if (!(IRArg.Flags & IRTArg::INDIRECT_HAS_SIZE))
        IndirectArgs.push_back(IndirectArgSize);
    }
  }

  auto DirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "");
  auto IndirectName =
      IConf.getRTName(IO.IP.isPRE() ? "pre_" : "post_", IO.getName(), "_ind");
  auto MakeSignature = [&](std::string &RetTy, std::string &Name,
                           SmallVectorImpl<std::string> &Args) {
    return RetTy + Name + "(" + join(Args, ", ") + ")";
  };

  if (RetTy) {
    auto UserRetTy = getAsCType(RetTy, 0).first;
    assert((DirectRetTy == UserRetTy || DirectRetTy == "void ") &&
           (IndirectRetTy == UserRetTy || IndirectRetTy == "void ") &&
           "Explicit return type but also implicit one!");
    IndirectRetTy = DirectRetTy = UserRetTy;
  }
  if (RequiresIndirection)
    return {"", MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
  if (!MightRequireIndirection)
    return {MakeSignature(DirectRetTy, DirectName, DirectArgs), ""};
  return {MakeSignature(DirectRetTy, DirectName, DirectArgs),
          MakeSignature(IndirectRetTy, IndirectName, IndirectArgs)};
}

void printRuntimeHeader(const InstrumentationConfig &IConf,
                        StringRef HeaderFileName, LLVMContext &Ctx) {
  if (HeaderFileName.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(HeaderFileName, EC);
  if (EC) {
    Ctx.emitError(
        Twine("failed to open instrumentor runtime header file for writing: ") +
        EC.message());
    return;
  }

  StringRef Prefix = IConf.getRTName();

  OS << "//===-- Instrumentor Runtime Helper Header "
        "-------------------------------===//\n";
  OS << "//\n";
  OS << "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n";
  OS << "// See https://llvm.org/LICENSE.txt for license information.\n";
  OS << "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n";
  OS << "//\n";
  OS << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  OS << "//\n";
  OS << "// This header provides helper structures and functions for reading "
        "data\n";
  OS << "// generated by the LLVM Instrumentor pass and passed to runtime "
        "functions.\n";
  OS << "//\n";
  OS << "// Generated with runtime prefix: " << Prefix << "\n";
  OS << "//\n";
  OS << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  OS << "#ifndef INSTRUMENTOR_RUNTIME_H\n";
  OS << "#define INSTRUMENTOR_RUNTIME_H\n\n";

  OS << "#ifdef __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#endif\n\n";

  OS << "#include <stdint.h>\n";
  OS << "#include <string.h>\n\n";

  OS << "#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif\n\n";

  // Value pack structures
  OS << "/// Header for each value in a value pack.\n";
  OS << "/// Value packs are used to pass function arguments and other "
        "variable-length\n";
  OS << "/// data to the runtime. The format is:\n";
  OS << "///   [ValueHeader][Padding][Value Data]\n";
  OS << "/// where padding aligns the value data to 8-byte boundaries.\n";
  OS << "typedef struct {\n";
  OS << "  uint32_t size;    // Size of the value in bytes\n";
  OS << "  uint32_t type_id; // LLVM Type::TypeID of the value\n";
  OS << "} ValuePackHeader;\n\n";

  // Value pack iterator
  OS << "/// Iterator for reading values from a value pack.\n";
  OS << "typedef struct {\n";
  OS << "  const char *current;  // Current position in the pack\n";
  OS << "  uint64_t offset;      // Byte offset from the start\n";
  OS << "  uint32_t count;       // Number of elements in the pack\n";
  OS << "  uint32_t index;       // Current element index\n";
  OS << "} ValuePackIterator;\n\n";

  OS << "/// Initialize a value pack iterator.\n";
  OS << "/// \\param iter The iterator to initialize\n";
  OS << "/// \\param pack_ptr Pointer to the start of the value pack\n";
  OS << "/// \\param num_elements Number of elements in the pack\n";
  OS << "static inline void initValuePackIterator(ValuePackIterator *iter, "
        "const void *pack_ptr, uint32_t num_elements) {\n";
  OS << "  iter->current = (const char *)pack_ptr;\n";
  OS << "  iter->offset = 0;\n";
  OS << "  iter->count = num_elements;\n";
  OS << "  iter->index = 0;\n";
  OS << "}\n\n";

  OS << "/// Get the header for the current value.\n";
  OS << "static inline ValuePackHeader getValuePackHeader(const "
        "ValuePackIterator *iter) {\n";
  OS << "  const ValuePackHeader *header = (const ValuePackHeader "
        "*)iter->current;\n";
  OS << "  return *header;\n";
  OS << "}\n\n";

  OS << "/// Get a pointer to the current value data.\n";
  OS << "static inline const void *getValuePackData(const ValuePackIterator "
        "*iter) {\n";
  OS << "  // Skip header (8 bytes: size + type_id)\n";
  OS << "  const char *data_start = iter->current + sizeof(ValuePackHeader);\n";
  OS << "  // Calculate padding for 8-byte alignment\n";
  OS << "  ValuePackHeader header = getValuePackHeader(iter);\n";
  OS << "  uint32_t padding = (8 - (header.size % 8)) % 8;\n";
  OS << "  // Skip padding\n";
  OS << "  return data_start + padding;\n";
  OS << "}\n\n";

  OS << "/// Move to the next value in the pack.\n";
  OS << "static inline void nextValuePack(ValuePackIterator *iter) {\n";
  OS << "  if (iter->index >= iter->count) {\n";
  OS << "    iter->current = NULL;\n";
  OS << "    return;\n";
  OS << "  }\n";
  OS << "  ValuePackHeader header = getValuePackHeader(iter);\n";
  OS << "  uint32_t padding = (8 - (header.size % 8)) % 8;\n";
  OS << "  uint64_t advance = sizeof(ValuePackHeader) + padding + "
        "header.size;\n";
  OS << "  iter->current += advance;\n";
  OS << "  iter->offset += advance;\n";
  OS << "  iter->index++;\n";
  OS << "}\n\n";

  OS << "/// Get the current offset in bytes from the start of the pack.\n";
  OS << "static inline uint64_t getValuePackOffset(const ValuePackIterator "
        "*iter) {\n";
  OS << "  return iter->offset;\n";
  OS << "}\n\n";

  OS << "/// Extract a specific value from a value pack by index.\n";
  OS << "///\n";
  OS << "/// \\param pack_ptr Pointer to the start of the value pack\n";
  OS << "/// \\param num_elements Number of elements in the pack\n";
  OS << "/// \\param index Zero-based index of the value to extract\n";
  OS << "/// \\param header Output parameter for the value header (can be "
        "NULL)\n";
  OS << "/// \\return Pointer to the value data, or NULL if index is out of "
        "bounds\n";
  OS << "static inline const void *getValuePackEntry(const void *pack_ptr, "
        "uint32_t num_elements,\n";
  OS << "                                             uint32_t index, "
        "ValuePackHeader *header) {\n";
  OS << "  if (!pack_ptr || index >= num_elements)\n";
  OS << "    return NULL;\n\n";
  OS << "  ValuePackIterator iter;\n";
  OS << "  initValuePackIterator(&iter, pack_ptr, num_elements);\n\n";
  OS << "  while (iter.current != NULL && iter.index < iter.count) {\n";
  OS << "    ValuePackHeader h = getValuePackHeader(&iter);\n";
  OS << "    if (iter.index == index) {\n";
  OS << "      if (header)\n";
  OS << "        *header = h;\n";
  OS << "      return getValuePackData(&iter);\n";
  OS << "    }\n";
  OS << "    nextValuePack(&iter);\n";
  OS << "  }\n\n";
  OS << "  return NULL; // Index out of bounds\n";
  OS << "}\n\n";

  // LLVM Type IDs enum
  OS << "/// LLVM Type IDs for interpreting value pack data.\n";
  OS << "/// These correspond to llvm::Type::TypeID enum values.\n";
  OS << "enum LLVMTypeID {\n";
  OS << "  HalfTyID = 0,  ///< 16-bit floating point type\n";
  OS << "  BFloatTyID,    ///< 16-bit floating point type (7-bit "
        "significand)\n";
  OS << "  FloatTyID,     ///< 32-bit floating point type\n";
  OS << "  DoubleTyID,    ///< 64-bit floating point type\n";
  OS << "  X86_FP80TyID,  ///< 80-bit floating point type (X87)\n";
  OS << "  FP128TyID,     ///< 128-bit floating point type (112-bit "
        "significand)\n";
  OS << "  PPC_FP128TyID, ///< 128-bit floating point type (two 64-bits, "
        "PowerPC)\n";
  OS << "  VoidTyID,      ///< type with no size\n";
  OS << "  LabelTyID,     ///< Labels\n";
  OS << "  MetadataTyID,  ///< Metadata\n";
  OS << "  X86_AMXTyID,   ///< AMX vectors (8192 bits, X86 specific)\n";
  OS << "  TokenTyID,     ///< Tokens\n";

  OS << "  // Derived types... see DerivedTypes.h file.\n";
  OS << "  IntegerTyID,        ///< Arbitrary bit width integers\n";
  OS << "  ByteTyID,           ///< Arbitrary bit width bytes\n";
  OS << "  FunctionTyID,       ///< Functions\n";
  OS << "  PointerTyID,        ///< Pointers\n";
  OS << "  StructTyID,         ///< Structures\n";
  OS << "  ArrayTyID,          ///< Arrays\n";
  OS << "  FixedVectorTyID,    ///< Fixed width SIMD vector type\n";
  OS << "  ScalableVectorTyID, ///< Scalable SIMD vector type\n";
  OS << "  TypedPointerTyID,   ///< Typed pointer used by some GPU targets\n";
  OS << "  TargetExtTyID,      ///< Target extension type\n";
  OS << "};\n\n";

  // Type ID printer function
  OS << "/// Get the string name of an LLVM Type ID.\n";
  OS << "static inline const char *getLLVMTypeIDName(uint32_t type_id) {\n";
  OS << "  switch (type_id) {\n";
  OS << "  case HalfTyID: return \"half\";\n";
  OS << "  case BFloatTyID: return \"bfloat\";\n";
  OS << "  case FloatTyID: return \"float\";\n";
  OS << "  case DoubleTyID: return \"double\";\n";
  OS << "  case X86_FP80TyID: return \"x86_fp80\";\n";
  OS << "  case FP128TyID: return \"fp128\";\n";
  OS << "  case PPC_FP128TyID: return \"ppc_fp128\";\n";
  OS << "  case VoidTyID: return \"void\";\n";
  OS << "  case LabelTyID: return \"label\";\n";
  OS << "  case MetadataTyID: return \"metadata\";\n";
  OS << "  case X86_AMXTyID: return \"x86_amx\";\n";
  OS << "  case TokenTyID: return \"token\";\n";
  OS << "  case IntegerTyID: return \"integer\";\n";
  OS << "  case ByteTyID: return \"integer\";\n";
  OS << "  case FunctionTyID: return \"function\";\n";
  OS << "  case PointerTyID: return \"pointer\";\n";
  OS << "  case StructTyID: return \"struct\";\n";
  OS << "  case ArrayTyID: return \"array\";\n";
  OS << "  case FixedVectorTyID: return \"fixed_vector\";\n";
  OS << "  case ScalableVectorTyID: return \"scalable_vector\";\n";
  OS << "  case TypedPointerTyID: return \"typed_pointer\";\n";
  OS << "  case TargetExtTyID: return \"target_ext\";\n";
  OS << "  default: return \"unknown\";\n";
  OS << "  }\n";
  OS << "}\n\n";

  // C++ overlays section
  OS << "#ifdef __cplusplus\n\n";

  OS << "// C++ overlays for range-based iteration and quality of life "
        "improvements\n\n";

  // ValuePackRange class
  OS << "/// Range wrapper for value packs enabling range-based for loops.\n";
  OS << "/// Example:\n";
  OS << "///   for (auto val : ValuePackRange(pack_ptr, num_elements)) {\n";
  OS << "///     // val provides access to header and data\n";
  OS << "///   }\n";
  OS << "class ValuePackRange {\n";
  OS << "public:\n";
  OS << "  struct ValueRef {\n";
  OS << "    ValuePackHeader header;\n";
  OS << "    const void *data;\n\n";
  OS << "    uint32_t type_id() const { return header.type_id; }\n";
  OS << "    uint32_t size() const { return header.size; }\n";
  OS << "    const char *type_name() const { return "
        "getLLVMTypeIDName(header.type_id); }\n\n";
  OS << "    template <typename T> const T &as() const {\n";
  OS << "      return *static_cast<const T*>(data);\n";
  OS << "    }\n";
  OS << "    template <typename T> const T *ptr() const {\n";
  OS << "      return static_cast<const T*>(data);\n";
  OS << "    }\n";
  OS << "  };\n\n";

  OS << "  class iterator {\n";
  OS << "  public:\n";
  OS << "    iterator(const void *ptr, uint32_t num_elements, uint64_t "
        "max_offset)\n";
  OS << "        : max_offset_(max_offset) {\n";
  OS << "      initValuePackIterator(&iter_, ptr, num_elements);\n";
  OS << "      if (ptr && !is_valid_position()) iter_.current = nullptr;\n";
  OS << "    }\n\n";
  OS << "    ValueRef operator*() const {\n";
  OS << "      return ValueRef{\n";
  OS << "        getValuePackHeader(&iter_),\n";
  OS << "        getValuePackData(&iter_)\n";
  OS << "      };\n";
  OS << "    }\n\n";
  OS << "    iterator &operator++() {\n";
  OS << "      nextValuePack(&iter_);\n";
  OS << "      if (!is_valid_position()) iter_.current = nullptr;\n";
  OS << "      return *this;\n";
  OS << "    }\n\n";
  OS << "    bool operator!=(const iterator &other) const {\n";
  OS << "      return iter_.current != other.iter_.current;\n";
  OS << "    }\n\n";
  OS << "  private:\n";
  OS << "    bool is_valid_position() const {\n";
  OS << "      if (!iter_.current) return false;\n";
  OS << "      if (iter_.index >= iter_.count) return false;\n";
  OS << "      if (max_offset_ > 0 && iter_.offset >= max_offset_) return "
        "false;\n";
  OS << "      return true;\n";
  OS << "    }\n\n";
  OS << "    ValuePackIterator iter_;\n";
  OS << "    uint64_t max_offset_;\n";
  OS << "  };\n\n";

  OS << "  ValuePackRange(const void *ptr, uint32_t num_elements, uint64_t "
        "max_size = 0)\n";
  OS << "      : ptr_(ptr), num_elements_(num_elements), max_size_(max_size) "
        "{}\n\n";
  OS << "  iterator begin() const { return iterator(ptr_, num_elements_, "
        "max_size_); }\n";
  OS << "  iterator end() const { return iterator(nullptr, 0, 0); }\n\n";
  OS << "private:\n";
  OS << "  const void *ptr_;\n";
  OS << "  uint32_t num_elements_;\n";
  OS << "  uint64_t max_size_;\n";
  OS << "};\n\n";

  // Helper template functions
  OS << "/// Template helper to extract a typed value from a value pack by "
        "index.\n";
  OS << "template <typename T>\n";
  OS << "inline const T *getValueAs(const void *pack_ptr, uint32_t "
        "num_elements, uint32_t index) {\n";
  OS << "  return static_cast<const T*>(getValuePackEntry(pack_ptr, "
        "num_elements, index, nullptr));\n";
  OS << "}\n\n";

  OS << "#endif // __cplusplus\n\n";

  OS << "#endif // INSTRUMENTOR_RUNTIME_H\n";
}

void printRuntimeStub(const InstrumentationConfig &IConf,
                      StringRef StubRuntimeName, LLVMContext &Ctx) {
  if (StubRuntimeName.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(StubRuntimeName, EC);
  if (EC) {
    Ctx.emitError(
        Twine("failed to open instrumentor stub runtime file for writing: ") +
        EC.message());
    return;
  }

  // Generate the header file alongside the stub
  StringRef Prefix = IConf.getRTName();
  std::string HeaderFileName = StubRuntimeName.str();
  size_t DotPos = HeaderFileName.rfind('.');
  if (DotPos != std::string::npos)
    HeaderFileName = HeaderFileName.substr(0, DotPos);
  HeaderFileName += ".h";
  printRuntimeHeader(IConf, HeaderFileName, Ctx);

  OS << "//===-- Instrumentor Runtime Stub "
        "-----------------------------------------===//\n";
  OS << "//\n";
  OS << "// Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n";
  OS << "// See https://llvm.org/LICENSE.txt for license information.\n";
  OS << "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n";
  OS << "//\n";
  OS << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  OS << "//\n";
  OS << "// This file is auto-generated by the LLVM Instrumentor pass.\n";
  OS << "// It provides stub implementations of instrumentation runtime "
        "functions\n";
  OS << "// that print human-readable information about instrumentation "
        "events.\n";
  OS << "//\n";
  OS << "// Generated with runtime prefix: " << Prefix << "\n";
  OS << "//\n";
  OS << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";
  OS << "#include <inttypes.h>\n";
  OS << "#include <stdint.h>\n";
  OS << "#include <stdio.h>\n";
  OS << "#include \"" << llvm::sys::path::filename(HeaderFileName) << "\"\n\n";
  OS << "#ifdef __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#endif\n\n";

  for (auto &ChoiceMap : IConf.IChoices) {
    for (auto &[_, IO] : ChoiceMap) {
      if (!IO->Enabled)
        continue;
      IRTCallDescription IRTCallDesc(*IO, IO->getRetTy(Ctx));
      const auto Signatures = IRTCallDesc.createCSignature(IConf);
      const auto Bodies = IRTCallDesc.createCBodies();
      if (!Signatures.first.empty()) {
        OS << Signatures.first << " {\n";
        OS << "  " << Bodies.first << "}\n\n";
      }
      if (!Signatures.second.empty()) {
        OS << Signatures.second << " {\n";
        OS << "  " << Bodies.second << "}\n\n";
      }
    }
  }

  OS << "#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif\n";
}

} // end namespace instrumentor
} // end namespace llvm
