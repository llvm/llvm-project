//===-- LTOModule.cpp - LLVM Link Time Optimizer --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "LTOModule.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/system_error.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

static cl::opt<bool>
EnableFPMAD("enable-fp-mad",
  cl::desc("Enable less precise MAD instructions to be generated"),
  cl::init(false));

static cl::opt<bool>
DisableFPElim("disable-fp-elim",
  cl::desc("Disable frame pointer elimination optimization"),
  cl::init(false));

static cl::opt<bool>
DisableFPElimNonLeaf("disable-non-leaf-fp-elim",
  cl::desc("Disable frame pointer elimination optimization for non-leaf funcs"),
  cl::init(false));

static cl::opt<bool>
EnableUnsafeFPMath("enable-unsafe-fp-math",
  cl::desc("Enable optimizations that may decrease FP precision"),
  cl::init(false));

static cl::opt<bool>
EnableNoInfsFPMath("enable-no-infs-fp-math",
  cl::desc("Enable FP math optimizations that assume no +-Infs"),
  cl::init(false));

static cl::opt<bool>
EnableNoNaNsFPMath("enable-no-nans-fp-math",
  cl::desc("Enable FP math optimizations that assume no NaNs"),
  cl::init(false));

static cl::opt<bool>
EnableHonorSignDependentRoundingFPMath("enable-sign-dependent-rounding-fp-math",
  cl::Hidden,
  cl::desc("Force codegen to assume rounding mode can change dynamically"),
  cl::init(false));

static cl::opt<bool>
GenerateSoftFloatCalls("soft-float",
  cl::desc("Generate software floating point library calls"),
  cl::init(false));

static cl::opt<llvm::FloatABI::ABIType>
FloatABIForCalls("float-abi",
  cl::desc("Choose float ABI type"),
  cl::init(FloatABI::Default),
  cl::values(
    clEnumValN(FloatABI::Default, "default",
               "Target default float ABI type"),
    clEnumValN(FloatABI::Soft, "soft",
               "Soft float ABI (implied by -soft-float)"),
    clEnumValN(FloatABI::Hard, "hard",
               "Hard float ABI (uses FP registers)"),
    clEnumValEnd));

static cl::opt<llvm::FPOpFusion::FPOpFusionMode>
FuseFPOps("fp-contract",
  cl::desc("Enable aggresive formation of fused FP ops"),
  cl::init(FPOpFusion::Standard),
  cl::values(
    clEnumValN(FPOpFusion::Fast, "fast",
               "Fuse FP ops whenever profitable"),
    clEnumValN(FPOpFusion::Standard, "on",
               "Only fuse 'blessed' FP ops."),
    clEnumValN(FPOpFusion::Strict, "off",
               "Only fuse FP ops when the result won't be effected."),
    clEnumValEnd));

static cl::opt<bool>
DontPlaceZerosInBSS("nozero-initialized-in-bss",
  cl::desc("Don't place zero-initialized symbols into bss section"),
  cl::init(false));

static cl::opt<bool>
EnableGuaranteedTailCallOpt("tailcallopt",
  cl::desc("Turn fastcc calls into tail calls by (potentially) changing ABI."),
  cl::init(false));

static cl::opt<bool>
DisableTailCalls("disable-tail-calls",
  cl::desc("Never emit tail calls"),
  cl::init(false));

static cl::opt<unsigned>
OverrideStackAlignment("stack-alignment",
  cl::desc("Override default stack alignment"),
  cl::init(0));

static cl::opt<bool>
EnableRealignStack("realign-stack",
  cl::desc("Realign stack if needed"),
  cl::init(true));

static cl::opt<std::string>
TrapFuncName("trap-func", cl::Hidden,
  cl::desc("Emit a call to trap function rather than a trap instruction"),
  cl::init(""));

static cl::opt<bool>
EnablePIE("enable-pie",
  cl::desc("Assume the creation of a position independent executable."),
  cl::init(false));

static cl::opt<bool>
SegmentedStacks("segmented-stacks",
  cl::desc("Use segmented stacks if possible."),
  cl::init(false));

static cl::opt<bool>
UseInitArray("use-init-array",
  cl::desc("Use .init_array instead of .ctors."),
  cl::init(false));

static cl::opt<unsigned>
SSPBufferSize("stack-protector-buffer-size", cl::init(8),
              cl::desc("Lower bound for a buffer to be considered for "
                       "stack protection"));

LTOModule::LTOModule(llvm::Module *m, llvm::TargetMachine *t)
  : _module(m), _target(t),
    _context(*_target->getMCAsmInfo(), *_target->getRegisterInfo(), NULL),
    _mangler(_context, *_target->getDataLayout()) {}

/// isBitcodeFile - Returns 'true' if the file (or memory contents) is LLVM
/// bitcode.
bool LTOModule::isBitcodeFile(const void *mem, size_t length) {
  return llvm::sys::IdentifyFileType((const char*)mem, length)
    == llvm::sys::Bitcode_FileType;
}

bool LTOModule::isBitcodeFile(const char *path) {
  return llvm::sys::Path(path).isBitcodeFile();
}

/// isBitcodeFileForTarget - Returns 'true' if the file (or memory contents) is
/// LLVM bitcode for the specified triple.
bool LTOModule::isBitcodeFileForTarget(const void *mem, size_t length,
                                       const char *triplePrefix) {
  MemoryBuffer *buffer = makeBuffer(mem, length);
  if (!buffer)
    return false;
  return isTargetMatch(buffer, triplePrefix);
}

bool LTOModule::isBitcodeFileForTarget(const char *path,
                                       const char *triplePrefix) {
  OwningPtr<MemoryBuffer> buffer;
  if (MemoryBuffer::getFile(path, buffer))
    return false;
  return isTargetMatch(buffer.take(), triplePrefix);
}

/// isTargetMatch - Returns 'true' if the memory buffer is for the specified
/// target triple.
bool LTOModule::isTargetMatch(MemoryBuffer *buffer, const char *triplePrefix) {
  std::string Triple = getBitcodeTargetTriple(buffer, getGlobalContext());
  delete buffer;
  return strncmp(Triple.c_str(), triplePrefix, strlen(triplePrefix)) == 0;
}

/// makeLTOModule - Create an LTOModule. N.B. These methods take ownership of
/// the buffer.
LTOModule *LTOModule::makeLTOModule(const char *path, std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer;
  if (error_code ec = MemoryBuffer::getFile(path, buffer)) {
    errMsg = ec.message();
    return NULL;
  }
  return makeLTOModule(buffer.take(), errMsg);
}

LTOModule *LTOModule::makeLTOModule(int fd, const char *path,
                                    size_t size, std::string &errMsg) {
  return makeLTOModule(fd, path, size, size, 0, errMsg);
}

LTOModule *LTOModule::makeLTOModule(int fd, const char *path,
                                    size_t file_size,
                                    size_t map_size,
                                    off_t offset,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer;
  if (error_code ec = MemoryBuffer::getOpenFile(fd, path, buffer, file_size,
                                                map_size, offset, false)) {
    errMsg = ec.message();
    return NULL;
  }
  return makeLTOModule(buffer.take(), errMsg);
}

LTOModule *LTOModule::makeLTOModule(const void *mem, size_t length,
                                    std::string &errMsg) {
  OwningPtr<MemoryBuffer> buffer(makeBuffer(mem, length));
  if (!buffer)
    return NULL;
  return makeLTOModule(buffer.take(), errMsg);
}

void LTOModule::getTargetOptions(TargetOptions &Options) {
  Options.LessPreciseFPMADOption = EnableFPMAD;
  Options.NoFramePointerElim = DisableFPElim;
  Options.NoFramePointerElimNonLeaf = DisableFPElimNonLeaf;
  Options.AllowFPOpFusion = FuseFPOps;
  Options.UnsafeFPMath = EnableUnsafeFPMath;
  Options.NoInfsFPMath = EnableNoInfsFPMath;
  Options.NoNaNsFPMath = EnableNoNaNsFPMath;
  Options.HonorSignDependentRoundingFPMathOption =
    EnableHonorSignDependentRoundingFPMath;
  Options.UseSoftFloat = GenerateSoftFloatCalls;
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;
  Options.NoZerosInBSS = DontPlaceZerosInBSS;
  Options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
  Options.DisableTailCalls = DisableTailCalls;
  Options.StackAlignmentOverride = OverrideStackAlignment;
  Options.RealignStack = EnableRealignStack;
  Options.TrapFuncName = TrapFuncName;
  Options.PositionIndependentExecutable = EnablePIE;
  Options.EnableSegmentedStacks = SegmentedStacks;
  Options.UseInitArray = UseInitArray;
  Options.SSPBufferSize = SSPBufferSize;
}

LTOModule *LTOModule::makeLTOModule(MemoryBuffer *buffer,
                                    std::string &errMsg) {
  static bool Initialized = false;
  if (!Initialized) {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    Initialized = true;
  }

  // parse bitcode buffer
  OwningPtr<Module> m(getLazyBitcodeModule(buffer, getGlobalContext(),
                                           &errMsg));
  if (!m) {
    delete buffer;
    return NULL;
  }

  std::string TripleStr = m->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // find machine architecture for this module
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (!march)
    return NULL;

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  std::string CPU;
  if (Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      CPU = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      CPU = "yonah";
  }
  TargetOptions Options;
  getTargetOptions(Options);
  TargetMachine *target = march->createTargetMachine(TripleStr, CPU, FeatureStr,
                                                     Options);
  LTOModule *Ret = new LTOModule(m.take(), target);
  if (Ret->parseSymbols(errMsg)) {
    delete Ret;
    return NULL;
  }

  return Ret;
}

/// makeBuffer - Create a MemoryBuffer from a memory range.
MemoryBuffer *LTOModule::makeBuffer(const void *mem, size_t length) {
  const char *startPtr = (const char*)mem;
  return MemoryBuffer::getMemBuffer(StringRef(startPtr, length), "", false);
}

/// objcClassNameFromExpression - Get string that the data pointer points to.
bool LTOModule::objcClassNameFromExpression(Constant *c, std::string &name) {
  if (ConstantExpr *ce = dyn_cast<ConstantExpr>(c)) {
    Constant *op = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(op)) {
      Constant *cn = gvn->getInitializer();
      if (ConstantDataArray *ca = dyn_cast<ConstantDataArray>(cn)) {
        if (ca->isCString()) {
          name = ".objc_class_name_" + ca->getAsCString().str();
          return true;
        }
      }
    }
  }
  return false;
}

/// addObjCClass - Parse i386/ppc ObjC class data structure.
void LTOModule::addObjCClass(GlobalVariable *clgv) {
  ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

  // second slot in __OBJC,__class is pointer to superclass name
  std::string superclassName;
  if (objcClassNameFromExpression(c->getOperand(1), superclassName)) {
    NameAndAttributes info;
    StringMap<NameAndAttributes>::value_type &entry =
      _undefines.GetOrCreateValue(superclassName);
    if (!entry.getValue().name) {
      const char *symbolName = entry.getKey().data();
      info.name = symbolName;
      info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
      info.isFunction = false;
      info.symbol = clgv;
      entry.setValue(info);
    }
  }

  // third slot in __OBJC,__class is pointer to class name
  std::string className;
  if (objcClassNameFromExpression(c->getOperand(2), className)) {
    StringSet::value_type &entry = _defines.GetOrCreateValue(className);
    entry.setValue(1);

    NameAndAttributes info;
    info.name = entry.getKey().data();
    info.attributes = LTO_SYMBOL_PERMISSIONS_DATA |
      LTO_SYMBOL_DEFINITION_REGULAR | LTO_SYMBOL_SCOPE_DEFAULT;
    info.isFunction = false;
    info.symbol = clgv;
    _symbols.push_back(info);
  }
}

/// addObjCCategory - Parse i386/ppc ObjC category data structure.
void LTOModule::addObjCCategory(GlobalVariable *clgv) {
  ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

  // second slot in __OBJC,__category is pointer to target class name
  std::string targetclassName;
  if (!objcClassNameFromExpression(c->getOperand(1), targetclassName))
    return;

  NameAndAttributes info;
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(targetclassName);

  if (entry.getValue().name)
    return;

  const char *symbolName = entry.getKey().data();
  info.name = symbolName;
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
  entry.setValue(info);
}

/// addObjCClassRef - Parse i386/ppc ObjC class list data structure.
void LTOModule::addObjCClassRef(GlobalVariable *clgv) {
  std::string targetclassName;
  if (!objcClassNameFromExpression(clgv->getInitializer(), targetclassName))
    return;

  NameAndAttributes info;
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(targetclassName);
  if (entry.getValue().name)
    return;

  const char *symbolName = entry.getKey().data();
  info.name = symbolName;
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
  entry.setValue(info);
}

/// addDefinedDataSymbol - Add a data symbol as defined to the list.
void LTOModule::addDefinedDataSymbol(GlobalValue *v) {
  // Add to list of defined symbols.
  addDefinedSymbol(v, false);

  if (!v->hasSection() /* || !isTargetDarwin */)
    return;

  // Special case i386/ppc ObjC data structures in magic sections:
  // The issue is that the old ObjC object format did some strange
  // contortions to avoid real linker symbols.  For instance, the
  // ObjC class data structure is allocated statically in the executable
  // that defines that class.  That data structures contains a pointer to
  // its superclass.  But instead of just initializing that part of the
  // struct to the address of its superclass, and letting the static and
  // dynamic linkers do the rest, the runtime works by having that field
  // instead point to a C-string that is the name of the superclass.
  // At runtime the objc initialization updates that pointer and sets
  // it to point to the actual super class.  As far as the linker
  // knows it is just a pointer to a string.  But then someone wanted the
  // linker to issue errors at build time if the superclass was not found.
  // So they figured out a way in mach-o object format to use an absolute
  // symbols (.objc_class_name_Foo = 0) and a floating reference
  // (.reference .objc_class_name_Bar) to cause the linker into erroring when
  // a class was missing.
  // The following synthesizes the implicit .objc_* symbols for the linker
  // from the ObjC data structures generated by the front end.

  // special case if this data blob is an ObjC class definition
  if (v->getSection().compare(0, 15, "__OBJC,__class,") == 0) {
    if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClass(gv);
    }
  }

  // special case if this data blob is an ObjC category definition
  else if (v->getSection().compare(0, 18, "__OBJC,__category,") == 0) {
    if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCCategory(gv);
    }
  }

  // special case if this data blob is the list of referenced classes
  else if (v->getSection().compare(0, 18, "__OBJC,__cls_refs,") == 0) {
    if (GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClassRef(gv);
    }
  }
}

/// addDefinedFunctionSymbol - Add a function symbol as defined to the list.
void LTOModule::addDefinedFunctionSymbol(Function *f) {
  // add to list of defined symbols
  addDefinedSymbol(f, true);
}

/// addDefinedSymbol - Add a defined symbol to the list.
void LTOModule::addDefinedSymbol(GlobalValue *def, bool isFunction) {
  // ignore all llvm.* symbols
  if (def->getName().startswith("llvm."))
    return;

  // string is owned by _defines
  SmallString<64> Buffer;
  _mangler.getNameWithPrefix(Buffer, def, false);

  // set alignment part log2() can have rounding errors
  uint32_t align = def->getAlignment();
  uint32_t attr = align ? CountTrailingZeros_32(def->getAlignment()) : 0;

  // set permissions part
  if (isFunction) {
    attr |= LTO_SYMBOL_PERMISSIONS_CODE;
  } else {
    GlobalVariable *gv = dyn_cast<GlobalVariable>(def);
    if (gv && gv->isConstant())
      attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
    else
      attr |= LTO_SYMBOL_PERMISSIONS_DATA;
  }

  // set definition part
  if (def->hasWeakLinkage() || def->hasLinkOnceLinkage() ||
      def->hasLinkerPrivateWeakLinkage())
    attr |= LTO_SYMBOL_DEFINITION_WEAK;
  else if (def->hasCommonLinkage())
    attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
  else
    attr |= LTO_SYMBOL_DEFINITION_REGULAR;

  // set scope part
  if (def->hasHiddenVisibility())
    attr |= LTO_SYMBOL_SCOPE_HIDDEN;
  else if (def->hasProtectedVisibility())
    attr |= LTO_SYMBOL_SCOPE_PROTECTED;
  else if (def->hasExternalLinkage() || def->hasWeakLinkage() ||
           def->hasLinkOnceLinkage() || def->hasCommonLinkage() ||
           def->hasLinkerPrivateWeakLinkage())
    attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  else if (def->hasLinkOnceODRAutoHideLinkage())
    attr |= LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN;
  else
    attr |= LTO_SYMBOL_SCOPE_INTERNAL;

  StringSet::value_type &entry = _defines.GetOrCreateValue(Buffer);
  entry.setValue(1);

  // fill information structure
  NameAndAttributes info;
  StringRef Name = entry.getKey();
  info.name = Name.data();
  assert(info.name[Name.size()] == '\0');
  info.attributes = attr;
  info.isFunction = isFunction;
  info.symbol = def;

  // add to table of symbols
  _symbols.push_back(info);
}

/// addAsmGlobalSymbol - Add a global symbol from module-level ASM to the
/// defined list.
void LTOModule::addAsmGlobalSymbol(const char *name,
                                   lto_symbol_attributes scope) {
  StringSet::value_type &entry = _defines.GetOrCreateValue(name);

  // only add new define if not already defined
  if (entry.getValue())
    return;

  entry.setValue(1);

  NameAndAttributes &info = _undefines[entry.getKey().data()];

  if (info.symbol == 0) {
    // FIXME: This is trying to take care of module ASM like this:
    //
    //   module asm ".zerofill __FOO, __foo, _bar_baz_qux, 0"
    //
    // but is gross and its mother dresses it funny. Have the ASM parser give us
    // more details for this type of situation so that we're not guessing so
    // much.

    // fill information structure
    info.name = entry.getKey().data();
    info.attributes =
      LTO_SYMBOL_PERMISSIONS_DATA | LTO_SYMBOL_DEFINITION_REGULAR | scope;
    info.isFunction = false;
    info.symbol = 0;

    // add to table of symbols
    _symbols.push_back(info);
    return;
  }

  if (info.isFunction)
    addDefinedFunctionSymbol(cast<Function>(info.symbol));
  else
    addDefinedDataSymbol(info.symbol);

  _symbols.back().attributes &= ~LTO_SYMBOL_SCOPE_MASK;
  _symbols.back().attributes |= scope;
}

/// addAsmGlobalSymbolUndef - Add a global symbol from module-level ASM to the
/// undefined list.
void LTOModule::addAsmGlobalSymbolUndef(const char *name) {
  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  _asm_undefines.push_back(entry.getKey().data());

  // we already have the symbol
  if (entry.getValue().name)
    return;

  uint32_t attr = LTO_SYMBOL_DEFINITION_UNDEFINED;;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  NameAndAttributes info;
  info.name = entry.getKey().data();
  info.attributes = attr;
  info.isFunction = false;
  info.symbol = 0;

  entry.setValue(info);
}

/// addPotentialUndefinedSymbol - Add a symbol which isn't defined just yet to a
/// list to be resolved later.
void LTOModule::addPotentialUndefinedSymbol(GlobalValue *decl, bool isFunc) {
  // ignore all llvm.* symbols
  if (decl->getName().startswith("llvm."))
    return;

  // ignore all aliases
  if (isa<GlobalAlias>(decl))
    return;

  SmallString<64> name;
  _mangler.getNameWithPrefix(name, decl, false);

  StringMap<NameAndAttributes>::value_type &entry =
    _undefines.GetOrCreateValue(name);

  // we already have the symbol
  if (entry.getValue().name)
    return;

  NameAndAttributes info;

  info.name = entry.getKey().data();

  if (decl->hasExternalWeakLinkage())
    info.attributes = LTO_SYMBOL_DEFINITION_WEAKUNDEF;
  else
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;

  info.isFunction = isFunc;
  info.symbol = decl;

  entry.setValue(info);
}

namespace {
  class RecordStreamer : public MCStreamer {
  public:
    enum State { NeverSeen, Global, Defined, DefinedGlobal, Used };

  private:
    StringMap<State> Symbols;

    void markDefined(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Global:
        S = DefinedGlobal;
        break;
      case NeverSeen:
      case Defined:
      case Used:
        S = Defined;
        break;
      }
    }
    void markGlobal(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Defined:
        S = DefinedGlobal;
        break;

      case NeverSeen:
      case Global:
      case Used:
        S = Global;
        break;
      }
    }
    void markUsed(const MCSymbol &Symbol) {
      State &S = Symbols[Symbol.getName()];
      switch (S) {
      case DefinedGlobal:
      case Defined:
      case Global:
        break;

      case NeverSeen:
      case Used:
        S = Used;
        break;
      }
    }

    // FIXME: mostly copied for the obj streamer.
    void AddValueSymbols(const MCExpr *Value) {
      switch (Value->getKind()) {
      case MCExpr::Target:
        // FIXME: What should we do in here?
        break;

      case MCExpr::Constant:
        break;

      case MCExpr::Binary: {
        const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
        AddValueSymbols(BE->getLHS());
        AddValueSymbols(BE->getRHS());
        break;
      }

      case MCExpr::SymbolRef:
        markUsed(cast<MCSymbolRefExpr>(Value)->getSymbol());
        break;

      case MCExpr::Unary:
        AddValueSymbols(cast<MCUnaryExpr>(Value)->getSubExpr());
        break;
      }
    }

  public:
    typedef StringMap<State>::const_iterator const_iterator;

    const_iterator begin() {
      return Symbols.begin();
    }

    const_iterator end() {
      return Symbols.end();
    }

    RecordStreamer(MCContext &Context) : MCStreamer(Context) {}

    virtual void EmitInstruction(const MCInst &Inst) {
      // Scan for values.
      for (unsigned i = Inst.getNumOperands(); i--; )
        if (Inst.getOperand(i).isExpr())
          AddValueSymbols(Inst.getOperand(i).getExpr());
    }
    virtual void EmitLabel(MCSymbol *Symbol) {
      Symbol->setSection(*getCurrentSection());
      markDefined(*Symbol);
    }
    virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
      // FIXME: should we handle aliases?
      markDefined(*Symbol);
    }
    virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) {
      if (Attribute == MCSA_Global)
        markGlobal(*Symbol);
    }
    virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size , unsigned ByteAlignment) {
      markDefined(*Symbol);
    }
    virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                  unsigned ByteAlignment) {
      markDefined(*Symbol);
    }

    // Noop calls.
    virtual void ChangeSection(const MCSection *Section) {}
    virtual void InitSections() {}
    virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {}
    virtual void EmitThumbFunc(MCSymbol *Func) {}
    virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}
    virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {}
    virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {}
    virtual void EmitCOFFSymbolStorageClass(int StorageClass) {}
    virtual void EmitCOFFSymbolType(int Type) {}
    virtual void EndCOFFSymbolDef() {}
    virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
    virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {}
    virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
    virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {}
    virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                               unsigned AddrSpace) {}
    virtual void EmitULEB128Value(const MCExpr *Value) {}
    virtual void EmitSLEB128Value(const MCExpr *Value) {}
    virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                      unsigned ValueSize,
                                      unsigned MaxBytesToEmit) {}
    virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit) {}
    virtual bool EmitValueToOffset(const MCExpr *Offset,
                                   unsigned char Value ) { return false; }
    virtual void EmitFileDirective(StringRef Filename) {}
    virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                          const MCSymbol *LastLabel,
                                          const MCSymbol *Label,
                                          unsigned PointerSize) {}
    virtual void FinishImpl() {}
  };
} // end anonymous namespace

/// addAsmGlobalSymbols - Add global symbols from module-level ASM to the
/// defined or undefined lists.
bool LTOModule::addAsmGlobalSymbols(std::string &errMsg) {
  const std::string &inlineAsm = _module->getModuleInlineAsm();
  if (inlineAsm.empty())
    return false;

  OwningPtr<RecordStreamer> Streamer(new RecordStreamer(_context));
  MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(inlineAsm);
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());
  OwningPtr<MCAsmParser> Parser(createMCAsmParser(SrcMgr,
                                                  _context, *Streamer,
                                                  *_target->getMCAsmInfo()));
  const Target &T = _target->getTarget();
  OwningPtr<MCSubtargetInfo>
    STI(T.createMCSubtargetInfo(_target->getTargetTriple(),
                                _target->getTargetCPU(),
                                _target->getTargetFeatureString()));
  OwningPtr<MCTargetAsmParser> TAP(T.createMCAsmParser(*STI, *Parser.get()));
  if (!TAP) {
    errMsg = "target " + std::string(T.getName()) +
      " does not define AsmParser.";
    return true;
  }

  Parser->setTargetParser(*TAP);
  if (Parser->Run(false))
    return true;

  for (RecordStreamer::const_iterator i = Streamer->begin(),
         e = Streamer->end(); i != e; ++i) {
    StringRef Key = i->first();
    RecordStreamer::State Value = i->second;
    if (Value == RecordStreamer::DefinedGlobal)
      addAsmGlobalSymbol(Key.data(), LTO_SYMBOL_SCOPE_DEFAULT);
    else if (Value == RecordStreamer::Defined)
      addAsmGlobalSymbol(Key.data(), LTO_SYMBOL_SCOPE_INTERNAL);
    else if (Value == RecordStreamer::Global ||
             Value == RecordStreamer::Used)
      addAsmGlobalSymbolUndef(Key.data());
  }

  return false;
}

/// isDeclaration - Return 'true' if the global value is a declaration.
static bool isDeclaration(const GlobalValue &V) {
  if (V.hasAvailableExternallyLinkage())
    return true;

  if (V.isMaterializable())
    return false;

  return V.isDeclaration();
}

/// parseSymbols - Parse the symbols from the module and model-level ASM and add
/// them to either the defined or undefined lists.
bool LTOModule::parseSymbols(std::string &errMsg) {
  // add functions
  for (Module::iterator f = _module->begin(), e = _module->end(); f != e; ++f) {
    if (isDeclaration(*f))
      addPotentialUndefinedSymbol(f, true);
    else
      addDefinedFunctionSymbol(f);
  }

  // add data
  for (Module::global_iterator v = _module->global_begin(),
         e = _module->global_end(); v !=  e; ++v) {
    if (isDeclaration(*v))
      addPotentialUndefinedSymbol(v, false);
    else
      addDefinedDataSymbol(v);
  }

  // add asm globals
  if (addAsmGlobalSymbols(errMsg))
    return true;

  // add aliases
  for (Module::alias_iterator a = _module->alias_begin(),
         e = _module->alias_end(); a != e; ++a) {
    if (isDeclaration(*a->getAliasedGlobal()))
      // Is an alias to a declaration.
      addPotentialUndefinedSymbol(a, false);
    else
      addDefinedDataSymbol(a);
  }

  // make symbols for all undefines
  for (StringMap<NameAndAttributes>::iterator u =_undefines.begin(),
         e = _undefines.end(); u != e; ++u) {
    // If this symbol also has a definition, then don't make an undefine because
    // it is a tentative definition.
    if (_defines.count(u->getKey())) continue;
    NameAndAttributes info = u->getValue();
    _symbols.push_back(info);
  }

  return false;
}
