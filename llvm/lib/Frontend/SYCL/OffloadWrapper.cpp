//===- SYCLOffloadWrapper.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/SYCL/OffloadWrapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <memory>
#include <string>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::offloading;
using namespace llvm::offloading::sycl;

using OffloadingImage = OffloadBinary::OffloadingImage;

namespace {

/// Wrapper helper class that creates all LLVM IRs wrapping given images.
struct Wrapper {
  Module &M;
  LLVMContext &C;
  SYCLWrappingOptions Options;

  StructType *EntryTy = nullptr;
  StructType *SyclDeviceImageTy = nullptr;
  StructType *SyclBinDescTy = nullptr;

  Wrapper(Module &M, const SYCLWrappingOptions &Options)
      : M(M), C(M.getContext()), Options(Options) {

    EntryTy = offloading::getEntryTy(M);
    SyclDeviceImageTy = getSyclDeviceImageTy();
    SyclBinDescTy = getSyclBinDescTy();
  }

  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerSize()) {
    case 4:
      return Type::getInt32Ty(C);
    case 8:
      return Type::getInt64Ty(C);
    }
    llvm_unreachable("unsupported pointer type size");
  }

  SmallVector<Constant *, 2> getSizetConstPair(size_t First, size_t Second) {
    IntegerType *SizeTTy = getSizeTTy();
    return SmallVector<Constant *, 2>{ConstantInt::get(SizeTTy, First),
                                      ConstantInt::get(SizeTTy, Second)};
  }

  /// Note: Properties aren't supported and the support is going
  /// to be added later.
  /// Creates a structure corresponding to:
  /// SYCL specific image descriptor type.
  /// \code
  /// struct __sycl.tgt_device_image {
  ///   // version of this structure - for backward compatibility;
  ///   // all modifications which change order/type/offsets of existing fields
  ///   // should increment the version.
  ///   uint16_t Version;
  ///   // the kind of offload model the image employs.
  ///   uint8_t OffloadKind;
  ///   // format of the image data - SPIRV, LLVMIR bitcode, etc
  ///   uint8_t Format;
  ///   // null-terminated string representation of the device's target
  ///   // architecture
  ///   const char *Arch;
  ///   // a null-terminated string; target- and compiler-specific options
  ///   // which are suggested to use to "compile" program at runtime
  ///   const char *CompileOptions;
  ///   // a null-terminated string; target- and compiler-specific options
  ///   // which are suggested to use to "link" program at runtime
  ///   const char *LinkOptions;
  ///   // Pointer to the device binary image start
  ///   void *ImageStart;
  ///   // Pointer to the device binary image end
  ///   void *ImageEnd;
  ///   // the entry table
  ///   __tgt_offload_entry *EntriesBegin;
  ///   __tgt_offload_entry *EntriesEnd;
  ///   const char *PropertiesBegin;
  ///   const char *PropertiesEnd;
  /// };
  /// \endcode
  StructType *getSyclDeviceImageTy() {
    return StructType::create(
        {
            Type::getInt16Ty(C),       // Version
            Type::getInt8Ty(C),        // OffloadKind
            Type::getInt8Ty(C),        // Format
            PointerType::getUnqual(C), // Arch
            PointerType::getUnqual(C), // CompileOptions
            PointerType::getUnqual(C), // LinkOptions
            PointerType::getUnqual(C), // ImageStart
            PointerType::getUnqual(C), // ImageEnd
            PointerType::getUnqual(C), // EntriesBegin
            PointerType::getUnqual(C), // EntriesEnd
            PointerType::getUnqual(C), // PropertiesBegin
            PointerType::getUnqual(C)  // PropertiesEnd
        },
        "__sycl.tgt_device_image");
  }

  /// Creates a structure for SYCL specific binary descriptor type. Corresponds
  /// to:
  ///
  /// \code
  ///  struct __sycl.tgt_bin_desc {
  ///    // version of this structure - for backward compatibility;
  ///    // all modifications which change order/type/offsets of existing fields
  ///    // should increment the version.
  ///    uint16_t Version;
  ///    uint16_t NumDeviceImages;
  ///    __sycl.tgt_device_image *DeviceImages;
  ///    // the offload entry table
  ///    __tgt_offload_entry *HostEntriesBegin;
  ///    __tgt_offload_entry *HostEntriesEnd;
  ///  };
  /// \endcode
  StructType *getSyclBinDescTy() {
    return StructType::create(
        {Type::getInt16Ty(C), Type::getInt16Ty(C), PointerType::getUnqual(C),
         PointerType::getUnqual(C), PointerType::getUnqual(C)},
        "__sycl.tgt_bin_desc");
  }

  /// Adds a global readonly variable that is initialized by given
  /// \p Initializer to the module.
  GlobalVariable *addGlobalArrayVariable(const Twine &Name,
                                         ArrayRef<char> Initializer,
                                         const Twine &Section = "") {
    auto *Arr = ConstantDataArray::get(M.getContext(), Initializer);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    SmallVector<char, 32> NameBuf;
    auto SectionName = Section.toStringRef(NameBuf);
    if (!SectionName.empty())
      Var->setSection(SectionName);
    return Var;
  }

  /// Adds given \p Buf as a global variable into the module.
  /// \returns Pair of pointers that point at the beginning and the end of the
  /// variable.
  std::pair<Constant *, Constant *>
  addArrayToModule(ArrayRef<char> Buf, const Twine &Name,
                   const Twine &Section = "") {
    auto *Var = addGlobalArrayVariable(Name, Buf, Section);
    auto *ImageB = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                  getSizetConstPair(0, 0));
    auto *ImageE = ConstantExpr::getGetElementPtr(
        Var->getValueType(), Var, getSizetConstPair(0, Buf.size()));
    return std::make_pair(ImageB, ImageE);
  }

  /// Adds given \p Data as constant byte array in the module.
  /// \returns Constant pointer to the added data. The pointer type does not
  /// carry size information.
  Constant *addRawDataToModule(ArrayRef<char> Data, const Twine &Name) {
    auto *Var = addGlobalArrayVariable(Name, Data);
    auto *DataPtr = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                   getSizetConstPair(0, 0));
    return DataPtr;
  }

  /// Creates a global variable of const char* type and creates an
  /// initializer that initializes it with \p Str.
  ///
  /// \returns Link-time constant pointer (constant expr) to that
  /// variable.
  Constant *addStringToModule(StringRef Str, const Twine &Name) {
    auto *Arr = ConstantDataArray::getString(C, Str);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    auto *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    return ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
  }

  /// Creates a global variable of array of structs and initializes
  /// it with the given values in \p ArrayData.
  ///
  /// \returns Pair of Constants that point at array content.
  /// If \p ArrayData is empty then a returned pair contains nullptrs.
  std::pair<Constant *, Constant *>
  addStructArrayToModule(ArrayRef<Constant *> ArrayData, Type *ElemTy) {
    if (ArrayData.empty()) {
      auto *PtrTy = llvm::PointerType::getUnqual(ElemTy->getContext());
      auto *NullPtr = Constant::getNullValue(PtrTy);
      return std::make_pair(NullPtr, NullPtr);
    }

    assert(ElemTy == ArrayData[0]->getType() && "elem type mismatch");
    auto *Arr =
        ConstantArray::get(ArrayType::get(ElemTy, ArrayData.size()), ArrayData);
    auto *ArrGlob = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                       GlobalVariable::InternalLinkage, Arr,
                                       "__sycl_offload_prop_sets_arr");
    auto *ArrB = ConstantExpr::getGetElementPtr(
        ArrGlob->getValueType(), ArrGlob, getSizetConstPair(0, 0));
    auto *ArrE =
        ConstantExpr::getGetElementPtr(ArrGlob->getValueType(), ArrGlob,
                                       getSizetConstPair(0, ArrayData.size()));
    return std::pair<Constant *, Constant *>(ArrB, ArrE);
  }

  /// Creates a global variable that is initiazed with the given \p Entries.
  ///
  /// \returns Pair of Constants that point at entries content.
  std::pair<Constant *, Constant *>
  addOffloadEntriesToModule(StringRef Entries) {
    if (Entries.empty()) {
      auto *NullPtr = Constant::getNullValue(PointerType::getUnqual(C));
      return std::pair<Constant *, Constant *>(NullPtr, NullPtr);
    }

    auto *I64Zero = ConstantInt::get(Type::getInt64Ty(C), 0);
    auto *I32Zero = ConstantInt::get(Type::getInt32Ty(C), 0);
    auto *NullPtr = Constant::getNullValue(PointerType::getUnqual(C));

    SmallVector<Constant *> EntriesInits;
    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(Entries);
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
      Constant *EntryData[] = {
          ConstantExpr::getNullValue(Type::getInt64Ty(C)),
          ConstantInt::get(Type::getInt16Ty(C), 1),
          ConstantInt::get(Type::getInt16Ty(C), object::OffloadKind::OFK_SYCL),
          I32Zero,
          NullPtr,
          addStringToModule(*LI, "__sycl_offload_entry_name"),
          I64Zero,
          I64Zero,
          NullPtr};

      EntriesInits.push_back(ConstantStruct::get(EntryTy, EntryData));
    }

    auto *Arr = ConstantArray::get(ArrayType::get(EntryTy, EntriesInits.size()),
                                   EntriesInits);
    auto *EntriesGV = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                         GlobalVariable::InternalLinkage, Arr,
                                         "__sycl_offload_entries_arr");

    auto *EntriesB = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV, getSizetConstPair(0, 0));
    auto *EntriesE = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV,
        getSizetConstPair(0, EntriesInits.size()));
    return std::make_pair(EntriesB, EntriesE);
  }

  /// Emits a global array that contains \p Address and \P Size. Also add
  /// it into llvm.used to force it to be emitted in the object file.
  void emitRegistrationFunctions(Constant *Address, size_t Size,
                                 const Twine &ImageID,
                                 StringRef OffloadKindTag) {
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(C);
    auto *ImgInfoArr =
        ConstantArray::get(ArrayType::get(IntPtrTy, 2),
                           {ConstantExpr::getPointerCast(Address, IntPtrTy),
                            ConstantInt::get(IntPtrTy, Size)});
    auto *ImgInfoVar = new GlobalVariable(
        M, ImgInfoArr->getType(), true, GlobalVariable::InternalLinkage,
        ImgInfoArr, Twine(OffloadKindTag) + ImageID + ".info");
    ImgInfoVar->setAlignment(
        MaybeAlign(M.getDataLayout().getTypeStoreSize(IntPtrTy) * 2u));
    ImgInfoVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    ImgInfoVar->setSection(".tgtimg");

    // Add image info to the used list to force it to be emitted to the
    // object.
    appendToUsed(M, ImgInfoVar);
  }

  Constant *wrapImage(const OffloadingImage &OI, const Twine &ImageID,
                      StringRef OffloadKindTag) {
    // Note: Intel DPC++ compiler had 2 versions of this structure
    // and clang++ has a third different structure. To avoid ABI incompatibility
    // between generated device images the Version here starts from 3.
    constexpr uint16_t DeviceImageStructVersion = 3;
    Constant *Version =
        ConstantInt::get(Type::getInt16Ty(C), DeviceImageStructVersion);
    Constant *OffloadKindConstant = ConstantInt::get(
        Type::getInt8Ty(C), static_cast<uint8_t>(OI.TheOffloadKind));
    Constant *ImageKindConstant = ConstantInt::get(
        Type::getInt8Ty(C), static_cast<uint8_t>(OI.TheImageKind));
    StringRef Triple = OI.StringData.lookup("triple");
    Constant *TripleConstant =
        addStringToModule(Triple, Twine(OffloadKindTag) + "target." + ImageID);
    Constant *CompileOptions =
        addStringToModule(Options.CompileOptions,
                          Twine(OffloadKindTag) + "opts.compile." + ImageID);
    Constant *LinkOptions = addStringToModule(
        Options.LinkOptions, Twine(OffloadKindTag) + "opts.link." + ImageID);

    // Note: NULL for now.
    std::pair<Constant *, Constant *> PropertiesConstants = {
        Constant::getNullValue(PointerType::getUnqual(C)),
        Constant::getNullValue(PointerType::getUnqual(C))};

    const MemoryBuffer &RawImage = *OI.Image;
    std::pair<Constant *, Constant *> Binary = addArrayToModule(
        ArrayRef<char>(RawImage.getBufferStart(), RawImage.getBufferEnd()),
        Twine(OffloadKindTag) + ImageID + ".data", ".llvm.offloading");

    // For SYCL images offload entries are defined here per image.
    std::pair<Constant *, Constant *> ImageEntriesPtrs =
        addOffloadEntriesToModule(OI.StringData.lookup("symbols"));
    Constant *WrappedBinary = ConstantStruct::get(
        SyclDeviceImageTy, Version, OffloadKindConstant, ImageKindConstant,
        TripleConstant, CompileOptions, LinkOptions, Binary.first,
        Binary.second, ImageEntriesPtrs.first, ImageEntriesPtrs.second,
        PropertiesConstants.first, PropertiesConstants.second);

    emitRegistrationFunctions(Binary.first, RawImage.getBufferSize(), ImageID,
                              OffloadKindTag);

    return WrappedBinary;
  }

  GlobalVariable *combineWrappedImages(ArrayRef<Constant *> WrappedImages,
                                       StringRef OffloadKindTag) {
    auto *ImagesData = ConstantArray::get(
        ArrayType::get(SyclDeviceImageTy, WrappedImages.size()), WrappedImages);
    auto *ImagesGV =
        new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                           GlobalValue::InternalLinkage, ImagesData,
                           Twine(OffloadKindTag) + "device_images");
    ImagesGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    auto *ImagesB = ConstantExpr::getGetElementPtr(ImagesGV->getValueType(),
                                                   ImagesGV, ZeroZero);

    Constant *EntriesB = Constant::getNullValue(PointerType::getUnqual(C));
    Constant *EntriesE = Constant::getNullValue(PointerType::getUnqual(C));
    static constexpr uint16_t BinDescStructVersion = 1;
    auto *DescInit = ConstantStruct::get(
        SyclBinDescTy,
        ConstantInt::get(Type::getInt16Ty(C), BinDescStructVersion),
        ConstantInt::get(Type::getInt16Ty(C), WrappedImages.size()), ImagesB,
        EntriesB, EntriesE);

    return new GlobalVariable(M, DescInit->getType(), /*isConstant*/ true,
                              GlobalValue::InternalLinkage, DescInit,
                              Twine(OffloadKindTag) + "descriptor");
  }

  /// Creates binary descriptor for the given device images. Binary descriptor
  /// is an object that is passed to the offloading runtime at program startup
  /// and it describes all device images available in the executable or shared
  /// library. It is defined as follows:
  ///
  /// \code
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__start_offloading_entries0;
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__stop_offloading_entries0;
  /// ...
  ///
  /// __attribute__((visibility("hidden")))
  /// extern const char *CompileOptions = "...";
  /// ...
  /// __attribute__((visibility("hidden")))
  /// extern const char *LinkOptions = "...";
  /// ...
  ///
  /// static const char Image0[] = { ... };
  ///  ...
  /// static const char ImageN[] = { ... };
  ///
  /// static const __sycl.tgt_device_image Images[] = {
  ///   {
  ///     Version,                      // Version
  ///     OffloadKind,                  // OffloadKind
  ///     Format,                       // format of the image - SPIRV, LLVMIR
  ///                                   // bc, etc
  //      TripleString,                 // Arch
  ///     CompileOptions0,              // CompileOptions
  ///     LinkOptions0,                 // LinkOptions
  ///     Image0,                       // ImageStart
  ///     Image0 + N,                   // ImageEnd
  ///     __start_offloading_entries0,  // EntriesBegin
  ///     __stop_offloading_entries0,   // EntriesEnd
  ///     NULL,                         // PropertiesBegin
  ///     NULL,                         // PropertiesEnd
  ///   },
  ///   ...
  /// };
  ///
  /// static const __sycl.tgt_bin_desc FatbinDesc = {
  ///   Version,                             //Version
  ///   sizeof(Images) / sizeof(Images[0]),  //NumDeviceImages
  ///   Images,                              //DeviceImages
  ///   NULL,                                //HostEntriesBegin
  ///   NULL                                 //HostEntriesEnd
  /// };
  /// \endcode
  ///
  /// \returns Global variable that represents FatbinDesc.
  GlobalVariable *createFatbinDesc(ArrayRef<OffloadingImage> Images) {
    StringRef OffloadKindTag = ".sycl_offloading.";
    SmallVector<Constant *> WrappedImages;
    WrappedImages.reserve(Images.size());
    for (size_t I = 0, E = Images.size(); I != E; ++I)
      WrappedImages.push_back(wrapImage(Images[I], Twine(I), OffloadKindTag));

    return combineWrappedImages(WrappedImages, OffloadKindTag);
  }

  void createRegisterFatbinFunction(GlobalVariable *FatbinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  Twine("sycl") + ".descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get RegFuncName function declaration.
    auto *RegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee RegFuncC =
        M.getOrInsertFunction("__sycl_register_lib", RegFuncTy);

    // Construct function body
    IRBuilder Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to constructors.
    appendToGlobalCtors(M, Func, /*Priority*/ 1);
  }

  void createUnregisterFunction(GlobalVariable *FatbinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  "sycl.descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get UnregFuncName function declaration.
    auto *UnRegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee UnRegFuncC =
        M.getOrInsertFunction("__sycl_unregister_lib", UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    appendToGlobalDtors(M, Func, /*Priority*/ 1);
  }
}; // end of Wrapper

} // anonymous namespace

Error llvm::offloading::sycl::wrapSYCLBinaries(llvm::Module &M,
                                               ArrayRef<ArrayRef<char>> Buffers,
                                               SYCLWrappingOptions Options) {
  Wrapper W(M, Options);
  SmallVector<std::unique_ptr<OffloadBinary>> OffloadBinaries;
  OffloadBinaries.reserve(Buffers.size());
  SmallVector<OffloadingImage> Images;
  Images.reserve(Buffers.size());
  for (auto Buf : Buffers) {
    MemoryBufferRef MBR(StringRef(Buf.begin(), Buf.size()), /*Identifier*/ "");
    auto OffloadBinaryOrErr = OffloadBinary::create(MBR);
    if (!OffloadBinaryOrErr)
      return OffloadBinaryOrErr.takeError();

    OffloadBinaries.emplace_back(std::move(*OffloadBinaryOrErr));
    Images.emplace_back(OffloadBinaries.back()->getOffloadingImage());
  }

  GlobalVariable *Desc = W.createFatbinDesc(Images);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No binary descriptors created.");

  W.createRegisterFatbinFunction(Desc);
  W.createUnregisterFunction(Desc);
  return Error::success();
}
