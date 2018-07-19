//===- RTDyldObjectLinkingLayer.h - RTDyld-based jit linking  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for an RTDyld-based, in-process object linking layer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTDYLDOBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_RTDYLDOBJECTLINKINGLAYER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace orc {

class RTDyldObjectLinkingLayerBase {
public:

  using ObjectPtr =
    std::shared_ptr<object::OwningBinary<object::ObjectFile>>;

protected:

  /// @brief Holds an object to be allocated/linked as a unit in the JIT.
  ///
  /// An instance of this class will be created for each object added
  /// via JITObjectLayer::addObject. Deleting the instance (via
  /// removeObject) frees its memory, removing all symbol definitions that
  /// had been provided by this instance. Higher level layers are responsible
  /// for taking any action required to handle the missing symbols.
  class LinkedObject {
  public:
    LinkedObject() = default;
    LinkedObject(const LinkedObject&) = delete;
    void operator=(const LinkedObject&) = delete;
    virtual ~LinkedObject() = default;

    virtual void finalize() = 0;

    virtual JITSymbol::GetAddressFtor
    getSymbolMaterializer(std::string Name) = 0;

    virtual void mapSectionAddress(const void *LocalAddress,
                                   JITTargetAddress TargetAddr) const = 0;

    JITSymbol getSymbol(StringRef Name, bool ExportedSymbolsOnly) {
      auto SymEntry = SymbolTable.find(Name);
      if (SymEntry == SymbolTable.end())
        return nullptr;
      if (!SymEntry->second.getFlags().isExported() && ExportedSymbolsOnly)
        return nullptr;
      if (!Finalized)
        return JITSymbol(getSymbolMaterializer(Name),
                         SymEntry->second.getFlags());
      return JITSymbol(SymEntry->second);
    }

  protected:
    StringMap<JITEvaluatedSymbol> SymbolTable;
    bool Finalized = false;
  };

  using LinkedObjectListT = std::list<std::unique_ptr<LinkedObject>>;

public:
  /// @brief Handle to a loaded object.
  using ObjHandleT = LinkedObjectListT::iterator;
};

/// @brief Bare bones object linking layer.
///
///   This class is intended to be used as the base layer for a JIT. It allows
/// object files to be loaded into memory, linked, and the addresses of their
/// symbols queried. All objects added to this layer can see each other's
/// symbols.
class RTDyldObjectLinkingLayer : public RTDyldObjectLinkingLayerBase {
public:

  using RTDyldObjectLinkingLayerBase::ObjectPtr;

  /// @brief Functor for receiving object-loaded notifications.
  using NotifyLoadedFtor =
    std::function<void(ObjHandleT, const ObjectPtr &Obj,
                       const RuntimeDyld::LoadedObjectInfo &)>;

  /// @brief Functor for receiving finalization notifications.
  using NotifyFinalizedFtor = std::function<void(ObjHandleT)>;

private:


  template <typename MemoryManagerPtrT, typename SymbolResolverPtrT,
            typename FinalizerFtor>
  class ConcreteLinkedObject : public LinkedObject {
  public:
    ConcreteLinkedObject(ObjectPtr Obj, MemoryManagerPtrT MemMgr,
                         SymbolResolverPtrT Resolver,
                         FinalizerFtor Finalizer,
                         bool ProcessAllSections)
      : MemMgr(std::move(MemMgr)),
        PFC(llvm::make_unique<PreFinalizeContents>(std::move(Obj),
                                                   std::move(Resolver),
                                                   std::move(Finalizer),
                                                   ProcessAllSections)) {
      buildInitialSymbolTable(PFC->Obj);
    }

    ~ConcreteLinkedObject() override {
      MemMgr->deregisterEHFrames();
    }

    void setHandle(ObjHandleT H) {
      PFC->Handle = H;
    }

    void finalize() override {
      assert(PFC && "mapSectionAddress called on finalized LinkedObject");

      RuntimeDyld RTDyld(*MemMgr, *PFC->Resolver);
      RTDyld.setProcessAllSections(PFC->ProcessAllSections);
      PFC->RTDyld = &RTDyld;

      this->Finalized = true;
      PFC->Finalizer(PFC->Handle, RTDyld, std::move(PFC->Obj),
                     [&]() {
                       this->updateSymbolTable(RTDyld);
                     });

      // Release resources.
      PFC = nullptr;
    }

    JITSymbol::GetAddressFtor getSymbolMaterializer(std::string Name) override {
      return
        [this, Name]() {
          // The symbol may be materialized between the creation of this lambda
          // and its execution, so we need to double check.
          if (!this->Finalized)
            this->finalize();
          return this->getSymbol(Name, false).getAddress();
        };
    }

    void mapSectionAddress(const void *LocalAddress,
                           JITTargetAddress TargetAddr) const override {
      assert(PFC && "mapSectionAddress called on finalized LinkedObject");
      assert(PFC->RTDyld && "mapSectionAddress called on raw LinkedObject");
      PFC->RTDyld->mapSectionAddress(LocalAddress, TargetAddr);
    }

  private:

    void buildInitialSymbolTable(const ObjectPtr &Obj) {
      for (auto &Symbol : Obj->getBinary()->symbols()) {
        if (Symbol.getFlags() & object::SymbolRef::SF_Undefined)
          continue;
        Expected<StringRef> SymbolName = Symbol.getName();
        // FIXME: Raise an error for bad symbols.
        if (!SymbolName) {
          consumeError(SymbolName.takeError());
          continue;
        }
        auto Flags = JITSymbolFlags::fromObjectSymbol(Symbol);
        SymbolTable.insert(
          std::make_pair(*SymbolName, JITEvaluatedSymbol(0, Flags)));
      }
    }

    void updateSymbolTable(const RuntimeDyld &RTDyld) {
      for (auto &SymEntry : SymbolTable)
        SymEntry.second = RTDyld.getSymbol(SymEntry.first());
    }

    // Contains the information needed prior to finalization: the object files,
    // memory manager, resolver, and flags needed for RuntimeDyld.
    struct PreFinalizeContents {
      PreFinalizeContents(ObjectPtr Obj, SymbolResolverPtrT Resolver,
                          FinalizerFtor Finalizer, bool ProcessAllSections)
        : Obj(std::move(Obj)), Resolver(std::move(Resolver)),
          Finalizer(std::move(Finalizer)),
          ProcessAllSections(ProcessAllSections) {}

      ObjectPtr Obj;
      SymbolResolverPtrT Resolver;
      FinalizerFtor Finalizer;
      bool ProcessAllSections;
      ObjHandleT Handle;
      RuntimeDyld *RTDyld;
    };

    MemoryManagerPtrT MemMgr;
    std::unique_ptr<PreFinalizeContents> PFC;
  };

  template <typename MemoryManagerPtrT, typename SymbolResolverPtrT,
            typename FinalizerFtor>
  std::unique_ptr<
    ConcreteLinkedObject<MemoryManagerPtrT, SymbolResolverPtrT, FinalizerFtor>>
  createLinkedObject(ObjectPtr Obj, MemoryManagerPtrT MemMgr,
                     SymbolResolverPtrT Resolver,
                     FinalizerFtor Finalizer,
                     bool ProcessAllSections) {
    using LOS = ConcreteLinkedObject<MemoryManagerPtrT, SymbolResolverPtrT,
                                     FinalizerFtor>;
    return llvm::make_unique<LOS>(std::move(Obj), std::move(MemMgr),
                                  std::move(Resolver), std::move(Finalizer),
                                  ProcessAllSections);
  }

public:

  /// @brief Functor for creating memory managers.
  using MemoryManagerGetter =
    std::function<std::shared_ptr<RuntimeDyld::MemoryManager>()>;

  /// @brief Construct an ObjectLinkingLayer with the given NotifyLoaded,
  ///        and NotifyFinalized functors.
  RTDyldObjectLinkingLayer(
      MemoryManagerGetter GetMemMgr,
      NotifyLoadedFtor NotifyLoaded = NotifyLoadedFtor(),
      NotifyFinalizedFtor NotifyFinalized = NotifyFinalizedFtor())
      : GetMemMgr(GetMemMgr),
        NotifyLoaded(std::move(NotifyLoaded)),
        NotifyFinalized(std::move(NotifyFinalized)),
        ProcessAllSections(false) {}

  /// @brief Set the 'ProcessAllSections' flag.
  ///
  /// If set to true, all sections in each object file will be allocated using
  /// the memory manager, rather than just the sections required for execution.
  ///
  /// This is kludgy, and may be removed in the future.
  void setProcessAllSections(bool ProcessAllSections) {
    this->ProcessAllSections = ProcessAllSections;
  }

  /// @brief Add an object to the JIT.
  ///
  /// @return A handle that can be used to refer to the loaded object (for 
  ///         symbol searching, finalization, freeing memory, etc.).
  Expected<ObjHandleT> addObject(ObjectPtr Obj,
                                 std::shared_ptr<JITSymbolResolver> Resolver) {
    auto Finalizer = [&](ObjHandleT H, RuntimeDyld &RTDyld,
                         const ObjectPtr &ObjToLoad,
                         std::function<void()> LOSHandleLoad) {
      std::unique_ptr<RuntimeDyld::LoadedObjectInfo> Info =
        RTDyld.loadObject(*ObjToLoad->getBinary());

      LOSHandleLoad();

      if (this->NotifyLoaded)
        this->NotifyLoaded(H, ObjToLoad, *Info);

      RTDyld.finalizeWithMemoryManagerLocking();

      if (this->NotifyFinalized)
        this->NotifyFinalized(H);
    };

    auto LO =
      createLinkedObject(std::move(Obj), GetMemMgr(),
                         std::move(Resolver), std::move(Finalizer),
                         ProcessAllSections);
    // LOS is an owning-ptr. Keep a non-owning one so that we can set the handle
    // below.
    auto *LOPtr = LO.get();

    ObjHandleT Handle = LinkedObjList.insert(LinkedObjList.end(), std::move(LO));
    LOPtr->setHandle(Handle);

    return Handle;
  }

  /// @brief Remove the object associated with handle H.
  ///
  ///   All memory allocated for the object will be freed, and the sections and
  /// symbols it provided will no longer be available. No attempt is made to
  /// re-emit the missing symbols, and any use of these symbols (directly or
  /// indirectly) will result in undefined behavior. If dependence tracking is
  /// required to detect or resolve such issues it should be added at a higher
  /// layer.
  Error removeObject(ObjHandleT H) {
    // How do we invalidate the symbols in H?
    LinkedObjList.erase(H);
    return Error::success();
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto I = LinkedObjList.begin(), E = LinkedObjList.end(); I != E;
         ++I)
      if (auto Symbol = findSymbolIn(I, Name, ExportedSymbolsOnly))
        return Symbol;

    return nullptr;
  }

  /// @brief Search for the given named symbol in the context of the loaded
  ///        object represented by the handle H.
  /// @param H The handle for the object to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given object.
  JITSymbol findSymbolIn(ObjHandleT H, StringRef Name,
                         bool ExportedSymbolsOnly) {
    return (*H)->getSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Map section addresses for the object associated with the handle H.
  void mapSectionAddress(ObjHandleT H, const void *LocalAddress,
                         JITTargetAddress TargetAddr) {
    (*H)->mapSectionAddress(LocalAddress, TargetAddr);
  }

  /// @brief Immediately emit and finalize the object represented by the given
  ///        handle.
  /// @param H Handle for object to emit/finalize.
  Error emitAndFinalize(ObjHandleT H) {
    (*H)->finalize();
    return Error::success();
  }

private:

  LinkedObjectListT LinkedObjList;
  MemoryManagerGetter GetMemMgr;
  NotifyLoadedFtor NotifyLoaded;
  NotifyFinalizedFtor NotifyFinalized;
  bool ProcessAllSections = false;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RTDYLDOBJECTLINKINGLAYER_H
