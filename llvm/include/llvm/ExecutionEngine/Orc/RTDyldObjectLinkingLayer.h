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
protected:
  /// @brief Holds a set of objects to be allocated/linked as a unit in the JIT.
  ///
  /// An instance of this class will be created for each set of objects added
  /// via JITObjectLayer::addObjectSet. Deleting the instance (via
  /// removeObjectSet) frees its memory, removing all symbol definitions that
  /// had been provided by this instance. Higher level layers are responsible
  /// for taking any action required to handle the missing symbols.
  class LinkedObjectSet {
  public:
    LinkedObjectSet() = default;
    LinkedObjectSet(const LinkedObjectSet&) = delete;
    void operator=(const LinkedObjectSet&) = delete;
    virtual ~LinkedObjectSet() = default;

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

  using LinkedObjectSetListT = std::list<std::unique_ptr<LinkedObjectSet>>;

public:
  /// @brief Handle to a set of loaded objects.
  using ObjSetHandleT = LinkedObjectSetListT::iterator;
};

/// @brief Default (no-op) action to perform when loading objects.
class DoNothingOnNotifyLoaded {
public:
  template <typename ObjSetT, typename LoadResult>
  void operator()(RTDyldObjectLinkingLayerBase::ObjSetHandleT, const ObjSetT &,
                  const LoadResult &) {}
};

/// @brief Bare bones object linking layer.
///
///   This class is intended to be used as the base layer for a JIT. It allows
/// object files to be loaded into memory, linked, and the addresses of their
/// symbols queried. All objects added to this layer can see each other's
/// symbols.
template <typename NotifyLoadedFtor = DoNothingOnNotifyLoaded>
class RTDyldObjectLinkingLayer : public RTDyldObjectLinkingLayerBase {
public:
  /// @brief Functor for receiving finalization notifications.
  using NotifyFinalizedFtor = std::function<void(ObjSetHandleT)>;

private:
  template <typename ObjSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT, typename FinalizerFtor>
  class ConcreteLinkedObjectSet : public LinkedObjectSet {
  public:
    ConcreteLinkedObjectSet(ObjSetT Objects, MemoryManagerPtrT MemMgr,
                            SymbolResolverPtrT Resolver,
                            FinalizerFtor Finalizer,
                            bool ProcessAllSections)
      : MemMgr(std::move(MemMgr)),
        PFC(llvm::make_unique<PreFinalizeContents>(std::move(Objects),
                                                   std::move(Resolver),
                                                   std::move(Finalizer),
                                                   ProcessAllSections)) {
      buildInitialSymbolTable(PFC->Objects);
    }

    ~ConcreteLinkedObjectSet() override {
      MemMgr->deregisterEHFrames();
    }
    
    void setHandle(ObjSetHandleT H) {
      PFC->Handle = H;
    }

    void finalize() override {
      assert(PFC && "mapSectionAddress called on finalized LinkedObjectSet");

      RuntimeDyld RTDyld(*MemMgr, *PFC->Resolver);
      RTDyld.setProcessAllSections(PFC->ProcessAllSections);
      PFC->RTDyld = &RTDyld;

      this->Finalized = true;
      PFC->Finalizer(PFC->Handle, RTDyld, std::move(PFC->Objects),
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
      assert(PFC && "mapSectionAddress called on finalized LinkedObjectSet");
      assert(PFC->RTDyld && "mapSectionAddress called on raw LinkedObjectSet");
      PFC->RTDyld->mapSectionAddress(LocalAddress, TargetAddr);
    }

  private:
    void buildInitialSymbolTable(const ObjSetT &Objects) {
      for (const auto &Obj : Objects)
        for (auto &Symbol : getObject(*Obj).symbols()) {
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
      PreFinalizeContents(ObjSetT Objects, SymbolResolverPtrT Resolver,
                          FinalizerFtor Finalizer, bool ProcessAllSections)
        : Objects(std::move(Objects)), Resolver(std::move(Resolver)),
          Finalizer(std::move(Finalizer)),
          ProcessAllSections(ProcessAllSections) {}

      ObjSetT Objects;
      SymbolResolverPtrT Resolver;
      FinalizerFtor Finalizer;
      bool ProcessAllSections;
      ObjSetHandleT Handle;
      RuntimeDyld *RTDyld;
    };

    MemoryManagerPtrT MemMgr;
    std::unique_ptr<PreFinalizeContents> PFC;
  };

  template <typename ObjSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT, typename FinalizerFtor>
  std::unique_ptr<
    ConcreteLinkedObjectSet<ObjSetT, MemoryManagerPtrT,
                            SymbolResolverPtrT, FinalizerFtor>>
  createLinkedObjectSet(ObjSetT Objects, MemoryManagerPtrT MemMgr,
                        SymbolResolverPtrT Resolver,
                        FinalizerFtor Finalizer,
                        bool ProcessAllSections) {
    using LOS = ConcreteLinkedObjectSet<ObjSetT, MemoryManagerPtrT,
                                        SymbolResolverPtrT, FinalizerFtor>;
    return llvm::make_unique<LOS>(std::move(Objects), std::move(MemMgr),
                                  std::move(Resolver), std::move(Finalizer),
                                  ProcessAllSections);
  }

public:
  /// @brief LoadedObjectInfo list. Contains a list of owning pointers to
  ///        RuntimeDyld::LoadedObjectInfo instances.
  using LoadedObjInfoList =
      std::vector<std::unique_ptr<RuntimeDyld::LoadedObjectInfo>>;

  /// @brief Construct an ObjectLinkingLayer with the given NotifyLoaded,
  ///        and NotifyFinalized functors.
  RTDyldObjectLinkingLayer(
      NotifyLoadedFtor NotifyLoaded = NotifyLoadedFtor(),
      NotifyFinalizedFtor NotifyFinalized = NotifyFinalizedFtor())
      : NotifyLoaded(std::move(NotifyLoaded)),
        NotifyFinalized(std::move(NotifyFinalized)) {}

  /// @brief Set the 'ProcessAllSections' flag.
  ///
  /// If set to true, all sections in each object file will be allocated using
  /// the memory manager, rather than just the sections required for execution.
  ///
  /// This is kludgy, and may be removed in the future.
  void setProcessAllSections(bool ProcessAllSections) {
    this->ProcessAllSections = ProcessAllSections;
  }

  /// @brief Add a set of objects (or archives) that will be treated as a unit
  ///        for the purposes of symbol lookup and memory management.
  ///
  /// @return A handle that can be used to refer to the loaded objects (for 
  ///         symbol searching, finalization, freeing memory, etc.).
  template <typename ObjSetT,
            typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ObjSetHandleT addObjectSet(ObjSetT Objects,
                             MemoryManagerPtrT MemMgr,
                             SymbolResolverPtrT Resolver) {
    auto Finalizer = [&](ObjSetHandleT H, RuntimeDyld &RTDyld,
                         const ObjSetT &Objs,
                         std::function<void()> LOSHandleLoad) {
      LoadedObjInfoList LoadedObjInfos;

      for (auto &Obj : Objs)
        LoadedObjInfos.push_back(RTDyld.loadObject(this->getObject(*Obj)));

      LOSHandleLoad();

      this->NotifyLoaded(H, Objs, LoadedObjInfos);

      RTDyld.finalizeWithMemoryManagerLocking();

      if (this->NotifyFinalized)
        this->NotifyFinalized(H);
    };

    auto LOS =
      createLinkedObjectSet(std::move(Objects), std::move(MemMgr),
                            std::move(Resolver), std::move(Finalizer),
                            ProcessAllSections);
    // LOS is an owning-ptr. Keep a non-owning one so that we can set the handle
    // below.
    auto *LOSPtr = LOS.get();

    ObjSetHandleT Handle = LinkedObjSetList.insert(LinkedObjSetList.end(),
                                                   std::move(LOS));
    LOSPtr->setHandle(Handle);

    return Handle;
  }

  /// @brief Remove the set of objects associated with handle H.
  ///
  ///   All memory allocated for the objects will be freed, and the sections and
  /// symbols they provided will no longer be available. No attempt is made to
  /// re-emit the missing symbols, and any use of these symbols (directly or
  /// indirectly) will result in undefined behavior. If dependence tracking is
  /// required to detect or resolve such issues it should be added at a higher
  /// layer.
  void removeObjectSet(ObjSetHandleT H) {
    // How do we invalidate the symbols in H?
    LinkedObjSetList.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto I = LinkedObjSetList.begin(), E = LinkedObjSetList.end(); I != E;
         ++I)
      if (auto Symbol = findSymbolIn(I, Name, ExportedSymbolsOnly))
        return Symbol;

    return nullptr;
  }

  /// @brief Search for the given named symbol in the context of the set of
  ///        loaded objects represented by the handle H.
  /// @param H The handle for the object set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given object set.
  JITSymbol findSymbolIn(ObjSetHandleT H, StringRef Name,
                         bool ExportedSymbolsOnly) {
    return (*H)->getSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Map section addresses for the objects associated with the handle H.
  void mapSectionAddress(ObjSetHandleT H, const void *LocalAddress,
                         JITTargetAddress TargetAddr) {
    (*H)->mapSectionAddress(LocalAddress, TargetAddr);
  }

  /// @brief Immediately emit and finalize the object set represented by the
  ///        given handle.
  /// @param H Handle for object set to emit/finalize.
  void emitAndFinalize(ObjSetHandleT H) {
    (*H)->finalize();
  }

private:
  static const object::ObjectFile& getObject(const object::ObjectFile &Obj) {
    return Obj;
  }

  template <typename ObjT>
  static const object::ObjectFile&
  getObject(const object::OwningBinary<ObjT> &Obj) {
    return *Obj.getBinary();
  }

  LinkedObjectSetListT LinkedObjSetList;
  NotifyLoadedFtor NotifyLoaded;
  NotifyFinalizedFtor NotifyFinalized;
  bool ProcessAllSections = false;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RTDYLDOBJECTLINKINGLAYER_H
