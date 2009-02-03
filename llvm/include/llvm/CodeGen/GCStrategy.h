//===-- llvm/CodeGen/GCStrategy.h - Garbage collection ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GCStrategy coordinates code generation algorithms and implements some itself
// in order to generate code compatible with a target code generator as
// specified in a function's 'gc' attribute. Algorithms are enabled by setting
// flags in a subclass's constructor, and some virtual methods can be
// overridden.
// 
// When requested, the GCStrategy will be populated with data about each
// function which uses it. Specifically:
// 
// - Safe points
//   Garbage collection is generally only possible at certain points in code.
//   GCStrategy can request that the collector insert such points:
//
//     - At and after any call to a subroutine
//     - Before returning from the current function
//     - Before backwards branches (loops)
// 
// - Roots
//   When a reference to a GC-allocated object exists on the stack, it must be
//   stored in an alloca registered with llvm.gcoot.
//
// This information can used to emit the metadata tables which are required by
// the target garbage collector runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GCSTRATEGY_H
#define LLVM_CODEGEN_GCSTRATEGY_H

#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/Support/Registry.h"
#include <string>

namespace llvm {
  
  class GCStrategy;
  
  /// The GC strategy registry uses all the defaults from Registry.
  /// 
  typedef Registry<GCStrategy> GCRegistry;
  
  /// GCStrategy describes a garbage collector algorithm's code generation
  /// requirements, and provides overridable hooks for those needs which cannot
  /// be abstractly described.
  class GCStrategy {
  public:
    typedef std::vector<GCFunctionInfo*> list_type;
    typedef list_type::iterator iterator;
    
  private:
    friend class GCModuleInfo;
    const Module *M;
    std::string Name;
    
    list_type Functions;
    
  protected:
    unsigned NeededSafePoints; //< Bitmask of required safe points.
    bool CustomReadBarriers;   //< Default is to insert loads.
    bool CustomWriteBarriers;  //< Default is to insert stores.
    bool CustomRoots;          //< Default is to pass through to backend.
    bool InitRoots;            //< If set, roots are nulled during lowering.
    bool UsesMetadata;         //< If set, backend must emit metadata tables.
    
  public:
    GCStrategy();
    
    virtual ~GCStrategy();
    
    
    /// getName - The name of the GC strategy, for debugging.
    /// 
    const std::string &getName() const { return Name; }

    /// getModule - The module within which the GC strategy is operating.
    /// 
    const Module &getModule() const { return *M; }

    /// needsSafePoitns - True if safe points of any kind are required. By
    //                    default, none are recorded.
    bool needsSafePoints() const { return NeededSafePoints != 0; }
    
    /// needsSafePoint(Kind) - True if the given kind of safe point is
    //                          required. By default, none are recorded.
    bool needsSafePoint(GC::PointKind Kind) const {
      return (NeededSafePoints & 1 << Kind) != 0;
    }
    
    /// customWriteBarrier - By default, write barriers are replaced with simple
    ///                      store instructions. If true, then
    ///                      performCustomLowering must instead lower them.
    bool customWriteBarrier() const { return CustomWriteBarriers; }
    
    /// customReadBarrier - By default, read barriers are replaced with simple
    ///                     load instructions. If true, then
    ///                     performCustomLowering must instead lower them.
    bool customReadBarrier() const { return CustomReadBarriers; }
    
    /// customRoots - By default, roots are left for the code generator so it
    ///               can generate a stack map. If true, then
    //                performCustomLowering must delete them.
    bool customRoots() const { return CustomRoots; }
    
    /// initializeRoots - If set, gcroot intrinsics should initialize their
    //                    allocas to null before the first use. This is
    //                    necessary for most GCs and is enabled by default.
    bool initializeRoots() const { return InitRoots; }
    
    /// usesMetadata - If set, appropriate metadata tables must be emitted by
    ///                the back-end (assembler, JIT, or otherwise).
    bool usesMetadata() const { return UsesMetadata; }
    
    /// begin/end - Iterators for function metadata.
    /// 
    iterator begin() { return Functions.begin(); }
    iterator end()   { return Functions.end();   }

    /// insertFunctionMetadata - Creates metadata for a function.
    /// 
    GCFunctionInfo *insertFunctionInfo(const Function &F);

    /// initializeCustomLowering/performCustomLowering - If any of the actions
    /// are set to custom, performCustomLowering must be overriden to transform
    /// the corresponding actions to LLVM IR. initializeCustomLowering is
    /// optional to override. These are the only GCStrategy methods through
    /// which the LLVM IR can be modified.
    virtual bool initializeCustomLowering(Module &F);
    virtual bool performCustomLowering(Function &F);
  };
  
}

#endif
