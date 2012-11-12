//===-- GCMetadata.h - Garbage collector metadata ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the GCFunctionInfo and GCModuleInfo classes, which are
// used as a communication channel from the target code generator to the target
// garbage collectors. This interface allows code generators and garbage
// collectors to be developed independently.
//
// The GCFunctionInfo class logs the data necessary to build a type accurate
// stack map. The code generator outputs:
//
//   - Safe points as specified by the GCStrategy's NeededSafePoints.
//   - Stack offsets for GC roots, as specified by calls to llvm.gcroot
//
// As a refinement, liveness analysis calculates the set of live roots at each
// safe point. Liveness analysis is not presently performed by the code
// generator, so all roots are assumed live.
//
// GCModuleInfo simply collects GCFunctionInfo instances for each Function as
// they are compiled. This accretion is necessary for collectors which must emit
// a stack map for the compilation unit as a whole. Therefore, GCFunctionInfo
// outlives the MachineFunction from which it is derived and must not refer to
// any code generator data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GCMETADATA_H
#define LLVM_CODEGEN_GCMETADATA_H

#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {
  class AsmPrinter;
  class GCStrategy;
  class Constant;
  class MCSymbol;

  namespace GC {
    /// PointKind - The type of a collector-safe point.
    ///
    enum PointKind {
      Loop,    ///< Instr is a loop (backwards branch).
      Return,  ///< Instr is a return instruction.
      PreCall, ///< Instr is a call instruction.
      PostCall ///< Instr is the return address of a call.
    };
  }

  /// GCPoint - Metadata for a collector-safe point in machine code.
  ///
  struct GCPoint {
    GC::PointKind Kind; ///< The kind of the safe point.
    MCSymbol *Label;    ///< A label.
    DebugLoc Loc;

    GCPoint(GC::PointKind K, MCSymbol *L, DebugLoc DL)
        : Kind(K), Label(L), Loc(DL) {}
  };

  /// GCRoot - Metadata for a pointer to an object managed by the garbage
  /// collector.
  struct GCRoot {
    int Num;            ///< Usually a frame index.
    int StackOffset;    ///< Offset from the stack pointer.
    const Constant *Metadata; ///< Metadata straight from the call
                              ///< to llvm.gcroot.

    GCRoot(int N, const Constant *MD) : Num(N), StackOffset(-1), Metadata(MD) {}
  };


  /// GCFunctionInfo - Garbage collection metadata for a single function.
  ///
  class GCFunctionInfo {
  public:
    typedef std::vector<GCPoint>::iterator iterator;
    typedef std::vector<GCRoot>::iterator roots_iterator;
    typedef std::vector<GCRoot>::const_iterator live_iterator;

  private:
    const Function &F;
    GCStrategy &S;
    uint64_t FrameSize;
    std::vector<GCRoot> Roots;
    std::vector<GCPoint> SafePoints;

    // FIXME: Liveness. A 2D BitVector, perhaps?
    //
    //   BitVector Liveness;
    //
    //   bool islive(int point, int root) =
    //     Liveness[point * SafePoints.size() + root]
    //
    // The bit vector is the more compact representation where >3.2% of roots
    // are live per safe point (1.5% on 64-bit hosts).

  public:
    GCFunctionInfo(const Function &F, GCStrategy &S);
    ~GCFunctionInfo();

    /// getFunction - Return the function to which this metadata applies.
    ///
    const Function &getFunction() const { return F; }

    /// getStrategy - Return the GC strategy for the function.
    ///
    GCStrategy &getStrategy() { return S; }

    /// addStackRoot - Registers a root that lives on the stack. Num is the
    ///                stack object ID for the alloca (if the code generator is
    //                 using  MachineFrameInfo).
    void addStackRoot(int Num, const Constant *Metadata) {
      Roots.push_back(GCRoot(Num, Metadata));
    }

    /// removeStackRoot - Removes a root.
    roots_iterator removeStackRoot(roots_iterator position) {
      return Roots.erase(position);
    }

    /// addSafePoint - Notes the existence of a safe point. Num is the ID of the
    /// label just prior to the safe point (if the code generator is using
    /// MachineModuleInfo).
    void addSafePoint(GC::PointKind Kind, MCSymbol *Label, DebugLoc DL) {
      SafePoints.push_back(GCPoint(Kind, Label, DL));
    }

    /// getFrameSize/setFrameSize - Records the function's frame size.
    ///
    uint64_t getFrameSize() const { return FrameSize; }
    void setFrameSize(uint64_t S) { FrameSize = S; }

    /// begin/end - Iterators for safe points.
    ///
    iterator begin() { return SafePoints.begin(); }
    iterator end()   { return SafePoints.end();   }
    size_t size() const { return SafePoints.size(); }

    /// roots_begin/roots_end - Iterators for all roots in the function.
    ///
    roots_iterator roots_begin() { return Roots.begin(); }
    roots_iterator roots_end  () { return Roots.end();   }
    size_t roots_size() const { return Roots.size(); }

    /// live_begin/live_end - Iterators for live roots at a given safe point.
    ///
    live_iterator live_begin(const iterator &p) { return roots_begin(); }
    live_iterator live_end  (const iterator &p) { return roots_end();   }
    size_t live_size(const iterator &p) const { return roots_size(); }
  };


  /// GCModuleInfo - Garbage collection metadata for a whole module.
  ///
  class GCModuleInfo : public ImmutablePass {
    typedef StringMap<GCStrategy*> strategy_map_type;
    typedef std::vector<GCStrategy*> list_type;
    typedef DenseMap<const Function*,GCFunctionInfo*> finfo_map_type;

    strategy_map_type StrategyMap;
    list_type StrategyList;
    finfo_map_type FInfoMap;

    GCStrategy *getOrCreateStrategy(const Module *M, const std::string &Name);

  public:
    typedef list_type::const_iterator iterator;

    static char ID;

    GCModuleInfo();
    ~GCModuleInfo();

    /// clear - Resets the pass. The metadata deleter pass calls this.
    ///
    void clear();

    /// begin/end - Iterators for used strategies.
    ///
    iterator begin() const { return StrategyList.begin(); }
    iterator end()   const { return StrategyList.end();   }

    /// get - Look up function metadata.
    ///
    GCFunctionInfo &getFunctionInfo(const Function &F);
  };

}

#endif
