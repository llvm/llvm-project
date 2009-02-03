//===-- RegisterCoalescer.h - Register Coalescing Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the abstract interface for register coalescers, 
// allowing them to interact with and query register allocators.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/IncludeFile.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/ADT/SmallPtrSet.h"

#ifndef LLVM_CODEGEN_REGISTER_COALESCER_H
#define LLVM_CODEGEN_REGISTER_COALESCER_H

namespace llvm {

  class MachineFunction;
  class RegallocQuery;
  class AnalysisUsage;
  class MachineInstr;

  /// An abstract interface for register coalescers.  Coalescers must
  /// implement this interface to be part of the coalescer analysis
  /// group.
  class RegisterCoalescer {
  public:
    static char ID; // Class identification, replacement for typeinfo
    RegisterCoalescer() {}
    virtual ~RegisterCoalescer();  // We want to be subclassed

    /// Run the coalescer on this function, providing interference
    /// data to query.  Return whether we removed any copies.
    virtual bool coalesceFunction(MachineFunction &mf,
                                  RegallocQuery &ifd) = 0;

    /// Reset state.  Can be used to allow a coalescer run by
    /// PassManager to be run again by the register allocator.
    virtual void reset(MachineFunction &mf) {};

    /// Register allocators must call this from their own
    /// getAnalysisUsage to cover the case where the coalescer is not
    /// a Pass in the proper sense and isn't managed by PassManager.
    /// PassManager needs to know which analyses to make available and
    /// which to invalidate when running the register allocator or any
    /// pass that might call coalescing.  The long-term solution is to
    /// allow hierarchies of PassManagers.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {};
  }; 

  /// An abstract interface for register allocators to interact with
  /// coalescers
  ///
  /// Example:
  ///
  /// This is simply an example of how to use the RegallocQuery
  /// interface.  It is not meant to be used in production.
  ///
  ///   class LinearScanRegallocQuery : public RegallocQuery {
  ///   private:
  ///     const LiveIntervals &li;
  ///
  ///   public:
  ///     LinearScanRegallocQuery(LiveIntervals &intervals) 
  ///         : li(intervals) {};
  ///
  ///     /// This is pretty slow and conservative, but since linear scan
  ///     /// allocation doesn't pre-compute interference information it's
  ///     /// the best we can do.  Coalescers are always free to ignore this
  ///     /// and implement their own discovery strategy.  See
  ///     /// SimpleRegisterCoalescing for an example.
  ///     void getInterferences(IntervalSet &interferences,
  ///                           const LiveInterval &a) const {
  ///       for(LiveIntervals::const_iterator iv = li.begin(),
  ///             ivend = li.end();
  ///           iv != ivend;
  ///           ++iv) {
  ///         if (interfere(a, iv->second)) {
  ///           interferences.insert(&iv->second);
  ///         }
  ///       }
  ///     };
  ///
  ///     /// This is *really* slow and stupid.  See above.
  ///     int getNumberOfInterferences(const LiveInterval &a) const {
  ///       IntervalSet intervals;
  ///       getInterferences(intervals, a);
  ///       return intervals.size();
  ///     };
  ///   };  
  ///
  ///   In the allocator:
  ///
  ///   RegisterCoalescer &coalescer = getAnalysis<RegisterCoalescer>();
  ///
  ///   // We don't reset the coalescer so if it's already been run this
  ///   // takes almost no time.
  ///   LinearScanRegallocQuery ifd(*li_);
  ///   coalescer.coalesceFunction(fn, ifd);
  ///
  class RegallocQuery {
  public:
    typedef SmallPtrSet<const LiveInterval *, 8> IntervalSet;

    virtual ~RegallocQuery() {};
    
    /// Return whether two live ranges interfere.
    virtual bool interfere(const LiveInterval &a,
                           const LiveInterval &b) const {
      // A naive test
      return a.overlaps(b);
    };

    /// Return the set of intervals that interfere with this one.
    virtual void getInterferences(IntervalSet &interferences,
                                  const LiveInterval &a) const = 0;

    /// This can often be cheaper than actually returning the
    /// interferences.
    virtual int getNumberOfInterferences(const LiveInterval &a) const = 0;

    /// Make any data structure updates necessary to reflect
    /// coalescing or other modifications.
    virtual void updateDataForMerge(const LiveInterval &a,
                                    const LiveInterval &b,
                                    const MachineInstr &copy) {};

    /// Allow the register allocator to communicate when it doesn't
    /// want a copy coalesced.  This may be due to assumptions made by
    /// the allocator about various invariants and so this question is
    /// a matter of legality, not performance.  Performance decisions
    /// about which copies to coalesce should be made by the
    /// coalescer.
    virtual bool isLegalToCoalesce(const MachineInstr &inst) const {
      return true;
    }
  };
}

// Because of the way .a files work, we must force the SimpleRC
// implementation to be pulled in if the RegisterCoalescing header is
// included.  Otherwise we run the risk of RegisterCoalescing being
// used, but the default implementation not being linked into the tool
// that uses it.
FORCE_DEFINING_FILE_TO_BE_LINKED(RegisterCoalescer)
FORCE_DEFINING_FILE_TO_BE_LINKED(SimpleRegisterCoalescing)

#endif
