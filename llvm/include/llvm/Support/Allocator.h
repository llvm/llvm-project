//===--- Allocator.h - Simple memory allocation abstraction -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MallocAllocator and BumpPtrAllocator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALLOCATOR_H
#define LLVM_SUPPORT_ALLOCATOR_H

#include "llvm/Support/AlignOf.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/DataTypes.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstddef>

namespace llvm {
template <typename T> struct ReferenceAdder { typedef T& result; };
template <typename T> struct ReferenceAdder<T&> { typedef T result; };

class MallocAllocator {
public:
  MallocAllocator() {}
  ~MallocAllocator() {}

  void Reset() {}

  void *Allocate(size_t Size, size_t /*Alignment*/) { return malloc(Size); }

  template <typename T>
  T *Allocate() { return static_cast<T*>(malloc(sizeof(T))); }

  template <typename T>
  T *Allocate(size_t Num) {
    return static_cast<T*>(malloc(sizeof(T)*Num));
  }

  void Deallocate(const void *Ptr) { free(const_cast<void*>(Ptr)); }

  void PrintStats() const {}
};

/// MemSlab - This structure lives at the beginning of every slab allocated by
/// the bump allocator.
class MemSlab {
public:
  size_t Size;
  MemSlab *NextPtr;
};

/// SlabAllocator - This class can be used to parameterize the underlying
/// allocation strategy for the bump allocator.  In particular, this is used
/// by the JIT to allocate contiguous swathes of executable memory.  The
/// interface uses MemSlab's instead of void *'s so that the allocator
/// doesn't have to remember the size of the pointer it allocated.
class SlabAllocator {
public:
  virtual ~SlabAllocator();
  virtual MemSlab *Allocate(size_t Size) = 0;
  virtual void Deallocate(MemSlab *Slab) = 0;
};

/// MallocSlabAllocator - The default slab allocator for the bump allocator
/// is an adapter class for MallocAllocator that just forwards the method
/// calls and translates the arguments.
class MallocSlabAllocator : public SlabAllocator {
  /// Allocator - The underlying allocator that we forward to.
  ///
  MallocAllocator Allocator;

public:
  MallocSlabAllocator() : Allocator() { }
  virtual ~MallocSlabAllocator();
  virtual MemSlab *Allocate(size_t Size) LLVM_OVERRIDE;
  virtual void Deallocate(MemSlab *Slab) LLVM_OVERRIDE;
};

/// BumpPtrAllocator - This allocator is useful for containers that need
/// very simple memory allocation strategies.  In particular, this just keeps
/// allocating memory, and never deletes it until the entire block is dead. This
/// makes allocation speedy, but must only be used when the trade-off is ok.
class BumpPtrAllocator {
  BumpPtrAllocator(const BumpPtrAllocator &) LLVM_DELETED_FUNCTION;
  void operator=(const BumpPtrAllocator &) LLVM_DELETED_FUNCTION;

  /// SlabSize - Allocate data into slabs of this size unless we get an
  /// allocation above SizeThreshold.
  size_t SlabSize;

  /// SizeThreshold - For any allocation larger than this threshold, we should
  /// allocate a separate slab.
  size_t SizeThreshold;

  /// Allocator - The underlying allocator we use to get slabs of memory.  This
  /// defaults to MallocSlabAllocator, which wraps malloc, but it could be
  /// changed to use a custom allocator.
  SlabAllocator &Allocator;

  /// CurSlab - The slab that we are currently allocating into.
  ///
  MemSlab *CurSlab;

  /// CurPtr - The current pointer into the current slab.  This points to the
  /// next free byte in the slab.
  char *CurPtr;

  /// End - The end of the current slab.
  ///
  char *End;

  /// BytesAllocated - This field tracks how many bytes we've allocated, so
  /// that we can compute how much space was wasted.
  size_t BytesAllocated;

  /// AlignPtr - Align Ptr to Alignment bytes, rounding up.  Alignment should
  /// be a power of two.  This method rounds up, so AlignPtr(7, 4) == 8 and
  /// AlignPtr(8, 4) == 8.
  static char *AlignPtr(char *Ptr, size_t Alignment);

  /// StartNewSlab - Allocate a new slab and move the bump pointers over into
  /// the new slab.  Modifies CurPtr and End.
  void StartNewSlab();

  /// DeallocateSlabs - Deallocate all memory slabs after and including this
  /// one.
  void DeallocateSlabs(MemSlab *Slab);

  static MallocSlabAllocator DefaultSlabAllocator;

  template<typename T> friend class SpecificBumpPtrAllocator;
public:
  BumpPtrAllocator(size_t size = 4096, size_t threshold = 4096,
                   SlabAllocator &allocator = DefaultSlabAllocator);
  ~BumpPtrAllocator();

  /// Reset - Deallocate all but the current slab and reset the current pointer
  /// to the beginning of it, freeing all memory allocated so far.
  void Reset();

  /// Allocate - Allocate space at the specified alignment.
  ///
  void *Allocate(size_t Size, size_t Alignment);

  /// Allocate space, but do not construct, one object.
  ///
  template <typename T>
  T *Allocate() {
    return static_cast<T*>(Allocate(sizeof(T),AlignOf<T>::Alignment));
  }

  /// Allocate space for an array of objects.  This does not construct the
  /// objects though.
  template <typename T>
  T *Allocate(size_t Num) {
    return static_cast<T*>(Allocate(Num * sizeof(T), AlignOf<T>::Alignment));
  }

  /// Allocate space for a specific count of elements and with a specified
  /// alignment.
  template <typename T>
  T *Allocate(size_t Num, size_t Alignment) {
    // Round EltSize up to the specified alignment.
    size_t EltSize = (sizeof(T)+Alignment-1)&(-Alignment);
    return static_cast<T*>(Allocate(Num * EltSize, Alignment));
  }

  void Deallocate(const void * /*Ptr*/) {}

  unsigned GetNumSlabs() const;

  void PrintStats() const;
  
  /// Compute the total physical memory allocated by this allocator.
  size_t getTotalMemory() const;
};

/// SpecificBumpPtrAllocator - Same as BumpPtrAllocator but allows only
/// elements of one type to be allocated. This allows calling the destructor
/// in DestroyAll() and when the allocator is destroyed.
template <typename T>
class SpecificBumpPtrAllocator {
  BumpPtrAllocator Allocator;
public:
  SpecificBumpPtrAllocator(size_t size = 4096, size_t threshold = 4096,
              SlabAllocator &allocator = BumpPtrAllocator::DefaultSlabAllocator)
    : Allocator(size, threshold, allocator) {}

  ~SpecificBumpPtrAllocator() {
    DestroyAll();
  }

  /// Call the destructor of each allocated object and deallocate all but the
  /// current slab and reset the current pointer to the beginning of it, freeing
  /// all memory allocated so far.
  void DestroyAll() {
    MemSlab *Slab = Allocator.CurSlab;
    while (Slab) {
      char *End = Slab == Allocator.CurSlab ? Allocator.CurPtr :
                                              (char *)Slab + Slab->Size;
      for (char *Ptr = (char*)(Slab+1); Ptr < End; Ptr += sizeof(T)) {
        Ptr = Allocator.AlignPtr(Ptr, alignOf<T>());
        if (Ptr + sizeof(T) <= End)
          reinterpret_cast<T*>(Ptr)->~T();
      }
      Slab = Slab->NextPtr;
    }
    Allocator.Reset();
  }

  /// Allocate space for a specific count of elements.
  T *Allocate(size_t num = 1) {
    return Allocator.Allocate<T>(num);
  }
};

}  // end namespace llvm

inline void *operator new(size_t Size, llvm::BumpPtrAllocator &Allocator) {
  struct S {
    char c;
    union {
      double D;
      long double LD;
      long long L;
      void *P;
    } x;
  };
  return Allocator.Allocate(Size, std::min((size_t)llvm::NextPowerOf2(Size),
                                           offsetof(S, x)));
}

inline void operator delete(void *, llvm::BumpPtrAllocator &) {}

#endif // LLVM_SUPPORT_ALLOCATOR_H
