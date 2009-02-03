/*===-- semispace.c - Simple semi-space copying garbage collector ---------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|* 
|*===----------------------------------------------------------------------===*|
|* 
|* This garbage collector is an extremely simple copying collector.  It splits
|* the managed region of memory into two pieces: the current space to allocate
|* from, and the copying space.  When the portion being allocated from fills up,
|* a garbage collection cycle happens, which copies all live blocks to the other
|* half of the managed space.
|*
\*===----------------------------------------------------------------------===*/

#include "../GCInterface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* AllocPtr - This points to the next byte that is available for allocation.
 */
static char *AllocPtr;

/* AllocEnd - This points to the first byte not available for allocation.  When
 * AllocPtr passes this, we have run out of space.
 */
static char *AllocEnd;

/* CurSpace/OtherSpace - These pointers point to the two regions of memory that
 * we switch between.  The unallocated portion of the CurSpace is known to be
 * zero'd out, but the OtherSpace contains junk.
 */
static void *CurSpace, *OtherSpace;

/* SpaceSize - The size of each space. */
static unsigned SpaceSize;

/* llvm_gc_initialize - Allocate the two spaces that we plan to switch between.
 */
void llvm_gc_initialize(unsigned InitialHeapSize) {
  SpaceSize = InitialHeapSize/2;
  CurSpace = AllocPtr = calloc(1, SpaceSize);
  OtherSpace = malloc(SpaceSize);
  AllocEnd = AllocPtr + SpaceSize;
}

/* We always want to inline the fast path, but never want to inline the slow
 * path.
 */
void *llvm_gc_allocate(unsigned Size) __attribute__((always_inline));
static void* llvm_gc_alloc_slow(unsigned Size) __attribute__((noinline));

void *llvm_gc_allocate(unsigned Size) {
  char *OldAP = AllocPtr;
  char *NewEnd = OldAP+Size;
  if (NewEnd > AllocEnd)
    return llvm_gc_alloc_slow(Size);
  AllocPtr = NewEnd;
  return OldAP;
}

static void* llvm_gc_alloc_slow(unsigned Size) {
  llvm_gc_collect();
  if (AllocPtr+Size > AllocEnd) {
    fprintf(stderr, "Garbage collector ran out of memory "
            "allocating object of size: %d\n", Size);
    exit(1);
  }

  return llvm_gc_allocate(Size);
}


static void process_pointer(void **Root, void *Meta) {
  printf("process_root[0x%p] = 0x%p\n", (void*) Root, (void*) *Root);
}

void llvm_gc_collect() {
  // Clear out the space we will be copying into.
  // FIXME: This should do the copy, then clear out whatever space is left.
  memset(OtherSpace, 0, SpaceSize);

  printf("Garbage collecting!!\n");
  llvm_cg_walk_gcroots(process_pointer);
  abort();
}

/* We use no read/write barriers */
void *llvm_gc_read(void *ObjPtr, void **FieldPtr) { return *FieldPtr; }
void llvm_gc_write(void *V, void *ObjPtr, void **FieldPtr) { *FieldPtr = V; }


/*===----------------------------------------------------------------------===**
 * FIXME: This should be in a code-generator specific library, but for now this
 * will work for all code generators.
 */
typedef struct FrameMap FrameMap;
struct FrameMap {
  int32_t NumRoots; // Number of roots in stack frame.
  int32_t NumMeta;  // Number of metadata descriptors. May be < NumRoots.
  void *Meta[];     // May be absent for roots without metadata.
};

typedef struct StackEntry StackEntry;
struct StackEntry {
  StackEntry *Next;       // Caller's stack entry.
  const FrameMap *Map;    // Pointer to constant FrameMap.
  void *Roots[];          // Stack roots (in-place array).
};
StackEntry *llvm_gc_root_chain;

void llvm_cg_walk_gcroots(void (*FP)(void **Root, void *Meta)) {
  StackEntry *R;
  for (R = llvm_gc_root_chain; R; R = R->Next) {
    unsigned i, e;
    for (i = 0, e = R->Map->NumMeta; i != e; ++i)
      FP(&R->Roots[i], R->Map->Meta[i]);
    for (e = R->Map->NumRoots; i != e; ++i)
      FP(&R->Roots[i], NULL);
  }
}
/* END FIXME! */


