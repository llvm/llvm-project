/*===-- GCInterface.h - Public interface exposed by garbage collectors ----===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
|*===----------------------------------------------------------------------===*|
|*
|* This file defines the common public interface that must be exposed by all
|* LLVM garbage collectors.
|*
\*===----------------------------------------------------------------------===*/

#ifndef GCINTERFACE_H
#define GCINTERFACE_H

/* llvm_cg_walk_gcroots - This function is exposed by the LLVM code generator,
 * and allows us to traverse the roots on the stack.
 */
void llvm_cg_walk_gcroots(void (*FP)(void **Root, void *Meta));


/* llvm_gc_initialize - This function is called to initalize the garbage
 * collector.
 */
void llvm_gc_initialize(unsigned InitialHeapSize);

/* llvm_gc_allocate - This function allocates Size bytes from the heap and
 * returns a pointer to it.
 */
void *llvm_gc_allocate(unsigned Size);

/* llvm_gc_collect - This function forces a garbage collection cycle.
 */
void llvm_gc_collect();

/* llvm_gc_read - This function should be implemented to include any read
 * barrier code that is needed by the garbage collector.
 */
void *llvm_gc_read(void *ObjPtr, void **FieldPtr);

/* llvm_gc_write - This function should be implemented to include any write
 * barrier code that is needed by the garbage collector.
 */
void llvm_gc_write(void *V, void *ObjPtr, void **FieldPtr);

#endif
