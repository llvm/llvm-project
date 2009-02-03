//===-- llvm/Support/Annotation.h - Annotation classes ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for two classes: Annotation & Annotable.
// Using these two simple classes, anything that derives from Annotable can have
// Annotation subclasses attached to them, ready for easy retrieval.
//
// Annotations are designed to be easily attachable to various classes.
//
// The AnnotationManager class is essential for using these classes.  It is
// responsible for turning Annotation name strings into tokens [unique id #'s]
// that may be used to search for and create annotations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ANNOTATION_H
#define LLVM_SUPPORT_ANNOTATION_H

#include <cassert>

namespace llvm {

class AnnotationID;
class Annotation;
class Annotable;
struct AnnotationManager;

//===----------------------------------------------------------------------===//
//
// AnnotationID - This class is a thin wrapper around an unsigned integer that
// is used to hopefully prevent errors using AnnotationID's.  They may be copied
// freely around and passed byvalue with little or no overhead.
//
class AnnotationID {
  friend struct AnnotationManager;
  unsigned ID;

  AnnotationID();                             // Default ctor is disabled

  // AnnotationID is only creatable from AnnMgr.
  explicit inline AnnotationID(unsigned i) : ID(i) {}
public:
  inline AnnotationID(const AnnotationID &A) : ID(A.ID) {}

  inline bool operator==(const AnnotationID &A) const {
    return A.ID == ID;
  }
  inline bool operator<(const AnnotationID &A) const {
    return ID < A.ID;
  }
};


//===----------------------------------------------------------------------===//
//
// Annotation Class - This class serves as a base class for any specific
// annotations that you might need.  Simply subclass this to add extra
// information to the annotations.
//
class Annotation {
  friend class Annotable;  // Annotable manipulates Next list
  AnnotationID ID;         // ID number, as obtained from AnnotationManager
  Annotation *Next;        // The next annotation in the linked list
public:
  explicit inline Annotation(AnnotationID id) : ID(id), Next(0) {}
  virtual ~Annotation();  // Designed to be subclassed

  // getID - Return the unique ID# of this annotation
  inline AnnotationID getID() const { return ID; }

  // getNext - Return the next annotation in the list...
  inline Annotation *getNext() const { return Next; }
};


//===----------------------------------------------------------------------===//
//
// Annotable - This class is used as a base class for all objects that would
// like to have annotation capability.
//
// Annotable objects keep their annotation list sorted as annotations are
// inserted and deleted.  This is used to ensure that annotations with identical
// ID#'s are stored sequentially.
//
class Annotable {
  mutable Annotation *AnnotationList;

  Annotable(const Annotable &);        // Do not implement
  void operator=(const Annotable &);   // Do not implement
public:
  Annotable() : AnnotationList(0) {}
  ~Annotable();

  // getAnnotation - Search the list for annotations of the specified ID.  The
  // pointer returned is either null (if no annotations of the specified ID
  // exist), or it points to the first element of a potentially list of elements
  // with identical ID #'s.
  //
  Annotation *getAnnotation(AnnotationID ID) const {
    for (Annotation *A = AnnotationList; A; A = A->getNext())
      if (A->getID() == ID) return A;
    return 0;
  }

  // getOrCreateAnnotation - Search through the annotation list, if there is
  // no annotation with the specified ID, then use the AnnotationManager to
  // create one.
  //
  inline Annotation *getOrCreateAnnotation(AnnotationID ID) const;

  // addAnnotation - Insert the annotation into the list in a sorted location.
  //
  void addAnnotation(Annotation *A) const {
    assert(A->Next == 0 && "Annotation already in list?!?");

    Annotation **AL = &AnnotationList;
    while (*AL && (*AL)->ID < A->getID())  // Find where to insert annotation
      AL = &((*AL)->Next);
    A->Next = *AL;                         // Link the annotation in
    *AL = A;
  }

  // unlinkAnnotation - Remove the first annotation of the specified ID... and
  // then return the unlinked annotation.  The annotation object is not deleted.
  //
  inline Annotation *unlinkAnnotation(AnnotationID ID) const {
    for (Annotation **A = &AnnotationList; *A; A = &((*A)->Next))
      if ((*A)->getID() == ID) {
        Annotation *Ret = *A;
        *A = Ret->Next;
        Ret->Next = 0;
        return Ret;
      }
    return 0;
  }

  // deleteAnnotation - Delete the first annotation of the specified ID in the
  // list.  Unlink unlinkAnnotation, this actually deletes the annotation object
  //
  bool deleteAnnotation(AnnotationID ID) const {
    Annotation *A = unlinkAnnotation(ID);
    delete A;
    return A != 0;
  }
};


//===----------------------------------------------------------------------===//
//
// AnnotationManager - This class is primarily responsible for maintaining a
// one-to-one mapping between string Annotation names and Annotation ID numbers.
//
// Compared to the rest of the Annotation system, these mapping methods are
// relatively slow, so they should be avoided by locally caching Annotation
// ID #'s.  These methods are safe to call at any time, even by static ctors, so
// they should be used by static ctors most of the time.
//
// This class also provides support for annotations that are created on demand
// by the Annotable::getOrCreateAnnotation method.  To get this to work, simply
// register an annotation handler
//
struct AnnotationManager {
  typedef Annotation *(*Factory)(AnnotationID, const Annotable *, void*);

  //===--------------------------------------------------------------------===//
  // Basic ID <-> Name map functionality

  static AnnotationID  getID(const char *Name);  // Name -> ID
  static const char *getName(AnnotationID ID);   // ID -> Name

  // getID - Name -> ID + registration of a factory function for demand driven
  // annotation support.
  static AnnotationID getID(const char *Name, Factory Fact, void *Data = 0);

  //===--------------------------------------------------------------------===//
  // Annotation creation on demand support...

  // registerAnnotationFactory - This method is used to register a callback
  // function used to create an annotation on demand if it is needed by the
  // Annotable::getOrCreateAnnotation method.
  //
  static void registerAnnotationFactory(AnnotationID ID, Factory Func,
                                        void *ExtraData = 0);

  // createAnnotation - Create an annotation of the specified ID for the
  // specified object, using a register annotation creation function.
  //
  static Annotation *createAnnotation(AnnotationID ID, const Annotable *Obj);
};



// getOrCreateAnnotation - Search through the annotation list, if there is
// no annotation with the specified ID, then use the AnnotationManager to
// create one.
//
inline Annotation *Annotable::getOrCreateAnnotation(AnnotationID ID) const {
  Annotation *A = getAnnotation(ID);   // Fast path, check for preexisting ann
  if (A) return A;

  // No annotation found, ask the annotation manager to create an annotation...
  A = AnnotationManager::createAnnotation(ID, this);
  assert(A && "AnnotationManager could not create annotation!");
  addAnnotation(A);
  return A;
}

} // End namespace llvm

#endif
