//===-- llvm/Support/ExtensibleRTTI.h - ExtensibleRTTI support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
//
// Defines an extensible RTTI mechanism designed to work with Casting.h.
//
// Extensible RTTI differs from LLVM's primary RTTI mechanism (see
// llvm.org/docs/HowToSetUpLLVMStyleRTTI.html) by supporting open type
// hierarchies, where new types can be added from outside libraries without
// needing to change existing code. LLVM's primary RTTI mechanism should be
// preferred where possible, but where open hierarchies are needed this system
// can be used.
//
// The RTTIRoot class defines methods for comparing type ids. Implementations
// of these methods can be injected into new classes using the RTTIExtends
// class template.
//
// E.g.
//
//   @code{.cpp}
//   class MyBaseClass : public RTTIExtends<MyBaseClass, RTTIRoot> {
//   public:
//     static char ID;
//     virtual void foo() = 0;
//   };
//
//   class MyDerivedClass1 : public RTTIExtends<MyDerivedClass1, MyBaseClass> {
//   public:
//     static char ID;
//     void foo() override {}
//   };
//
//   class MyDerivedClass2 : public RTTIExtends<MyDerivedClass2, MyBaseClass> {
//   public:
//     static char ID;
//     void foo() override {}
//   };
//
//   char MyBaseClass::ID = 0;
//   char MyDerivedClass1::ID = 0;
//   char MyDerivedClass2:: ID = 0;
//
//   void fn() {
//     std::unique_ptr<MyBaseClass> B = llvm::make_unique<MyDerivedClass1>();
//     llvm::outs() << isa<MyBaseClass>(B) << "\n"; // Outputs "1".
//     llvm::outs() << isa<MyDerivedClass1>(B) << "\n"; // Outputs "1".
//     llvm::outs() << isa<MyDerivedClass2>(B) << "\n"; // Outputs "0'.
//   }
//
//   @endcode
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_EXTENSIBLERTTI_H
#define LLVM_SUPPORT_EXTENSIBLERTTI_H

namespace llvm {

/// Base class for the extensible RTTI hierarchy.
///
/// This class defines virtual methods, dynamicClassID and isA, that enable
/// type comparisons.
class RTTIRoot {
public:
  virtual ~RTTIRoot() = default;

  /// Returns the class ID for this type.
  static const void *classID() { return &ID; }

  /// Returns the class ID for the dynamic type of this RTTIRoot instance.
  virtual const void *dynamicClassID() const = 0;

  /// Returns true if this class's ID matches the given class ID.
  virtual bool isA(const void *const ClassID) const {
    return ClassID == classID();
  }

private:
  virtual void anchor();

  static char ID;
};

/// Inheritance utility for extensible RTTI.
///
/// Multiple inheritance is supported, but RTTIExtends only inherits
/// constructors from the first base class. All subsequent bases will be
/// default constructed. Virtual and non-public inheritance are not supported.
///
/// RTTIExtents uses CRTP so the first template argument to RTTIExtends is the
/// newly introduced type, and the *second and later* arguments are the parent
/// classes.
///
/// @code{.cpp}
/// class MyType : public RTTIExtends<MyType, RTTIRoot> {
/// public:
///   static char ID;
/// };
///
/// class MyDerivedType : public RTTIExtends<MyDerivedType, MyType> {
/// public:
///   static char ID;
/// };
///
/// class MyOtherType : public RTTIExtends<MyOtherType, MyType> {
/// public:
///   static char ID;
/// };
///
/// class MyMultipleInheritanceType
///   : public RTTIExtends<MyMultipleInheritanceType,
///                        MyDerivedType, MyOtherType> {
/// public:
///   static char ID;
/// };
///
/// @endcode
///
template <typename ThisT, typename ParentT, typename... ParentTs>
class RTTIExtends : public ParentT, public ParentTs... {
public:
  // Inherit constructors from the first Parent.
  using ParentT::ParentT;

  static const void *classID() { return &ThisT::ID; }

  const void *dynamicClassID() const override { return &ThisT::ID; }

  /// Check whether this instance is a subclass of QueryT.
  template <typename QueryT> bool isA() const { return isA(QueryT::classID()); }

  bool isA(const void *const ClassID) const override {
    return ClassID == classID() || ParentT::isA(ClassID) ||
           (ParentTs::isA(ClassID) || ...);
  }

  template <typename T> static bool classof(const T *R) {
    return R->template isA<ThisT>();
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_EXTENSIBLERTTI_H
