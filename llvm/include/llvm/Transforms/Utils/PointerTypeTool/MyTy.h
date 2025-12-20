#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_MYTY_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_MYTY_H

#include <string>
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace llvm {

class MyTy {
public:
  enum MyTypeID { Basic, Pointer, Unknown, Array, Struct, Vector, Void };
  MyTy();
  virtual string toString();
  virtual bool update(shared_ptr<MyTy>);
  shared_ptr<MyTy> getPointeeTyAsPtr();
  MyTypeID getTypeID() const;
  void setTypeID(MyTypeID);
  bool isBasic() const;
  bool isPointer() const;
  bool isVoid() const;
  bool isArray() const;
  bool isUnknown() const;
  bool isStruct() const;
  bool isVector() const;
  bool compatibleWith(shared_ptr<MyTy>);
  static shared_ptr<MyTy> leastCompatibleType(shared_ptr<MyTy>,
                                              shared_ptr<MyTy>);
  static shared_ptr<MyTy> getStructLCA(shared_ptr<MyTy>, shared_ptr<MyTy>);
  template <typename T, typename U>
  static shared_ptr<T> ptr_cast(shared_ptr<U>);
protected:
  MyTypeID typeId;
  static int floatBitWidth[7];
  static shared_ptr<MyTy> basic_with_basic(shared_ptr<MyTy>, shared_ptr<MyTy>);
  static shared_ptr<MyTy> ptr_with_array(shared_ptr<MyTy>, shared_ptr<MyTy>);
  static shared_ptr<MyTy> int_with_int(Type *, Type *);
  static shared_ptr<MyTy> float_with_float(Type *, Type *);
};

class MyVoidTy : public MyTy {
public:
  MyVoidTy();
  string toString() override;
};

class MyPointerTy : public MyTy {
  shared_ptr<MyTy> pointeeTy;

public:
  MyPointerTy(shared_ptr<MyTy>);
  shared_ptr<MyTy> getPointeeTy();
  string toString() override;
  bool update(shared_ptr<MyTy>) override;
};

class MyBasicTy : public MyTy {
  Type *basicTy;

public:
  MyBasicTy(Type *basic);
  Type *getBasic();
  string toString() override;
};

class MyArrayTy : public MyTy {
  int elementCnt;
  shared_ptr<MyTy> elementTy;

public:
  MyArrayTy(shared_ptr<MyTy> eTy, int eCnt);
  shared_ptr<MyTy> getElementTy();
  int getElementCnt() const;
  string toString() override;
  bool update(shared_ptr<MyTy> pointee) override;
};

class MyVectorTy : public MyTy {
  int elementCnt;
  shared_ptr<MyTy> elementTy;
  bool fixed;

public:
  MyVectorTy(shared_ptr<MyTy> eTy, int eCnt, bool fixed);
  shared_ptr<MyTy> getElementTy();
  int getElementCnt() const;
  string toString() override;
  bool update(shared_ptr<MyTy> pointee) override;
};

class MyStructTy : public MyTy {
  SmallVector<shared_ptr<MyTy>> elementTy;
  string name;
  bool opaque;

public:
  MyStructTy(string name, SmallVector<shared_ptr<MyTy>> vec, bool opaque);
  shared_ptr<MyTy> getElementTy(int index = 0);
  string toString() override;
  bool hasName() const;
  bool isOpaque() const;
  int getElementCnt();
  bool update(shared_ptr<MyTy> pointee) override;
  bool updateElement(shared_ptr<MyTy> ty, int index = 0);
};

}
#endif