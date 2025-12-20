#include "llvm/Transforms/Utils/PointerTypeTool/MyTy.h"

using namespace llvm;

int MyTy::floatBitWidth[7] = {16, 16, 32, 64, 80, 128, 128};

llvm::MyTy::MyTy() { typeId = MyTypeID::Unknown; }

string llvm::MyTy::toString() { return "void"; }

bool llvm::MyTy::update(shared_ptr<MyTy>) { return false; }

shared_ptr<MyTy> llvm::MyTy::getPointeeTyAsPtr() {
  assert(isArray() || isStruct() || isPointer());
  if (isPointer()) {
    return static_cast<MyPointerTy *>(this)->getPointeeTy();
  } else {
    return static_cast<MyArrayTy *>(this)->getElementTy();
  }
}

llvm::MyTy::MyTypeID llvm::MyTy::getTypeID() const { return typeId; }

void llvm::MyTy::setTypeID(MyTypeID t) { typeId = t; }

bool MyTy::isArray() const { return getTypeID() == MyTypeID::Array; }

bool MyTy::isBasic() const { return getTypeID() == MyTypeID::Basic; }

bool MyTy::isPointer() const { return getTypeID() == MyTypeID::Pointer; }

bool MyTy::isVoid() const { return getTypeID() == MyTypeID::Void; }

bool MyTy::isUnknown() const { return getTypeID() == MyTypeID::Unknown; }

bool MyTy::isStruct() const { return getTypeID() == MyTypeID::Struct; }

bool MyTy::isVector() const { return getTypeID() == MyTypeID::Vector; }

shared_ptr<MyTy> MyTy::getStructLCA(
    shared_ptr<MyTy> t1,
    shared_ptr<MyTy> t2) {
  return t1->toString() == t2->toString() ? t1 : make_shared<MyVoidTy>();
}

template <typename T, typename U>
shared_ptr<T> MyTy::ptr_cast(shared_ptr<U> u) {
  if (auto t = static_cast<T *>(u.get())) {
    return shared_ptr<T>(u, t);
  }
  return nullptr;
}

shared_ptr<MyTy> MyTy::int_with_int(Type* t1, Type* t2) {
  auto i1 = static_cast<IntegerType *>(t1);
  auto i2 = static_cast<IntegerType *>(t2);
  return make_shared<MyBasicTy>(i1->getBitWidth() > i2->getBitWidth() ? i1 : i2);
}

shared_ptr<MyTy> MyTy::float_with_float(Type *t1, Type *t2) {
  return make_shared<MyBasicTy>(
      floatBitWidth[t1->getTypeID()] > floatBitWidth[t2->getTypeID()] ? t1 : t2);
}

shared_ptr<MyTy> MyTy::basic_with_basic(
    shared_ptr<MyTy> t1,
    shared_ptr<MyTy> t2) {
  auto b1 = ptr_cast<MyBasicTy>(t1)->getBasic();
  auto b2 = ptr_cast<MyBasicTy>(t2)->getBasic();
  if (b1->isIntegerTy()) {
    if (b2->isIntegerTy()) {
      return int_with_int(b1, b2);
    } else {
      return make_shared<MyVoidTy>();
    }
  } else {
    if (b2->isIntegerTy()) {
      return make_shared<MyVoidTy>();
    } else {
      return float_with_float(b1, b2);
    }
  }
  return make_shared<MyTy>();
}

shared_ptr<MyTy> MyTy::ptr_with_array(
    shared_ptr<MyTy> t1,
    shared_ptr<MyTy> t2) {
  auto pt1 = ptr_cast<MyPointerTy>(t1);
  auto pt2 = ptr_cast<MyArrayTy>(t2);
  if (pt2->getElementTy()->compatibleWith(pt1)) {
    pt2->update(pt1);
    return pt2;
  } else {
    return make_shared<MyVoidTy>();
  }
}

bool MyTy::compatibleWith(shared_ptr<MyTy> ty) {
  if (ty->isUnknown() || this->isUnknown()) {
    return true;
  } else if (this->getTypeID() != ty->getTypeID()) {
    return false;
  } else {
    if (this->isPointer()) {
      return static_cast<MyPointerTy *>(this)->getPointeeTy()->compatibleWith(
          ptr_cast<MyPointerTy>(ty)->getPointeeTy());
    } else if (this->isArray()) {
      return static_cast<MyArrayTy *>(this)->getElementTy()->compatibleWith(
          ptr_cast<MyArrayTy>(ty)->getElementTy());
    } else if (this->isBasic()) {
      auto b1 = static_cast<MyBasicTy *>(this)->getBasic();
      auto b2 = ptr_cast<MyBasicTy>(ty)->getBasic();
      if (b1->isIntegerTy()) {
        if (b2->isIntegerTy()) {
          auto i1 = static_cast<IntegerType *>(b1);
          auto i2 = static_cast<IntegerType *>(b2);
          return i1->getBitWidth() >= i2->getBitWidth();
        } else {
          return false;
        }
      } else {
        if (b2->isIntegerTy()) {
          return false;
        } else {
          return floatBitWidth[this->getTypeID()] >=
                 floatBitWidth[ty->getTypeID()];
        }
      }
      return false;
    } else if (this->isStruct()) {
      return static_cast<MyStructTy *>(this)->toString() == ptr_cast<MyStructTy>(ty)->toString();
    } else {
      return false;
    }
  }
}

shared_ptr<MyTy> llvm::MyTy::leastCompatibleType(
    shared_ptr<MyTy> t1,
    shared_ptr<MyTy> t2) {
  shared_ptr<MyTy> ret;
  if (t1->isUnknown()) {
    ret = t2;
  } else if (t1->isVoid()) {
    ret = t1;
  } else if (t1->isBasic()) {
    if (t2->isUnknown()) {
      ret = t1;
    } else if (t2->isVoid()) {
      ret = t2;
    } else if (t2->isBasic()) {
      ret = basic_with_basic(t1, t2);
    } else {
      ret = make_shared<MyVoidTy>();
    }
  } else if (t1->isPointer()) {
    if (t2->isUnknown()) {
      ret = t1;
    } else if (t2->isVoid()) {
      ret = t2;
    } else if (t2->isPointer()) {
      auto pt1 = ptr_cast<MyPointerTy>(t1);
      auto pt2 = ptr_cast<MyPointerTy>(t2);
      ret = make_shared<MyPointerTy>(
          leastCompatibleType(pt1->getPointeeTy(), pt2->getPointeeTy()));
    } else if (t2->isArray()) {
      ret = ptr_with_array(t1, t2);
    } else {
      ret = make_shared<MyVoidTy>();
    }
  } else if (t1->isArray()) {
    if (t2->isUnknown()) {
      ret = t1;
    } else if (t2->isVoid()) {
      ret = t2;
    } else if (t2->isPointer()) {
      ret = ptr_with_array(t2, t1);
    } else if (t2->isArray()) {
      auto pt1 = ptr_cast<MyArrayTy>(t1);
      auto pt2 = ptr_cast<MyArrayTy>(t2);
      ret = make_shared<MyArrayTy>(
          leastCompatibleType(pt1->getElementTy(), pt2->getElementTy()),
          std::max(pt1->getElementCnt(), pt2->getElementCnt()));
    } else {
      ret = make_shared<MyVoidTy>();
    }
  } else if (t1->isStruct()) {
    if (t2->isUnknown()) {
      ret = t1;
    } else if (t2->isVoid()) {
      ret = t2;
    } else if (t2->isStruct()) {
      ret = getStructLCA(t1, t2);
    }
  } else {
    ret = make_shared<MyVoidTy>();
  }
  return ret;
}

MyVoidTy::MyVoidTy() { setTypeID(MyTypeID::Void); }

string MyVoidTy::toString() { return "void"; }

MyPointerTy::MyPointerTy(shared_ptr<MyTy> pointee) {
  pointeeTy = pointee;
  setTypeID(MyTypeID::Pointer);
}

shared_ptr<MyTy> MyPointerTy::getPointeeTy() { return pointeeTy; }

bool MyPointerTy::update(shared_ptr<MyTy> pointee) {
  string last = this->toString();
  if (pointeeTy->isArray()) {
    auto at = ptr_cast<MyArrayTy>(pointeeTy);
    if (at->getElementTy()->compatibleWith(pointee)) {
      at->update(pointee);
    } else {
      pointeeTy = MyTy::leastCompatibleType(pointeeTy, pointee);
    }
  } else if (pointeeTy->isStruct()) {
    auto st = ptr_cast<MyStructTy>(pointeeTy);
    if (st->getElementTy()->compatibleWith(pointee)) {
      st->updateElement(MyTy::leastCompatibleType(st->getElementTy(), pointee));
    } else {
      pointeeTy = MyTy::leastCompatibleType(pointeeTy, pointee);
    }
  } else  {
    pointeeTy = MyTy::leastCompatibleType(pointeeTy, pointee);
  }
  return last != this->toString();
}

string MyPointerTy::toString() { return pointeeTy->toString() + "*"; }

MyBasicTy::MyBasicTy(Type *basic) {
  basicTy = basic;
  setTypeID(MyTypeID::Basic);
}

Type *MyBasicTy::getBasic() { return basicTy; }

string MyBasicTy::toString() {
  switch (basicTy->getTypeID()) {
  case Type::IntegerTyID: {
    auto intTy = static_cast<IntegerType *>(basicTy);
    return "i" + std::to_string(intTy->getBitWidth());
  }
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::HalfTyID:
    return "half";
  case Type::BFloatTyID:
    return "bfloat";
  case Type::X86_FP80TyID:
    return "x86_fp80";
  case Type::FP128TyID:
    return "fp128";
  case Type::PPC_FP128TyID:
    return "ppc_fp128";
  case Type::LabelTyID:
    return "label";
  default:
    return "not_added";
  }
}

MyArrayTy::MyArrayTy(shared_ptr<MyTy> eTy, int eCnt) {
  elementCnt = eCnt;
  elementTy = eTy;
  setTypeID(MyTypeID::Array);
}

shared_ptr<MyTy> MyArrayTy::getElementTy() { return elementTy; }

int MyArrayTy::getElementCnt() const { return elementCnt; }

string MyArrayTy::toString() {
  return "[" + std::to_string(elementCnt) + " x " + elementTy->toString() +
         "]";
}

bool MyArrayTy::update(shared_ptr<MyTy> pointee) {
  string last = this->toString();
  elementTy = MyTy::leastCompatibleType(elementTy, pointee);
  return last != this->toString();
}

MyVectorTy::MyVectorTy(shared_ptr<MyTy> eTy, int eCnt, bool fixed) {
  setTypeID(MyTypeID::Vector);
  elementTy = eTy;
  elementCnt = eCnt;
  this->fixed = fixed;
}

shared_ptr<MyTy> MyVectorTy::getElementTy() { return elementTy; }

int MyVectorTy::getElementCnt() const { return elementCnt; }

string MyVectorTy::toString() {
  string scale = fixed ? "" : "vscale x ";
  return "<" + scale + std::to_string(elementCnt) + " x " +
                           elementTy->toString() + ">";
}

bool MyVectorTy::update(shared_ptr<MyTy> pointee) {
  string last = this->toString();
  elementTy = MyTy::leastCompatibleType(elementTy, pointee);
  return last != this->toString();
}

MyStructTy::MyStructTy(string name, SmallVector<shared_ptr<MyTy>> vec, bool opaque) {
  setTypeID(MyTypeID::Struct);
  this->name = name;
  elementTy = vec;
  this->opaque = opaque;
}

bool MyStructTy::isOpaque() const { return opaque; }

shared_ptr<MyTy> MyStructTy::getElementTy(int index) {
  assert(getElementCnt() > index);                  
  return elementTy[index];
}

int MyStructTy::getElementCnt() { return elementTy.size(); }

bool MyStructTy::hasName() const { return name != ""; }

bool MyStructTy::update(shared_ptr<MyTy> pointee) { return updateElement(pointee); }

bool MyStructTy::updateElement(shared_ptr<MyTy> ty, int index) {
  string last = this->toString();
  elementTy[index] = leastCompatibleType(elementTy[index], ty);
  return last != this->toString();
}

string MyStructTy::toString() {
  if (hasName()) {
    return "%" + name;
  } else {
    string ret = "{";
    for (auto i = 0; i < getElementCnt(); i++) {
      if (i != 0) {
        ret = ret + ",";
      }
      ret = ret + " " + elementTy[i]->toString();
      if (i == getElementCnt() - 1) {
        ret = ret + " ";
      }
    }
    ret = ret + "}";
    return ret;
  }
}

template shared_ptr<MyPointerTy>
MyTy::ptr_cast<MyPointerTy, MyTy>(shared_ptr<MyTy>);

template shared_ptr<MyArrayTy>
MyTy::ptr_cast<MyArrayTy, MyTy>(shared_ptr<MyTy>);

template shared_ptr<MyBasicTy>
MyTy::ptr_cast<MyBasicTy, MyTy>(shared_ptr<MyTy>);

template shared_ptr<MyStructTy>
MyTy::ptr_cast<MyStructTy, MyTy>(shared_ptr<MyTy>);

template shared_ptr<MyVectorTy>
MyTy::ptr_cast<MyVectorTy, MyTy>(shared_ptr<MyTy>);
