#include "llvm/Transforms/Utils/MyTy.h"

using namespace llvm;

int MyTy::floatBitWidth[7] = {16, 16, 32, 64, 80, 128, 128};

llvm::MyTy::MyTy() { typeId = MyTypeID::Unknown; }

std::string llvm::MyTy::to_string() { return "unknown"; }

void llvm::MyTy::update(std::shared_ptr<MyTy>) { return; }

llvm::MyTy::MyTypeID llvm::MyTy::getTypeID() { return typeId; }

void llvm::MyTy::setTypeID(MyTypeID t) { typeId = t; }

std::shared_ptr<MyTy> MyTy::from(Type *type) {
  switch (type->getTypeID()) {
  case Type::IntegerTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
    return std::make_shared<MyBasicTy>(type);
  case Type::ArrayTyID:
    return std::make_shared<MyArrayTy>(type);
  case Type::PointerTyID:
    return std::make_shared<MyPointerTy>(std::make_shared<MyTy>());
  case Type::StructTyID:
    return std::make_shared<MyStructTy>(type);
  default:
    return nullptr;
  }
}

bool MyTy::isArray() { return getTypeID() == MyTypeID::Array; }

bool MyTy::isBasic() { return getTypeID() == MyTypeID::Basic; }

bool MyTy::isPointer() { return getTypeID() == MyTypeID::Pointer; }

bool MyTy::isVoid() { return getTypeID() == MyTypeID::Void; }

bool MyTy::isUnknown() { return getTypeID() == MyTypeID::Unknown; }

bool MyTy::isStruct() { return getTypeID() == MyTypeID::Struct; }

std::shared_ptr<MyTy> MyTy::getStructLCA(
    std::shared_ptr<MyTy> t1,
    std::shared_ptr<MyTy> t2) {
  return t1->to_string() == t2->to_string() ? t1 : std::make_shared<MyVoidTy>();
}

template <typename T, typename U>
std::shared_ptr<T> MyTy::ptr_cast(std::shared_ptr<U> u) {
  if (T *t = cast<T>(u.get())) {
    return std::shared_ptr<T>(u, t);
  }
  return nullptr;
}

std::shared_ptr<MyTy> MyTy::int_with_int(Type* t1, Type* t2) {
  auto i1 = cast<IntegerType>(t1);
  auto i2 = cast<IntegerType>(t2);
  return std::make_shared<MyBasicTy>(i1->getBitWidth() > i2->getBitWidth() ? i1 : i2);
}

std::shared_ptr<MyTy> MyTy::float_with_float(Type *t1, Type *t2) {
  return std::make_shared<MyBasicTy>(
      floatBitWidth[t1->getTypeID()] > floatBitWidth[t2->getTypeID()] ? t1 : t2);
}

std::shared_ptr<MyTy> MyTy::basic_with_basic(
    std::shared_ptr<MyTy> t1,
    std::shared_ptr<MyTy> t2) {
  auto b1 = ptr_cast<MyBasicTy>(t1)->getBasic();
  auto b2 = ptr_cast<MyBasicTy>(t2)->getBasic();
  if (b1->isIntegerTy()) {
    if (b2->isIntegerTy()) {
      return int_with_int(b1, b2);
    } else {
      return std::make_shared<MyVoidTy>();
    }
  } else {
    if (b2->isIntegerTy()) {
      return std::make_shared<MyVoidTy>();
    } else {
      return float_with_float(b1, b2);
    }
  }
  return std::make_shared<MyTy>();
}

std::shared_ptr<MyTy> MyTy::ptr_with_array(
    std::shared_ptr<MyTy> t1,
    std::shared_ptr<MyTy> t2) {
  auto pt1 = ptr_cast<MyPointerTy>(t1);
  auto pt2 = ptr_cast<MyArrayTy>(t2);
  if (pt2->getElementTy()->compatibleWith(pt1->getInner())) {
    return pt2;
  } else {
    return std::make_shared<MyVoidTy>();
  }
}

bool MyTy::compatibleWith(std::shared_ptr<MyTy> ty) {
  errs() << "Check " << this->to_string() << " and " << ty->to_string() << "\n";
  if (ty->isUnknown()) {
    return true;
  } else if (this->getTypeID() != ty->getTypeID()) {
    return false;
  } else {
    if (this->isPointer()) {
      return cast<MyPointerTy>(this)->getInner()->compatibleWith(
          ptr_cast<MyPointerTy>(ty)->getInner());
    } else if (this->isArray()) {
      return cast<MyArrayTy>(this)->getElementTy()->compatibleWith(
          ptr_cast<MyArrayTy>(ty)->getElementTy());
    } else if (this->isBasic()) {
      auto b1 = cast<MyBasicTy>(this)->getBasic();
      auto b2 = ptr_cast<MyBasicTy>(ty)->getBasic();
      if (b1->isIntegerTy()) {
        if (b2->isIntegerTy()) {
          auto i1 = cast<IntegerType>(b1);
          auto i2 = cast<IntegerType>(b2);
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
      return cast<MyStructTy>(this)->to_string() == ptr_cast<MyStructTy>(ty)->to_string();
    } else {
      return false;
    }
  }
}

std::shared_ptr<MyTy> llvm::MyTy::leastCompatibleType(
    std::shared_ptr<MyTy> t1,
    std::shared_ptr<MyTy> t2) {
  std::shared_ptr<MyTy> ret;
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
      ret = std::make_shared<MyVoidTy>();
    }
  } else if (t1->isPointer()) {
    if (t2->isUnknown()) {
      ret = t1;
    } else if (t2->isVoid()) {
      ret = t2;
    } else if (t2->isPointer()) {
      auto pt1 = ptr_cast<MyPointerTy>(t1);
      auto pt2 = ptr_cast<MyPointerTy>(t2);
      ret = std::make_shared<MyPointerTy>(
          leastCompatibleType(pt1->getInner(), pt2->getInner()));
    } else if (t2->isArray()) {
      ret = ptr_with_array(t1, t2);
    } else {
      ret = std::make_shared<MyVoidTy>();
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
      ret = std::make_shared<MyArrayTy>(
          leastCompatibleType(pt1->getElementTy(), pt2->getElementTy()),
          std::max(pt1->getElementCnt(), pt2->getElementCnt()));
    } else {
      ret = std::make_shared<MyVoidTy>();
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
    ret = std::make_shared<MyVoidTy>();
  }
  /*errs() << t1->to_string() << " and " << t2->to_string() << " common is "
         << ret->to_string() << "\n";*/
  return ret;
}

MyVoidTy::MyVoidTy() { setTypeID(MyTypeID::Void); }

std::string MyVoidTy::to_string() { return "void"; }

MyPointerTy::MyPointerTy(std::shared_ptr<MyTy> inner) {
  innerTy = inner;
  setTypeID(MyTypeID::Pointer);
}

std::shared_ptr<MyTy> MyPointerTy::getInner() { return innerTy; }

void MyPointerTy::update(std::shared_ptr<MyTy> inner) {
  errs() << "The pointer to update is " << this->to_string() << "\n";
  if (innerTy->isArray()) {
    errs() << "Update Array Pointer " << this->to_string() << " with "
         << inner->to_string() << "\n";
    auto at = ptr_cast<MyArrayTy>(innerTy);
    if (at->getElementTy()->compatibleWith(inner)) {
      errs() << "Array pointer used as element pointer\n";
      at->update(inner);
    } else {
      errs() << "Array pointer used as array pointer\n";
      innerTy = MyTy::leastCompatibleType(innerTy, inner);
    }
  } else if (innerTy->isStruct()) {
    auto st = ptr_cast<MyStructTy>(innerTy);
    if (st->getElementTy()->compatibleWith(inner)) {
      errs() << "Struct pointer used as element pointer\n";
      st->updateElement(MyTy::leastCompatibleType(st->getElementTy(), inner));
    } else {
      errs() << "Struct pointer used as struct pointer\n";
      innerTy = MyTy::leastCompatibleType(innerTy, inner);
    }
  } else  {
    innerTy = MyTy::leastCompatibleType(innerTy, inner);
  }
  errs() << "Update to " << this->to_string() << "\n";
}

std::string MyPointerTy::to_string() { return innerTy->to_string() + "*"; }

MyBasicTy::MyBasicTy(Type *basic) {
  basicTy = basic;
  setTypeID(MyTypeID::Basic);
}

Type *MyBasicTy::getBasic() { return basicTy; }

std::string MyBasicTy::to_string() {
  switch (basicTy->getTypeID()) {
  case Type::IntegerTyID: {
    auto intTy = cast<IntegerType>(basicTy);
    return "i" + std::to_string(intTy->getBitWidth());
  }
  case Type::FloatTyID: {
    return "float";
  }
  case Type::DoubleTyID: {
    return "double";
  }
  default:
    return "not_added";
  }
}

MyArrayTy::MyArrayTy(Type *array) {
  auto arrayTy = cast<ArrayType>(array);
  elementCnt = arrayTy->getNumElements();
  elementTy = MyTy::from(arrayTy->getElementType());
  setTypeID(MyTypeID::Array);
}

MyArrayTy::MyArrayTy(std::shared_ptr<MyTy> eTy, int eCnt) {
  elementCnt = eCnt;
  elementTy = eTy;
  setTypeID(MyTypeID::Array);
}

std::shared_ptr<MyTy> llvm::MyArrayTy::getElementTy() { return elementTy; }

int MyArrayTy::getElementCnt() const { return elementCnt; }

std::string MyArrayTy::to_string() {
  return "[" + std::to_string(elementCnt) + " x " + elementTy->to_string() +
         "]";
}

void MyArrayTy::update(std::shared_ptr<MyTy> inner) {
  elementTy = MyTy::leastCompatibleType(elementTy, inner);
}

MyStructTy::MyStructTy(Type* stc) {
  setTypeID(MyTypeID::Struct);
  auto structTy = cast<StructType>(stc);
  if (structTy->hasName()) {
    name = structTy->getName();
  } else {
    name = ""; 
  }  
  int cnt = structTy->getNumElements();
  for (auto i = 0; i < cnt; i++) {
    auto ty = MyTy::from(structTy->getElementType(i));
    elementTy.push_back(ty);
  }
}

std::shared_ptr<MyTy> MyStructTy::getElementTy(int index) {
  return elementTy[index];
}

int MyStructTy::getElementCnt() { return elementTy.size(); }

bool MyStructTy::hasName() const { return name != ""; }

void MyStructTy::update(std::shared_ptr<MyTy> inner) { updateElement(inner); }

void MyStructTy::updateElement(std::shared_ptr<MyTy> ty, int index) {
  elementTy[index] = ty;
}

std::string MyStructTy::to_string() {
  if (hasName()) {
    return "%" + name;
  } else {
    std::string ret = "{";
    for (auto i = 0; i < getElementCnt(); i++) {
      if (i != 0) {
        ret = ret + ",";
      }
      ret = ret + " " + elementTy[i]->to_string();
      if (i == getElementCnt() - 1) {
        ret = ret + " ";
      }
    }
    ret = ret + "}";
    return ret;
  }
}

template std::shared_ptr<MyPointerTy>
MyTy::ptr_cast<MyPointerTy, MyTy>(std::shared_ptr<MyTy>);

template std::shared_ptr<MyArrayTy>
MyTy::ptr_cast<MyArrayTy, MyTy>(std::shared_ptr<MyTy>);

template std::shared_ptr<MyBasicTy>
MyTy::ptr_cast<MyBasicTy, MyTy>(std::shared_ptr<MyTy>);

template std::shared_ptr<MyStructTy>
MyTy::ptr_cast<MyStructTy, MyTy>(std::shared_ptr<MyTy>);
