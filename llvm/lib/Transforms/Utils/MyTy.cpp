#include "llvm/Transforms/Utils/MyTy.h"

using namespace llvm;

llvm::MyTy::MyTy() { typeId = MyTypeID::Unknown; }

std::string llvm::MyTy::to_string() { return "unknown"; }

void llvm::MyTy::update(std::shared_ptr<MyTy>) { return; }

llvm::MyTy::MyTypeID llvm::MyTy::getTypeID() { return typeId; }

void llvm::MyTy::setTypeID(MyTypeID t) { typeId = t; }

std::shared_ptr<MyTy> MyTy::from(Type *type) {
  switch (type->getTypeID()) {
  case Type::IntegerTyID:
    return std::make_shared<MyBasicTy>(type);
  case Type::ArrayTyID:
    return std::make_shared<MyArrayTy>(type);
  case Type::PointerTyID:
    return std::make_shared<MyPointerTy>(std::make_shared<MyTy>());
  default:
    return nullptr;
  }
}

bool MyTy::isArray() { return getTypeID() == MyTypeID::Array; }

bool MyTy::isBasic() { return getTypeID() == MyTypeID::Basic; }

bool MyTy::isPointer() { return getTypeID() == MyTypeID::Pointer; }

bool MyTy::isVoid() { return getTypeID() == MyTypeID::Void; }

bool MyTy::isUnknown() { return getTypeID() == MyTypeID::Unknown; }

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

std::shared_ptr<MyTy> MyTy::basic_with_basic(std::shared_ptr<MyTy> t1,
                                             std::shared_ptr<MyTy> t2) {
  auto b1 = ptr_cast<MyBasicTy>(t1)->getBasic();
  auto b2 = ptr_cast<MyBasicTy>(t2)->getBasic();
  if (b1->isIntegerTy()) {
    if (b2->isIntegerTy()) {
      return int_with_int(b1, b2);
    } else {
      // Figure out whether I should add Float.
    }
  }
  return std::make_shared<MyTy>();
}

std::shared_ptr<MyTy> MyTy::ptr_with_array(std::shared_ptr<MyTy> t1,
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
  if (this->isUnknown() || ty->isUnknown()) {
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
          // Figure out whether I should add Float.
        }
      }
      return false;
    } else {
      return false;
    }
  }
}

std::shared_ptr<MyTy> llvm::MyTy::leastCompatibleType(std::shared_ptr<MyTy> t1,
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
          std::max(pt1->size(), pt2->size()));
    } else {
      ret = std::make_shared<MyVoidTy>();
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

std::shared_ptr<MyTy> llvm::MyPointerTy::getInner() { return innerTy; }

void MyPointerTy::update(std::shared_ptr<MyTy> inner) {
  errs() << "The pointer to update is " << this->to_string() << "\n";
  if (innerTy->isArray()) {
    errs() << "Update Array Pointer " << this->to_string() << " with "
         << inner->to_string() << "\n";
    auto at = ptr_cast<MyArrayTy>(innerTy);
    if (at->getElementTy()->compatibleWith(inner)) {
      errs() << "Array pointer used as element pointer\n";
      innerTy = std::make_shared<MyArrayTy>(
          MyTy::leastCompatibleType(at->getElementTy(), inner), at->size());
    } else {
      errs() << "Array pointer used as array pointer\n";
      innerTy = MyTy::leastCompatibleType(innerTy, inner);
    }
  } else {
    innerTy = MyTy::leastCompatibleType(innerTy, inner);
  }
  errs() << "Update to " << this->to_string() << "\n";
}

std::string MyPointerTy::to_string() { return innerTy->to_string() + "*"; }

MyBasicTy::MyBasicTy(Type *basic) {
  basicTy = basic;
  setTypeID(MyTypeID::Basic);
}

Type *llvm::MyBasicTy::getBasic() { return basicTy; }

std::string MyBasicTy::to_string() {
  switch (basicTy->getTypeID()) {
  case Type::IntegerTyID: {
    auto intTy = cast<IntegerType>(basicTy);
    return "i" + std::to_string(intTy->getBitWidth());
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

llvm::MyArrayTy::MyArrayTy(std::shared_ptr<MyTy> eTy, int eCnt) {
  elementCnt = eCnt;
  elementTy = eTy;
  setTypeID(MyTypeID::Array);
}

std::shared_ptr<MyTy> llvm::MyArrayTy::getElementTy() { return elementTy; }

int llvm::MyArrayTy::size() { return elementCnt; }

std::string MyArrayTy::to_string() {
  return "[" + std::to_string(elementCnt) + " x " + elementTy->to_string() +
         "]";
}

void MyArrayTy::update(std::shared_ptr<MyTy> inner) {
  elementTy = MyTy::leastCompatibleType(elementTy, inner);
}

template std::shared_ptr<MyPointerTy>
MyTy::ptr_cast<MyPointerTy, MyTy>(std::shared_ptr<MyTy>);

template std::shared_ptr<MyArrayTy>
MyTy::ptr_cast<MyArrayTy, MyTy>(std::shared_ptr<MyTy>);

template std::shared_ptr<MyBasicTy>
MyTy::ptr_cast<MyBasicTy, MyTy>(std::shared_ptr<MyTy>);
