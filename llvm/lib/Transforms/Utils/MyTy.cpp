#include "llvm/Transforms/Utils/MyTy.h"

using namespace llvm;

llvm::MyTy::MyTy() { typeId = MyTypeID::Null; }

std::string llvm::MyTy::to_string() { return "null"; }

void llvm::MyTy::update(MyTy *) { return; }

MyTy *llvm::MyTy::clone() {
  return new MyTy();
}

llvm::MyTy::MyTypeID llvm::MyTy::getTypeID() { return typeId; }

void llvm::MyTy::setTypeID(MyTypeID t) { typeId = t; }

MyTy *MyTy::from(Type *type) {
  switch (type->getTypeID()) {
  case Type::IntegerTyID:
    return new MyBasicTy(type);
  case Type::ArrayTyID:
    return new MyArrayTy(type);
  case Type::PointerTyID:
    return new MyPointerTy(new MyVoidTy());
  case Type::StructTyID:
    return new MyStructTy(type);
  default:
    return nullptr;
  }
}

template <typename MySomeTy> MySomeTy *MyTy::castAs() {
  return static_cast<MySomeTy *>(this);
}

MyTy *llvm::MyTy::leastCommonType(MyTy *t1, MyTy *t2) {
  MyTy *ret = nullptr;
  if (t1->getTypeID() == MyTypeID::Void) {
    ret = t2->clone();
  } else if (t1->getTypeID() == MyTypeID::Basic) {
    if (t2->getTypeID() == MyTypeID::Void) {
      ret = t1->clone();
    } else if (t2->getTypeID() == MyTypeID::Basic) {
      // Need to check ALL basic types.
      Type *b1 = t1->castAs<MyBasicTy>()->getBasic();
      Type *b2 = t2->castAs<MyBasicTy>()->getBasic();
      if (b1->isIntegerTy()) {
        if (b2->isIntegerTy()) {
          auto *i1 = cast<IntegerType>(b1);
          auto *i2 = cast<IntegerType>(b2);
          ret = new MyBasicTy(i1->getBitWidth() > i2->getBitWidth() ? i1 : i2);
        } else {
          // Figure out whether I should add Float.
        }
      }
    } else {
      ret = new MyVoidTy();
    }
  } else if (t1->getTypeID() == MyTypeID::Pointer) {
    if (t2->getTypeID() == MyTypeID::Void) {
      ret = t1->clone();
    } else if (t2->getTypeID() == MyTypeID::Pointer) {
      MyPointerTy *pt1 = t1->castAs<MyPointerTy>();
      MyPointerTy *pt2 = t2->castAs<MyPointerTy>();
      if (pt1->getInner()->getTypeID() == MyTypeID::Array) {
        // 由于内存布局，数组指针*可以*作为其元素指针使用
        // 新建一个指针类型，递归调用该函数，按照结果回来更新数组指针
        auto *aTy =
            new MyPointerTy(pt1->getInner()->castAs<MyArrayTy>()->getElement());
        // 暂且只允许相同类型，这样不用递归调用
        if (MyTy::equal(aTy, pt2)) {
          ret = pt1->clone();
        } else {
          ret = new MyPointerTy(
              leastCommonType(pt1->getInner(), pt2->getInner()));
        }
      } else if (pt2->getInner()->getTypeID() == MyTypeID::Array) {
        auto *aTy =
            new MyPointerTy(pt2->getInner()->castAs<MyArrayTy>()->getElement());
        if (MyTy::equal(aTy, pt1)) {
          ret = pt2->clone();
        } else {
          ret = new MyPointerTy(
              leastCommonType(pt1->getInner(), pt2->getInner()));
        }
      } else {
        ret = new MyPointerTy(leastCommonType(pt1->getInner(), pt2->getInner()));
      }
    } else if (t2->getTypeID() == MyTypeID::Array) {
      MyPointerTy *pt1 = t1->castAs<MyPointerTy>();
      MyArrayTy *pt2 = t2->castAs<MyArrayTy>();
      if (MyTy::equal(pt1->getInner(), pt2->getElement())) {
        ret = pt2->clone();
      } else {
        ret = new MyVoidTy();
      }
    } else {
      ret = new MyVoidTy();
    }
  } else if (t1->getTypeID() == MyTypeID::Array) {
    if (t2->getTypeID() == MyTypeID::Void) {
      ret = t1->clone();
    } else if (t2->getTypeID() == MyTypeID::Pointer) {
      MyArrayTy *pt1 = t1->castAs<MyArrayTy>();
      MyPointerTy *pt2 = t2->castAs<MyPointerTy>();
      if (MyTy::equal(pt1->getElement(), pt2->getInner())) {
        ret = pt1->clone();
      } else {
        ret = new MyVoidTy();
      }
    } else if (t2->getTypeID() == MyTypeID::Array) {
      MyArrayTy *pt1 = (MyArrayTy *)t1;
      MyArrayTy *pt2 = (MyArrayTy *)t2;
      ret = new MyArrayTy(leastCommonType(pt1->getElement(), pt2->getElement()),
                          std::max(pt1->size(), pt2->size()));
    } else {
      ret = new MyVoidTy();
    }
  } else {
    ret = new MyVoidTy();
  }
  // 需要正确处理内存问题！
  // delete t1;
  // delete t2;
  /*errs() << t1->to_string() << " and " << t2->to_string() << " common is "
         << ret->to_string() << "\n";*/
  return ret;
}

bool llvm::MyTy::equal(MyTy *t1, MyTy *t2) {
  return t1->to_string() == t2->to_string();
}

MyVoidTy::MyVoidTy() { setTypeID(MyTypeID::Void); }

std::string MyVoidTy::to_string() { return "void"; }

MyTy *MyVoidTy::clone() { return new MyVoidTy(); }

MyPointerTy::MyPointerTy(MyTy *inner) {
  innerTy = inner;
  setTypeID(MyTypeID::Pointer);
}

MyTy *llvm::MyPointerTy::getInner() { return innerTy; }

void MyPointerTy::update(MyTy *inner) {
  // Todo: Find common type.
  innerTy = MyTy::leastCommonType(innerTy, inner);
}

std::string MyPointerTy::to_string() { return innerTy->to_string() + "*"; }

MyTy *MyPointerTy::clone() { return new MyPointerTy(this->innerTy); }

MyBasicTy::MyBasicTy(Type *basic) {
  basicTy = basic;
  setTypeID(MyTypeID::Basic);
}

Type *llvm::MyBasicTy::getBasic() { return basicTy; }

std::string MyBasicTy::to_string() {
  switch (basicTy->getTypeID()) {
  case Type::IntegerTyID: {
    auto *intTy = cast<IntegerType>(basicTy);
    return "i" + std::to_string(intTy->getBitWidth());
  }
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  default:
    return "error";
  }
}

MyTy *MyBasicTy::clone() { return new MyBasicTy(this->basicTy); }

MyArrayTy::MyArrayTy(Type *array) {
  auto *arrayTy = cast<ArrayType>(array);
  elementCnt = arrayTy->getNumElements();
  elementTy = MyTy::from(arrayTy->getElementType());
  setTypeID(MyTypeID::Array);
}

llvm::MyArrayTy::MyArrayTy(MyTy *eTy, int eCnt) {
  elementCnt = eCnt;
  elementTy = eTy;
}

MyTy *llvm::MyArrayTy::getElement() { return elementTy; }

int llvm::MyArrayTy::size() { return elementCnt; }

std::string MyArrayTy::to_string() {
  return "[" + std::to_string(elementCnt) + " x " + elementTy->to_string() +
         "]";
}

void MyArrayTy::update(MyTy *inner) {
  // Todo: Find common type.
  elementTy = MyTy::leastCommonType(elementTy, inner);
}

MyTy *MyArrayTy::clone() { return new MyArrayTy(elementTy, elementCnt); }

MyStructTy::MyStructTy(Type *_struct) {
  structTy = cast<StructType>(_struct);
  elementCnt = structTy->getNumElements();
  for (auto i = 0; i < elementCnt; i++) {
    elementTy.push_back(MyTy::from(structTy->getElementType(i)));
  }
  setTypeID(MyTy::Struct);
}

std::string MyStructTy::to_string() { return structTy->getName().str(); }

MyTy *llvm::MyStructTy::clone() { return new MyStructTy(structTy); }
