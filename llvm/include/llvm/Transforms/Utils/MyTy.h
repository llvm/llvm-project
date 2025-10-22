#ifndef LLVM_TRANSFORMS_UTILS_MYTY_H
#define LLVM_TRANSFORMS_UTILS_MYTY_H

#include <string>
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
	class MyTy {
	public:
		enum MyTypeID { Basic, Pointer, Null, Array, Struct, Void };
		MyTy();
		virtual std::string to_string();
		virtual void update(MyTy *);
		virtual MyTy *clone();
		MyTypeID getTypeID();
		void setTypeID(MyTypeID);
		static MyTy *from(Type *);
		static MyTy *leastCommonType(MyTy *t1, MyTy *t2);
		static bool equal(MyTy *t1, MyTy *t2);
		template <typename MySomeTy> MySomeTy *castAs();
	protected:
		MyTypeID typeId;
	};

	class MyVoidTy : public MyTy {
	public:
		MyVoidTy();
		std::string to_string() override;
        MyTy *clone() override;
	};

	class MyPointerTy : public MyTy {
	private:
		MyTy *innerTy;

	public:
		MyPointerTy(MyTy *inner);
		MyTy *getInner();
		std::string to_string() override;
		void update(MyTy *inner) override;
		MyTy *clone() override;
	};

	class MyBasicTy : public MyTy {
	private:
		Type *basicTy;

	public:
		MyBasicTy(Type *basic);
		Type *getBasic();
		std::string to_string() override;
		MyTy *clone() override;
	};

	class MyArrayTy : public MyTy {
	private:
		int elementCnt;
		MyTy *elementTy;

	public:
		MyArrayTy(Type *array);
		MyArrayTy(MyTy *eTy, int eCnt);
		MyTy *getElement();
		int size();
		std::string to_string() override;
		void update(MyTy *inner) override;
		MyTy *clone() override;
	};

	class MyStructTy : public MyTy {
	private:
		int elementCnt;
		SmallVector<MyTy *, 16> elementTy;
		StructType *structTy;

	public:
		MyStructTy(Type *_struct);
		std::string to_string() override;
		MyTy *clone() override;
	};
}
#endif