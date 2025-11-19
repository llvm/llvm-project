#ifndef LLVM_TRANSFORMS_UTILS_MYTY_H
#define LLVM_TRANSFORMS_UTILS_MYTY_H

#include <string>
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
	class MyTy {
	public:
		enum MyTypeID { Basic, Pointer, Unknown, Array, Struct, Void };
		MyTy();
		virtual std::string to_string();
		virtual void update(std::shared_ptr<MyTy>);
		MyTypeID getTypeID();
		void setTypeID(MyTypeID);
        bool isBasic();
        bool isPointer();
        bool isVoid();
        bool isArray();
        bool isUnknown();
        bool isStruct();
        bool compatibleWith(std::shared_ptr<MyTy>);
        static std::shared_ptr<MyTy> from(Type *);
        static std::shared_ptr<MyTy> leastCompatibleType(
			std::shared_ptr<MyTy>,
            std::shared_ptr<MyTy>);
        static std::shared_ptr<MyTy> getStructLCA(
			std::shared_ptr<MyTy>,
            std::shared_ptr<MyTy>);
		template <typename T, typename U>
        static std::shared_ptr<T> ptr_cast(std::shared_ptr<U>);
	protected:
		MyTypeID typeId;
        static int floatBitWidth[7];
        static std::shared_ptr<MyTy> basic_with_basic(
			std::shared_ptr<MyTy>,
            std::shared_ptr<MyTy>);
        static std::shared_ptr<MyTy> ptr_with_array(
			std::shared_ptr<MyTy>,
			std::shared_ptr<MyTy>);
        static std::shared_ptr<MyTy> int_with_int(Type *, Type *);
        static std::shared_ptr<MyTy> float_with_float(Type *, Type *);
	};

	class MyVoidTy : public MyTy {
	public:
		MyVoidTy();
		std::string to_string() override;
	};

	class MyPointerTy : public MyTy {
	private:
        std::shared_ptr<MyTy> innerTy;

	public:
        MyPointerTy(std::shared_ptr<MyTy>);
        std::shared_ptr<MyTy> getInner();
		std::string to_string() override;
        void update(std::shared_ptr<MyTy>) override;
	};

	class MyBasicTy : public MyTy {
	private:
        Type *basicTy;

	public:
        MyBasicTy(Type *basic);
        Type *getBasic();
		std::string to_string() override;
	};

	class MyArrayTy : public MyTy {
	private:
		int elementCnt;
        std::shared_ptr<MyTy> elementTy;

	public:
        MyArrayTy(Type *array);
        MyArrayTy(std::shared_ptr<MyTy> eTy, int eCnt);
        std::shared_ptr<MyTy> getElementTy();
		int getElementCnt() const;
		std::string to_string() override;
        void update(std::shared_ptr<MyTy> inner) override;
	};

	class MyStructTy : public MyTy {
    private:
        SmallVector<std::shared_ptr<MyTy>> elementTy;
        std::string name;

	public:
        MyStructTy(Type *stc);
        std::shared_ptr<MyTy> getElementTy(int index = 0);
        std::string to_string() override;
        bool hasName() const;
        int getElementCnt();
        void update(std::shared_ptr<MyTy> inner) override;
        void updateElement(std::shared_ptr<MyTy> ty, int index = 0);
	};
}
#endif