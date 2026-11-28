#ifndef POINTER_ABSTRACT_VALUE_H
#define POINTER_ABSTRACT_VALUE_H

#include "llvm/Transforms/AbstractValue.h"

// This class defines that an abstract value that distinguishes address (lvalue)
// of a variable from the value (rvalue) of the variable. Hence, it keeps a flag
// `isAddressType_' to track whether the abstract value represents the address or the
// value of the variable.
template<typename T>
class PointerAbstractValue : public AbstractValue<T> { 
 public:
  PointerAbstractValue() : isAddressType_(false) {}

  virtual ~PointerAbstractValue() = 0; 

  bool isAddressType() const { return isAddressType_; }

  void setAddressType() { isAddressType_ = true; }

  void setValueType() { isAddressType_ = false; }

  virtual std::string getString() const = 0;

  // Returns a merge of this value with value v.
  virtual T join(const T& v) const = 0;

 private:
  // Does this value represent the address (lvalue) of the variable or the
  // value (rvalue) of the variable?
  bool isAddressType_;
};

template<typename T>
PointerAbstractValue<T>::~PointerAbstractValue() {}

#endif /* PointerAbstractValue.h */
