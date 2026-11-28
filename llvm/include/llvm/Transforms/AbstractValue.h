#ifndef ABSTRACT_VALUE_H
#define ABSTRACT_VALUE_H

#include <string>

// This class defines an abstract value and semantics for the various
// operations performed during the abstract execution of the program.
template<typename T>
class AbstractValue { 
 public:
  AbstractValue() {}

  virtual ~AbstractValue() = 0; 

  virtual std::string getString() const = 0;

  // Returns a merge of this value with value v.
  virtual T join(const T& v) const = 0;
};

template<typename T>
AbstractValue<T>::~AbstractValue() {}

#endif /* AbstractValue.h */
