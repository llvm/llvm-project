#ifndef ABSTRACT_STATE_H
#define ABSTRACT_STATE_H

#include "llvm/Transforms/AbstractValue.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include <map>
#include <string>

using namespace llvm;

// Defines an abstract state used by the abstract execution engine to store
// the current state of the abstract execution.
template <typename T, typename U>
class AbstractState {
 static_assert(
   std::is_base_of<AbstractValue<T>, T>::value, 
   "T must be a descendant of AbstractValue<T>"
 );
 public:
  AbstractState() {}

  virtual ~AbstractState() = 0;

  void clear() { valueMap_.clear(); }

  bool operator==(const AbstractState& st) const {
    return (valueMap_ == st.valueMap_);
  }

  void operator=(const AbstractState& st) {
    valueMap_ = st.valueMap_;
  }

  bool hasValue(const Value* in) const {
    return (valueMap_.find(in) != valueMap_.end());
  }

  virtual T getValue(const Value* in) const {
    return valueMap_.at(in);
  }

  virtual void setValue(const Value* in, T v) { valueMap_[in] = v; }

  virtual U mergeState(const U& st) const;

  // Pretty printing 
  virtual std::string getString() const;
  virtual std::string printInstructionState(const Instruction* I) const;

 private: 
  // Map from variables to their abstract values.
  std::map<const Value*, T> valueMap_;
};

template<typename T, typename U>
AbstractState<T, U>::~AbstractState() {}

template<typename T, typename U>
U AbstractState<T, U>::mergeState(const U& st) const {
  U result = st;
  for (auto it = this->valueMap_.begin(), ite = this->valueMap_.end(); 
                                                      it != ite; ++it) {
    const Value* in = it->first;
    T v = it->second;
    if (st.hasValue(in)) {
      result.setValue(in, v.join(st.valueMap_.at(in)));
    } else {
      result.setValue(in, v);
    }
  }
  return result;
}

template<typename T, typename U>
std::string AbstractState<T, U>::getString() const {
  std::string s;
  s.append("[");
  for (auto it = valueMap_.begin(), ite = valueMap_.end(); it != ite; ++it) {
    const Value* in = it->first;
    T v = it->second;
    s.append(in->getName()).append(":").append(v.getString()).append(", ");
  }
  s.append("]");
  return s;
}

template<typename T, typename U>
std::string AbstractState<T, U>::printInstructionState(
    const Instruction* I) const {
  std::string s;
  s.append("[");
  for (unsigned i = 0; i < I->getNumOperands(); i++) {
    T v = this->getValue(I->getOperand(i));
    s.append(I->getOperand(i)->getName()).append(":").append(v.getString()).append(",");
  }
  s.append("]");
  return s;
}

#endif /* AbstractState.h */
