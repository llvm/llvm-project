#include "llvm/Transforms/BSizeGPUState.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

BSizeDependenceValue BSizeGPUState::getValue(const Value* in) const {
  if (isa<ConstantInt>(in) || isa<ConstantFP>(in) ||
      isa<ConstantPointerNull>(in)) {
    // Return value (const);
    return BSizeDependenceValue(CONST, false, in);
  } else if (hasValue(in)) {
    // Return value found in state.
    return AbstractState::getValue(in);
  } else {
    // Default: return (bot) or (unset) value.
    return BSizeDependenceValue(BOT);
  }
}

BSizeGPUState BSizeGPUState::mergeState(const BSizeGPUState& st) const {
  BSizeGPUState result = AbstractState::mergeState(st);
  result.numThreads_ = numThreads_.join(st.numThreads_);
  return result;
}

std::string BSizeGPUState::getString() const {
  std::string s = AbstractState::getString();
  s.append(" #t:").append(numThreads_.getString());
  return s;
}
std::string BSizeGPUState::printInstructionState(const Instruction* I) const {
  std::string s = AbstractState::printInstructionState(I);
  s.append(" #t:").append(numThreads_.getString());
  return s;
}

bool BSizeGPUState::testBSizeGPUState() {
  BSizeGPUState st1, st2;

  static LLVMContext context;
  const Value* g = new GlobalVariable(
               /*Type */ Type::getInt32Ty(context),
               /*isConstant */ false,
               /*Linkage */ GlobalValue::CommonLinkage,
               /*Initializer=*/0, // has initializer, specified below
               /*Name=*/"g");

  st1.setValue(g, BSizeDependenceValue(CONST, false, g));
  errs() << " st1 : " << st1.getString() << "\n";
  errs() << " st2 : " << st2.getString() << "\n";
  errs() << " st1 == st2 : " << (st1 == st2 ? "true": "false") << "\n";
  if (st1 == st2) return false;

  BSizeGPUState st3 = st1.mergeState(st2);
  errs() << " st1.merge(st2) : " << st3.getString() << "\n";
  st3 = st2.mergeState(st1);
  errs() << " st2.merge(st1) : " << st3.getString() << "\n\n";

  st2 = st1;
  errs() << " st1 : " << st1.getString() << "\n";
  errs() << " st2 (st2 := st1) : " << st2.getString() << "\n";
  errs() << " st1 == st2 : " << (st1 == st2 ? "true": "false") << "\n\n";
  if (!(st1 == st2)) return false;

  BSizeDependenceValue v = st1.getValue(g);
  v.setAddressType();
  st1.setValue(g, v);
  errs() << " st1 : " << st1.getString() << "\n";
  errs() << " st2 : " << st2.getString() << "\n";
  errs() << " st1 == st2 : " << (st1 == st2 ? "true": "false") << "\n";
  if (!(st1 == st2)) return false;
  st3 = st2.mergeState(st1);
  errs() << " merge(st1, st2) : " << st3.getString() << "\n\n";
  if (st3.getValue(g) != st1.getValue(g).join(st2.getValue(g))) {
    return false;
  }

  st2.setValue(g, BSizeDependenceValue(TID));
  errs() << " st1 : " << st1.getString() << "\n";
  errs() << " st2 : " << st2.getString() << "\n";
  errs() << " st1 == st2 : " << (st1 == st2 ? "true": "false") << "\n";
  if (st1 == st2) return false;
  st3 = st2.mergeState(st1);
  errs() << " merge(st1, st2) : " << st3.getString() << "\n\n";
  if (st3.getValue(g) != st1.getValue(g).join(st2.getValue(g))) {
    return false;
  }
 
  st1.setValue(g, BSizeDependenceValue(BID));
  errs() << " st1 : " << st1.getString() << "\n";
  errs() << " st2 : " << st2.getString() << "\n";
  st3 = st2.mergeState(st1);
  errs() << " merge(st1, st2) : " << st3.getString() << "\n\n";
  if (st3.getValue(g) != st1.getValue(g).join(st2.getValue(g))) {
    return false;
  }
  return true;
}
