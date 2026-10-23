#define DEBUG_TYPE "bsize-invariance-analysis"

#include "llvm/Transforms/BlockSizeInvarianceAnalysis.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Builds state with (const) for all arguments.
BSizeGPUState BlockSizeInvarianceAnalysis::BuildInitialState() const {
  BSizeGPUState st;
  for (Function::const_arg_iterator argIt = F_->arg_begin();
                              argIt != F_->arg_end(); argIt++) {
    BSizeDependenceValue v;
    const Value* arg = &*argIt;
    v = BSizeDependenceValue(CONST, false, arg);
    // If argument is a pointer, set v to address type.
    if (arg->getType()->isPointerTy()) { v.setAddressType(); }
    st.setValue(arg, v);
  }
  return st;
}

bool BlockSizeInvarianceAnalysis::isBSILibraryCall(const StringRef& name) {
  LLVM_DEBUG(errs() << "... Called function name: " << name << "\n");
  if (name == ("llvm.dbg.declare")||
      name == ("llvm.ctlz.i32") ||
      name == ("llvm.trap") || // RISKY?
      name == ("malloc") || // RISKY?
      name == ("llvm.memcpy.p0i8.p0i8.i64") || // RISKY!
      name == ("llvm.nvvm.barrier0") || // __syncthreads() barrier
      name == ("llvm.nvvm.read.ptx.sreg.tid.x") ||
      name == ("llvm.nvvm.read.ptx.sreg.tid.y") ||
      name == ("llvm.nvvm.read.ptx.sreg.tid.z") ||
      name == ("llvm.nvvm.read.ptx.sreg.ntid.x") ||
      name == ("llvm.nvvm.read.ptx.sreg.ntid.y") ||
      name == ("llvm.nvvm.read.ptx.sreg.ntid.z") ||
      name == ("llvm.nvvm.read.ptx.sreg.ctaid.x") ||
      name == ("llvm.nvvm.read.ptx.sreg.ctaid.y") ||
      name == ("llvm.nvvm.read.ptx.sreg.ctaid.z") ||
      name == ("llvm.nvvm.read.ptx.sreg.nctaid.x") ||
      name == ("llvm.nvvm.read.ptx.sreg.nctaid.y") ||
      name == ("llvm.nvvm.read.ptx.sreg.nctad.z") ||
      name == ("llvm.nvvm.sqrt.f") ||
      name == ("llvm.nvvm.saturate.f") ||
      name == ("llvm.nvvm.log.f") ||
      name == ("llvm.nvvm.lg2.approx.f") ||
      name == ("llvm.nvvm.fmax.f") ||
      name == ("llvm.nvvm.fmin.f") ||
      name == ("llvm.nvvm.mul24.ui") ||
      name == ("llvm.umul.with.overflow.i64") ||
      name == ("llvm.nvvm.sin.f") ||
      name == ("llvm.nvvm.cos.f") ||
      // Special case for inlined math functions.
      name.contains("_wrapper"))  return true;
   return false;
}

bool BlockSizeInvarianceAnalysis::AreFunctionCallArgsBSI(
    const CallInst* CI, const BSizeGPUState& st) {
  // Check if all arguments are (const).
  for (unsigned i = 0; i < CI->getNumOperands(); i++) {
    auto argv = st.getValue(CI->getArgOperand(i));
    if (argv.getType() != CONST && argv.getType() != B_CONST) {
      return false;
    }
  }
  return true;
}

BSizeDependenceValue 
BlockSizeInvarianceAnalysis::getCalledFunctionValue(const StringRef& name,
    const CallInst* I, const BSizeGPUState& st) {
  if (name == ("llvm.nvvm.read.ptx.sreg.tid.x")) {
    if (ThreadDim_ == 0) return BSizeDependenceValue(TID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.tid.y")) {
    if (ThreadDim_ == 1) return BSizeDependenceValue(TID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.tid.z")) {
    if (ThreadDim_ == 2) return BSizeDependenceValue(TID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ntid.x")) {
    if (ThreadDim_ == 0) return BSizeDependenceValue(BSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ntid.y")) {
    if (ThreadDim_ == 1) return BSizeDependenceValue(BSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ntid.z") ){
    if (ThreadDim_ == 2) return BSizeDependenceValue(BSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ctaid.x")) {
    if (ThreadDim_ == 0) return BSizeDependenceValue(BID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ctaid.y")) {
    if (ThreadDim_ == 1) return BSizeDependenceValue(BID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.ctaid.z")) {
    if (ThreadDim_ == 2) return BSizeDependenceValue(BID);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.nctaid.x")) {
    if (ThreadDim_ == 0) return BSizeDependenceValue(GSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.nctaid.y")) {
    if (ThreadDim_ == 1) return BSizeDependenceValue(GSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == ("llvm.nvvm.read.ptx.sreg.nctaid.z") ){
    if (ThreadDim_ == 2) return BSizeDependenceValue(GSIZE);
    else return BSizeDependenceValue(CONST, false, I);
  } else if (name == "llvm.nvvm.barrier0") {
    // syncthreads barrier found!
    SyncThreads_.insert(I);
    LLVM_DEBUG(errs() << "SYNCTHREADS FOUND at ");
    LLVM_DEBUG(cast<Instruction>(I)->getDebugLoc().print(errs()));
    LLVM_DEBUG(errs() << " in \n      " << *I << "\n\n");
  }
  // Handle special intrinsic functions.
  // These functions do not introduce any block-size dependence themselves.
  // If any of the arguments is non-constant return (top); else return (const).
  if (isBSILibraryCall(name) && AreFunctionCallArgsBSI(I, st)) {
    return BSizeDependenceValue(CONST, false, I);
  }
  return BSizeDependenceValue(TOP);
}

std::string BlockSizeInvarianceAnalysis::printAccessPattern(
    const Value* root, const std::vector<BSizeDependenceValue>& pattern) {
  std::string s;
  raw_string_ostream ss(s);
  ss << "{" << *root << "} ";
  s = ss.str();
  for (auto v : pattern) {
    s.append("[").append(v.getString()).append("]");
  }
  return s;
}

// Given a shared memory access to p, it does the following:
// - Sets up the shared memory access pattern.
// - Returns (const) if the current pattern matches the shared memory pattern.
//   Otherwise, it returns (top).
// - Updates the current pattern for the load/store instruction I.
BSizeDependenceValue BlockSizeInvarianceAnalysis::HandleSharedMemoryAccess(
    const Value* p, const Instruction* I) {
  BSizeDependenceValue v;
  auto pair = CurrentAccessPatternMap_[p];
  CurrentAccessPatternMap_[I] = pair;
  // Check if the access is consistent.
  const Value* root = pair.first;
  auto curr_acc = pair.second;
  auto acc = SharedMemoryAccessPatternMap_[root];
  bool isConsistent = true;
  if (acc.size() < curr_acc.size()) {
    SharedMemoryAccessPatternMap_[root] = curr_acc;
  }
  for (unsigned i = 0; i < acc.size() && i < curr_acc.size(); ++i) {
    if (curr_acc[i] != acc[i]) {
      // pattern does not match with original pattern
      isConsistent = false;
      break;
    }
  }
  for (unsigned i = acc.size(); i < curr_acc.size(); ++i) {
    if (curr_acc[i].getType() != CONST && curr_acc[i].getType() != TID) {
      isConsistent = false;
      break;
    }
  }
  if (isConsistent) {
    v = BSizeDependenceValue(CONST);
  } else {
    v = BSizeDependenceValue(TOP); 
    LLVM_DEBUG(errs() << "Check for shared memory access consistency failed!!! \n");
    LLVM_DEBUG(errs() << " - Shared memory access pattern : "
                 << printAccessPattern(root, acc) << "\n");
    LLVM_DEBUG(errs() << " - Current access pattern       : "
                 << printAccessPattern(root, curr_acc) << "\n");
  }
  // Set it to address type.
  v.setAddressType();
  return v;
}

// Updates CurrentAccessPatternMap_ with new index.
void BlockSizeInvarianceAnalysis::UpdateCurrentAccessPattern(const Value* initp,
    const Value* finalp, BSizeDependenceValue vIdx) {
  auto pair = CurrentAccessPatternMap_[initp];
  const Value* root = pair.first;
  auto curr_acc = pair.second;
  curr_acc.push_back(vIdx);
  CurrentAccessPatternMap_[finalp] =
      std::pair<const Value*,
                std::vector<BSizeDependenceValue>>(root, curr_acc);
}
 
// Given a constant expression pointer,
// - computes the type of memory space (shared, constant).
// - for a shared memory pointer, initializes CurrentAccessPatternMap_.
// - for gep instructions, computes the value for the pointer.
// - returns the value of the accessed pointer.
BSizeDependenceValue BlockSizeInvarianceAnalysis::HandleConstantExprPointer(
    const Value* p, const BSizeGPUState& st) {
  BSizeDependenceValue v = BSizeDependenceValue(BOT);
  GetElementPtrInst* gep = nullptr;

  ConstantExpr *pe = const_cast<ConstantExpr*>(cast<ConstantExpr>(p));
  // Handle inline getelementptr instruction.
  if (pe->getOpcode() == Instruction::GetElementPtr) {
    gep = cast<GetElementPtrInst>(pe->getAsInstruction());
    if (isa<ConstantExpr>(gep->getPointerOperand())) {
      pe = cast<ConstantExpr>(gep->getPointerOperand());
    }
  }
  // Handle addrspace cast operation for special memory spaces.
  if (pe->getOpcode() == Instruction::AddrSpaceCast) {
    const AddrSpaceCastInst* asci =
        cast<AddrSpaceCastInst>(pe->getAsInstruction());
    v = BSizeDependenceValue(CONST, false, nullptr);
    // Shared memory
    if (asci->getSrcAddressSpace() == 3) {
      CurrentAccessPatternMap_[p] =
          std::pair<const Value*, std::vector<BSizeDependenceValue>>(
              p, std::vector<BSizeDependenceValue>());
    }
    delete asci;
  } else {
    v = st.getValue(pe);
  }
  if (gep) {
    // Sum of indices (assuming there is always a first index).
    BSizeDependenceValue vIdx = st.getValue(gep->getOperand(1));
    for (unsigned i = 1; i < gep->getNumIndices(); i++) {
      const Value* idx = gep->getOperand(i+1);
      vIdx = abstractSum(vIdx, st.getValue(idx), nullptr);
    }
    // Handle shared memory access.
    if (CurrentAccessPatternMap_.find(p) !=
        CurrentAccessPatternMap_.end()) {
      UpdateCurrentAccessPattern(p, p, vIdx);
      // v is recomputed during load/store (see HandleSharedMemoryAccess()).
      v = BSizeDependenceValue(BOT);
    } else {
      // Sum of values of pointer and indices.
      v = abstractSum(v, vIdx, nullptr);
    }
    delete gep;
  }
  // v is assumed to be a pointer!
  v.setAddressType();
  return v;
}

BSizeGPUState BlockSizeInvarianceAnalysis::ExecuteInstruction(
    const Instruction* I, BSizeGPUState st) {
 if (isa<BinaryOperator>(I)) {
    const BinaryOperator* BO = cast<BinaryOperator>(I);
    const Value* in1 = BO->getOperand(0);
    const Value* in2 = BO->getOperand(1);
    // Get values
    BSizeDependenceValue v1, v2;
    v1 = st.getValue(in1);
    v2 = st.getValue(in2);
    // Apply operation to get the resultant value.
    BSizeDependenceValue v;
    auto op = BO->getOpcode();
    switch (op) {
      case Instruction::Add:
        v = abstractSum(v1, v2, I);
        break;
      case Instruction::Sub:
        v = abstractSum(v1, abstractNeg(v2, nullptr), I);
        break;
      case Instruction::Mul:
        v = abstractProd(v1, v2, I);
        break;
      case Instruction::Or:
        v = v1 || v2;
        break;
      case Instruction::And:
        v = v1 && v2;
        break;
      case Instruction::Xor:
        v = (v1 && abstractNeg(v2, nullptr)) || (abstractNeg(v1, nullptr) && v2);
        break;
      default:
        if (v1.getType() == BOT || v2.getType() == BOT) {
          v = BSizeDependenceValue(BOT);
        } else if (v1.getType() == CONST && v2.getType() == CONST) {
          v = BSizeDependenceValue(CONST, false, I);
        } else if (v1.getType() == B_CONST && v2.getType() == B_CONST) {
          v = BSizeDependenceValue(B_CONST);
        } else { v = BSizeDependenceValue(TOP); }
        break;
    }
    st.setValue(BO, v);

  } else if (isa<CastInst>(I)) {
    st.setValue(I, st.getValue(I->getOperand(0)));

  } else if (isa<ExtractValueInst>(I)) {
    const ExtractValueInst *EI = cast<ExtractValueInst>(I);
    st.setValue(I, st.getValue(EI->getAggregateOperand()));

  } else if (isa<CallInst>(I)) {
    const CallInst *CI = cast<CallInst>(I);
    if (CI->isInlineAsm()) {
      // Assembly instructions.
      // ASSUME: the function return value is always block-size dependent.
      st.setValue(CI, BSizeDependenceValue(TOP));
    } else {
      // Function calls.
      Function *calledF = CI->getCalledFunction();

      // Check if the function call is block-size dependent.
      // If the body of the call is block-size dependent, report the call as
      // block-size dependent.
      // ASSUME: If function is not available or is a library call,
      // it must be block-size independent.
      if (FunctionBSIMap_ &&
          FunctionBSIMap_->find(calledF) != FunctionBSIMap_->end()) {
        if (!FunctionBSIMap_->at(calledF) &&
            !isBSILibraryCall(calledF->getName())) {
          BlockSizeDependentAccesses_.insert(CI);
          LLVM_DEBUG(errs() << "BLOCK-SIZE DEPENDENT FUNCTION-CALL FOUND in access at ");
          LLVM_DEBUG(cast<Instruction>(CI)->getDebugLoc().print(errs()));
          LLVM_DEBUG(errs() << " in \n      " << *CI << "\n\n");
        }
      }
      // Handle calls for functions that have not been analyzed.
      else if (!calledF || !calledF->hasName() ||
               !isBSILibraryCall(calledF->getName())) {
        BlockSizeDependentAccesses_.insert(CI);
        LLVM_DEBUG(errs() << "BLOCK-SIZE DEPENDENT FUNCTION-CALL FOUND in access at ");
        LLVM_DEBUG(cast<Instruction>(CI)->getDebugLoc().print(errs()));
        LLVM_DEBUG(errs() << " in \n      " << *CI << "\n\n");
      }

      // Compute return value.
      // Check if it is one of the already analyzed functions (no recursion!)
      if (calledF != F_ && FunctionReturnValueMap_ &&
          FunctionReturnValueMap_->find(calledF) !=
              FunctionReturnValueMap_->end() &&
          !isBSILibraryCall(calledF->getName())) {
        auto v = FunctionReturnValueMap_->at(calledF);
        // Check if all function call arguments are (const) and the function
        // return value is also (const).
        if (AreFunctionCallArgsBSI(CI, st)
            && (v.getType() == CONST || v.getType() == B_CONST)) {
          st.setValue(CI, BSizeDependenceValue(CONST, false, CI));
        } else {
          st.setValue(CI, BSizeDependenceValue(TOP));
        }
      } else {  
        if (!calledF || !calledF->hasName()) {
          // If function has no name, return (top).
          st.setValue(CI, BSizeDependenceValue(TOP));
        } else {
          auto name = calledF->getName();
          // Set value for the called function.
          st.setValue(CI, getCalledFunctionValue(name, CI, st));
        }
      }
    }

  } else if (isa<AllocaInst>(I)) {
    const AllocaInst *AI = cast<AllocaInst>(I);
    // If the allocated type is a pointer, the initial value of the
    // pointer is assumed to be block-size independent and initialized to
    // (const) value. This is useful for external pointers?
    if (AI->getAllocatedType()->isPointerTy()) {
      st.setValue(AI, BSizeDependenceValue(CONST));
    }
    // If the allocated type is a struct or array, it is considered to be
    // of address type with a value (const), since it represents a constant
    // pointer.
    if (AI->getAllocatedType()->isArrayTy() ||
        AI->getAllocatedType()->isStructTy()) {
      auto v = BSizeDependenceValue(CONST);
      v.setAddressType();
      st.setValue(AI, v);
    }

  } else if (isa<LoadInst>(I)) {
    const LoadInst *LI = cast<LoadInst>(I);
    const Value* p = LI->getPointerOperand();
    BSizeDependenceValue v;
    // Handle constant expr pointers!
    if (isa<ConstantExpr>(p)) {
      v = HandleConstantExprPointer(p, st);
    } else {
      v = st.getValue(p);
    }
    if (CurrentAccessPatternMap_.find(p) !=
        CurrentAccessPatternMap_.end()) {
      // Handle shared memory access.
      LLVM_DEBUG(errs() << "Shared memory access found! \n");
      v = HandleSharedMemoryAccess(p, I);
    }
    // When p stores address, if p is (const), return (const); else return (top).
    if (v.isAddressType()) {
      if (v.getType() == CONST) { v = BSizeDependenceValue(CONST);}
      else { v = BSizeDependenceValue(TOP);}
    }
    // If I is a pointer type, v corresponds to the address of a variable.
    if (LI->getType()->isPointerTy()) { v.setAddressType(); }
    st.setValue(LI, v);

  } else if (isa<GetElementPtrInst>(I)) {
    const GetElementPtrInst* GEPI = cast<GetElementPtrInst>(I);
    const Value* p = GEPI->getPointerOperand();
    BSizeDependenceValue v;
    // Handle constant expr pointers!
    if (isa<ConstantExpr>(p)) {
      v = HandleConstantExprPointer(p, st);
    } else {
      v = st.getValue(p);
    }
    // Sum of indices (assuming there is always a first index).
    BSizeDependenceValue vIdx = st.getValue(GEPI->getOperand(1));
    for (unsigned i = 1; i < GEPI->getNumIndices(); i++) {
      const Value* idx = GEPI->getOperand(i+1);
      vIdx = abstractSum(vIdx, st.getValue(idx), nullptr);
    }
    // Handle shared memory access.
    if (CurrentAccessPatternMap_.find(p) !=
        CurrentAccessPatternMap_.end()) {
      LLVM_DEBUG(errs() << "Shared memory access found! \n");
      UpdateCurrentAccessPattern(p, GEPI, vIdx);
      // v is recomputed during load/store (see HandleSharedMemoryAccess()).
      v = BSizeDependenceValue(BOT);
    } else {
      // Sum of values of pointer and indices.
      v = abstractSum(v, vIdx, nullptr);
    }
    // Set v to address type if p stores address, since v corresponds to the
    // offset increment in address stored in pointer p.
    if (st.getValue(p).isAddressType()) v.setAddressType();
    st.setValue(GEPI, v);

  } else if (isa<StoreInst>(I)) {
    const StoreInst *SI = cast<StoreInst>(I);
    const Value* p = SI->getPointerOperand();
    const Value* val = SI->getValueOperand();
    BSizeDependenceValue v;
    // Handle constant expr pointers!
    if (isa<ConstantExpr>(p)) {
      v = HandleConstantExprPointer(p, st);
    } else {
      v = st.getValue(p);
    }
    if (CurrentAccessPatternMap_.find(p) !=
        CurrentAccessPatternMap_.end()) {
      // Handle shared memory access.
      LLVM_DEBUG(errs() << "Shared memory access found! \n");
      v = HandleSharedMemoryAccess(p, I);
    }
    BSizeDependenceValue vVal = st.getValue(val);
    // Detect block-size dependent accesses.
    // These are accesses where the pointer, the value written, or
    // the path-predicate are block-size dependent.
    if (v.isAddressType()
        && (v.getType() != CONST
            || vVal.getType() != CONST
            || (st.getNumThreads().getType() != B_CONST
                && st.getNumThreads().getType() != CONST))) {
      BlockSizeDependentAccesses_.insert(SI);
      LLVM_DEBUG(errs() << "BLOCK-SIZE DEPENDENT ACCESS FOUND in access at ");
      LLVM_DEBUG(cast<Instruction>(SI)->getDebugLoc().print(errs()));
      LLVM_DEBUG(errs() << " in \n      " << *SI << "\n\n");
    }
    if (!v.isAddressType()) {
      // Store value only if p is not address type.
      // Set it to value type explicitly.
      vVal.setValueType();
      // If the store happens within a block-size dependent path,
      // then the value written is also block-size dependent.
      if (st.getNumThreads().getType() != B_CONST &&
          st.getNumThreads().getType() != CONST) {
        vVal = BSizeDependenceValue(TOP);
      }
      st.setValue(p, vVal);
    }

  } else if (isa<SelectInst>(I)) {
    const SelectInst *SI = cast<SelectInst>(I);
    const Value* cond = SI->getCondition();
    const Value* tv = SI->getTrueValue();
    const Value* fv = SI->getFalseValue();
    BSizeDependenceValue v = st.getValue(tv);
    v.join(st.getValue(fv));
    // If condition is bsize-independent, set it to the merge of two values.
    // Else, set it to (top).
    if (st.getValue(cond).getType() == B_CONST ||
        st.getValue(cond).getType() == CONST) {
      st.setValue(I, v);
    } else {
      st.setValue(I, BSizeDependenceValue(TOP));
    }

  } else if(isa<PHINode>(I)) {
    // Merge all incoming values.
    const PHINode *PHI = cast<PHINode>(I);
    BSizeDependenceValue v = st.getValue(PHI);
    for(unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
      v = v.join(st.getValue(PHI->getIncomingValue(i)));
    }
    st.setValue(PHI, v);
 
  } else if (isa<CmpInst>(I)) {
    const CmpInst* CI = cast<CmpInst>(I);
    const Value* in1 = CI->getOperand(0);
    const Value* in2 = CI->getOperand(1);
    // Compute the abstract value of the predicate.
    BSizeDependenceValue v;
    v = abstractRel(st.getValue(in1), st.getValue(in2));
    st.setValue(CI, v);

  } else if (isa<CondBrInst>(I)) {
    const CondBrInst* BI = cast<CondBrInst>(I);
    const Value* cond = BI->getCondition();
    const BasicBlock* nb1 = BI->getSuccessor(0);
    const BasicBlock* nb2 = BI->getSuccessor(1);
    BSizeGPUState st1 = st;
    BSizeGPUState st2 = st;
    // Get the abstract value for branch condition.
    BSizeDependenceValue v = st.getValue(cond);
    // Compute number of threads on the two branches.
    st1.setNumThreads(st.getNumThreads() && v);
    st2.setNumThreads(st.getNumThreads() && abstractNeg(v, nullptr));
    // Add new items to the buffer.
    AddBlockToExecute(nb1, st1);
    AddBlockToExecute(nb2, st2);

  } else if (IsTerminator(I)) {
    // If this is a return instruction, also update the function return
    // value. Also, if FunctionReturnValueMap_ is not null.
    if (isa<ReturnInst>(I) && FunctionReturnValueMap_) {
      const ReturnInst *RI = cast<ReturnInst>(I);
      if (RI->getReturnValue()) {
        BSizeDependenceValue v = st.getValue(RI->getReturnValue());
        if (FunctionReturnValueMap_->find(F_) != FunctionReturnValueMap_->end()) {
          v.join(FunctionReturnValueMap_->at(F_));
        }
        FunctionReturnValueMap_->emplace(F_, v);
      } else {
        FunctionReturnValueMap_->emplace(F_,
            BSizeDependenceValue(CONST, false, nullptr));
      }
    }
    // Add next blocks.
    for (unsigned i = 0; i < I->getNumSuccessors(); i++) {
      const BasicBlock *nb = I->getSuccessor(i);
      AddBlockToExecute(nb, st);
    }
  }
  return st;
}

void BlockSizeInvarianceAnalysis::BuildAnalysisInfo(BSizeGPUState st) {
  BlockSizeDependentAccesses_.clear();
  LLVM_DEBUG(errs() << "Analysis for thread dimension " << ThreadDim_ << "\n");
  initialState_ = st;
  entryBlock_ = &F_->getEntryBlock();
  Execute();
}
