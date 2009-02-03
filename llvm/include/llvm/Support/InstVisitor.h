//===- llvm/Support/InstVisitor.h - Define instruction visitors -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_SUPPORT_INSTVISITOR_H
#define LLVM_SUPPORT_INSTVISITOR_H

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"

namespace llvm {

// We operate on opaque instruction classes, so forward declare all instruction
// types now...
//
#define HANDLE_INST(NUM, OPCODE, CLASS)   class CLASS;
#include "llvm/Instruction.def"

#define DELEGATE(CLASS_TO_VISIT) \
  return static_cast<SubClass*>(this)-> \
               visit##CLASS_TO_VISIT(static_cast<CLASS_TO_VISIT&>(I))


/// @brief Base class for instruction visitors
///
/// Instruction visitors are used when you want to perform different action for
/// different kinds of instruction without without having to use lots of casts 
/// and a big switch statement (in your code that is). 
///
/// To define your own visitor, inherit from this class, specifying your
/// new type for the 'SubClass' template parameter, and "override" visitXXX
/// functions in your class. I say "overriding" because this class is defined 
/// in terms of statically resolved overloading, not virtual functions.  
/// 
/// For example, here is a visitor that counts the number of malloc 
/// instructions processed:
///
///  /// Declare the class.  Note that we derive from InstVisitor instantiated
///  /// with _our new subclasses_ type.
///  ///
///  struct CountMallocVisitor : public InstVisitor<CountMallocVisitor> {
///    unsigned Count;
///    CountMallocVisitor() : Count(0) {}
///
///    void visitMallocInst(MallocInst &MI) { ++Count; }
///  };
///
///  And this class would be used like this:
///    CountMallocVistor CMV;
///    CMV.visit(function);
///    NumMallocs = CMV.Count;
///
/// The defined has 'visit' methods for Instruction, and also for BasicBlock,
/// Function, and Module, which recursively process all conained instructions.
///
/// Note that if you don't implement visitXXX for some instruction type,
/// the visitXXX method for instruction superclass will be invoked. So
/// if instructions are added in the future, they will be automatically
/// supported, if you handle on of their superclasses.
///
/// The optional second template argument specifies the type that instruction 
/// visitation functions should return. If you specify this, you *MUST* provide 
/// an implementation of visitInstruction though!.
///
/// Note that this class is specifically designed as a template to avoid
/// virtual function call overhead.  Defining and using an InstVisitor is just
/// as efficient as having your own switch statement over the instruction
/// opcode.
template<typename SubClass, typename RetTy=void>
class InstVisitor {
  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the InstVisitor that you
  // use to visit instructions...
  //

public:
  // Generic visit method - Allow visitation to all instructions in a range
  template<class Iterator>
  void visit(Iterator Start, Iterator End) {
    while (Start != End)
      static_cast<SubClass*>(this)->visit(*Start++);
  }

  // Define visitors for functions and basic blocks...
  //
  void visit(Module &M) {
    static_cast<SubClass*>(this)->visitModule(M);
    visit(M.begin(), M.end());
  }
  void visit(Function &F) {
    static_cast<SubClass*>(this)->visitFunction(F);
    visit(F.begin(), F.end());
  }
  void visit(BasicBlock &BB) {
    static_cast<SubClass*>(this)->visitBasicBlock(BB);
    visit(BB.begin(), BB.end());
  }

  // Forwarding functions so that the user can visit with pointers AND refs.
  void visit(Module       *M)  { visit(*M); }
  void visit(Function     *F)  { visit(*F); }
  void visit(BasicBlock   *BB) { visit(*BB); }
  RetTy visit(Instruction *I)  { return visit(*I); }

  // visit - Finally, code to visit an instruction...
  //
  RetTy visit(Instruction &I) {
    switch (I.getOpcode()) {
    default: assert(0 && "Unknown instruction type encountered!");
             abort();
      // Build the switch statement using the Instruction.def file...
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: return \
           static_cast<SubClass*>(this)-> \
                      visit##OPCODE(static_cast<CLASS&>(I));
#include "llvm/Instruction.def"
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // When visiting a module, function or basic block directly, these methods get
  // called to indicate when transitioning into a new unit.
  //
  void visitModule    (Module &M) {}
  void visitFunction  (Function &F) {}
  void visitBasicBlock(BasicBlock &BB) {}

  // Define instruction specific visitor functions that can be overridden to
  // handle SPECIFIC instructions.  These functions automatically define
  // visitMul to proxy to visitBinaryOperator for instance in case the user does
  // not need this generality.
  //
  // The one problem case we have to handle here though is that the PHINode
  // class and opcode name are the exact same.  Because of this, we cannot
  // define visitPHINode (the inst version) to forward to visitPHINode (the
  // generic version) without multiply defined symbols and recursion.  To handle
  // this, we do not autoexpand "Other" instructions, we do it manually.
  //
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    RetTy visit##OPCODE(CLASS &I) { DELEGATE(CLASS); }
#include "llvm/Instruction.def"

  // Specific Instruction type classes... note that all of the casts are
  // necessary because we use the instruction classes as opaque types...
  //
  RetTy visitReturnInst(ReturnInst &I)              { DELEGATE(TerminatorInst);}
  RetTy visitBranchInst(BranchInst &I)              { DELEGATE(TerminatorInst);}
  RetTy visitSwitchInst(SwitchInst &I)              { DELEGATE(TerminatorInst);}
  RetTy visitInvokeInst(InvokeInst &I)              { DELEGATE(TerminatorInst);}
  RetTy visitUnwindInst(UnwindInst &I)              { DELEGATE(TerminatorInst);}
  RetTy visitUnreachableInst(UnreachableInst &I)    { DELEGATE(TerminatorInst);}
  RetTy visitICmpInst(ICmpInst &I)                  { DELEGATE(CmpInst);}
  RetTy visitFCmpInst(FCmpInst &I)                  { DELEGATE(CmpInst);}
  RetTy visitVICmpInst(VICmpInst &I)                { DELEGATE(CmpInst);}
  RetTy visitVFCmpInst(VFCmpInst &I)                { DELEGATE(CmpInst);}
  RetTy visitMallocInst(MallocInst &I)              { DELEGATE(AllocationInst);}
  RetTy visitAllocaInst(AllocaInst &I)              { DELEGATE(AllocationInst);}
  RetTy visitFreeInst(FreeInst     &I)              { DELEGATE(Instruction); }
  RetTy visitLoadInst(LoadInst     &I)              { DELEGATE(Instruction); }
  RetTy visitStoreInst(StoreInst   &I)              { DELEGATE(Instruction); }
  RetTy visitGetElementPtrInst(GetElementPtrInst &I){ DELEGATE(Instruction); }
  RetTy visitPHINode(PHINode       &I)              { DELEGATE(Instruction); }
  RetTy visitTruncInst(TruncInst &I)                { DELEGATE(CastInst); }
  RetTy visitZExtInst(ZExtInst &I)                  { DELEGATE(CastInst); }
  RetTy visitSExtInst(SExtInst &I)                  { DELEGATE(CastInst); }
  RetTy visitFPTruncInst(FPTruncInst &I)            { DELEGATE(CastInst); }
  RetTy visitFPExtInst(FPExtInst &I)                { DELEGATE(CastInst); }
  RetTy visitFPToUIInst(FPToUIInst &I)              { DELEGATE(CastInst); }
  RetTy visitFPToSIInst(FPToSIInst &I)              { DELEGATE(CastInst); }
  RetTy visitUIToFPInst(UIToFPInst &I)              { DELEGATE(CastInst); }
  RetTy visitSIToFPInst(SIToFPInst &I)              { DELEGATE(CastInst); }
  RetTy visitPtrToIntInst(PtrToIntInst &I)          { DELEGATE(CastInst); }
  RetTy visitIntToPtrInst(IntToPtrInst &I)          { DELEGATE(CastInst); }
  RetTy visitBitCastInst(BitCastInst &I)            { DELEGATE(CastInst); }
  RetTy visitSelectInst(SelectInst &I)              { DELEGATE(Instruction); }
  RetTy visitCallInst(CallInst     &I)              { DELEGATE(Instruction); }
  RetTy visitVAArgInst(VAArgInst   &I)              { DELEGATE(Instruction); }
  RetTy visitExtractElementInst(ExtractElementInst &I) { DELEGATE(Instruction);}
  RetTy visitInsertElementInst(InsertElementInst &I) { DELEGATE(Instruction); }
  RetTy visitShuffleVectorInst(ShuffleVectorInst &I) { DELEGATE(Instruction); }
  RetTy visitExtractValueInst(ExtractValueInst &I)  { DELEGATE(Instruction);}
  RetTy visitInsertValueInst(InsertValueInst &I)    { DELEGATE(Instruction); }

  // Next level propagators... if the user does not overload a specific
  // instruction type, they can overload one of these to get the whole class
  // of instructions...
  //
  RetTy visitTerminatorInst(TerminatorInst &I) { DELEGATE(Instruction); }
  RetTy visitBinaryOperator(BinaryOperator &I) { DELEGATE(Instruction); }
  RetTy visitAllocationInst(AllocationInst &I) { DELEGATE(Instruction); }
  RetTy visitCmpInst(CmpInst &I)               { DELEGATE(Instruction); }
  RetTy visitCastInst(CastInst &I)             { DELEGATE(Instruction); }

  // If the user wants a 'default' case, they can choose to override this
  // function.  If this function is not overloaded in the users subclass, then
  // this instruction just gets ignored.
  //
  // Note that you MUST override this function if your return type is not void.
  //
  void visitInstruction(Instruction &I) {}  // Ignore unhandled instructions
};

#undef DELEGATE

} // End llvm namespace

#endif
