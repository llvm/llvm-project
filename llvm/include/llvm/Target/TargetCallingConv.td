//===- TargetCallingConv.td - Target Calling Conventions ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces with which targets
// describe their calling conventions.
//
//===----------------------------------------------------------------------===//

class CCAction;
class CallingConv;

/// CCPredicateAction - Instances of this class check some predicate, then
/// delegate to another action if the predicate is true.
class CCPredicateAction<CCAction A> : CCAction {
  CCAction SubAction = A;
}

/// CCIfType - If the current argument is one of the specified types, apply
/// Action A.
class CCIfType<list<ValueType> vts, CCAction A> : CCPredicateAction<A> {
  list<ValueType> VTs = vts;
}

/// CCIf - If the predicate matches, apply A.
class CCIf<string predicate, CCAction A> : CCPredicateAction<A> {
  string Predicate = predicate;
}

/// CCIfByVal - If the current argument has ByVal parameter attribute, apply
/// Action A.
class CCIfByVal<CCAction A> : CCIf<"ArgFlags.isByVal()", A> {
}

/// CCIfCC - Match of the current calling convention is 'CC'.
class CCIfCC<string CC, CCAction A>
  : CCIf<!strconcat("State.getCallingConv() == ", CC), A> {}

/// CCIfInReg - If this argument is marked with the 'inreg' attribute, apply
/// the specified action.
class CCIfInReg<CCAction A> : CCIf<"ArgFlags.isInReg()", A> {}

/// CCIfNest - If this argument is marked with the 'nest' attribute, apply
/// the specified action.
class CCIfNest<CCAction A> : CCIf<"ArgFlags.isNest()", A> {}

/// CCIfNotVarArg - If the current function is not vararg - apply the action
class CCIfNotVarArg<CCAction A> : CCIf<"!State.isVarArg()", A> {}

/// CCAssignToReg - This action matches if there is a register in the specified
/// list that is still available.  If so, it assigns the value to the first
/// available register and succeeds.
class CCAssignToReg<list<Register> regList> : CCAction {
  list<Register> RegList = regList;
}

/// CCAssignToRegWithShadow - Same as CCAssignToReg, but with list of registers
/// which became shadowed, when some register is used.
class CCAssignToRegWithShadow<list<Register> regList,
                              list<Register> shadowList> : CCAction {
  list<Register> RegList = regList;
  list<Register> ShadowRegList = shadowList;
}

/// CCAssignToStack - This action always matches: it assigns the value to a
/// stack slot of the specified size and alignment on the stack.  If size is
/// zero then the ABI size is used; if align is zero then the ABI alignment
/// is used - these may depend on the target or subtarget.
class CCAssignToStack<int size, int align> : CCAction {
  int Size = size;
  int Align = align;
}

/// CCPassByVal - This action always matches: it assigns the value to a stack
/// slot to implement ByVal aggregate parameter passing. Size and alignment
/// specify the minimum size and alignment for the stack slot.
class CCPassByVal<int size, int align> : CCAction {
  int Size = size;
  int Align = align;
}

/// CCPromoteToType - If applied, this promotes the specified current value to
/// the specified type.
class CCPromoteToType<ValueType destTy> : CCAction {
  ValueType DestTy = destTy;
}

/// CCDelegateTo - This action invokes the specified sub-calling-convention.  It
/// is successful if the specified CC matches.
class CCDelegateTo<CallingConv cc> : CCAction {
  CallingConv CC = cc;
}

/// CallingConv - An instance of this is used to define each calling convention
/// that the target supports.
class CallingConv<list<CCAction> actions> {
  list<CCAction> Actions = actions;
}
