//===-- llvm/DebugProgramInstruction.h - Stream of debug info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data structures for storing variable assignment information in LLVM. In the
// dbg.value design, a dbg.value intrinsic specifies the position in a block
// a source variable take on an LLVM Value:
//
//    %foo = add i32 1, %0
//    dbg.value(metadata i32 %foo, ...)
//    %bar = void call @ext(%foo);
//
// and all information is stored in the Value / Metadata hierachy defined
// elsewhere in LLVM. In the "DPValue" design, each instruction /may/ have a
// connection with a DPMarker, which identifies a position immediately before the
// instruction, and each DPMarker /may/ then have connections to DPValues which
// record the variable assignment information. To illustrate:
//
//    %foo = add i32 1, %0
//       ; foo->DbgMarker == nullptr
//       ;; There are no variable assignments / debug records "in front" of
//       ;; the instruction for %foo, therefore it has no DbgMarker.
//    %bar = void call @ext(%foo)
//       ; bar->DbgMarker = {
//       ;   StoredDPValues = {
//       ;     DPValue(metadata i32 %foo, ...)
//       ;   }
//       ; }
//       ;; There is a debug-info record in front of the %bar instruction,
//       ;; thus it points at a DPMarker object. That DPMarker contains a
//       ;; DPValue in it's ilist, storing the equivalent information to the
//       ;; dbg.value above: the Value, DILocalVariable, etc.
//
// This structure separates the two concerns of the position of the debug-info
// in the function, and the Value that it refers to. It also creates a new
// "place" in-between the Value / Metadata hierachy where we can customise
// storage and allocation techniques to better suite debug-info workloads.
// NB: as of the initial prototype, none of that has actually been attempted
// yet.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGPROGRAMINSTRUCTION_H
#define LLVM_IR_DEBUGPROGRAMINSTRUCTION_H

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/SymbolTableListTraits.h"

namespace llvm {

class Instruction;
class BasicBlock;
class MDNode;
class Module;
class DbgVariableIntrinsic;
class DIAssignID;
class DPMarker;
class DPValue;
class raw_ostream;

/// Record of a variable value-assignment, aka a non instruction representation
/// of the dbg.value intrinsic. Features various methods copied across from the
/// Instruction class to aid ease-of-use. DPValue objects should always be
/// linked into a DPMarker's StoredDPValues list. The marker connects a DPValue
/// back to it's position in the BasicBlock.
///
/// This class inherits from DebugValueUser to allow LLVM's metadata facilities
/// to update our references to metadata beneath our feet.
class DPValue : public ilist_node<DPValue>, private DebugValueUser {
  friend class DebugValueUser;

  // NB: there is no explicit "Value" field in this class, it's effectively the
  // DebugValueUser superclass instead. The referred to Value can either be a
  // ValueAsMetadata or a DIArgList.

  DILocalVariable *Variable;
  DIExpression *Expression;
  DebugLoc DbgLoc;
  DIExpression *AddressExpression;

public:
  void deleteInstr();

  const Instruction *getInstruction() const;
  const BasicBlock *getParent() const;
  BasicBlock *getParent();
  void dump() const;
  void removeFromParent();
  void eraseFromParent();

  DPValue *getNextNode() { return &*std::next(getIterator()); }
  DPValue *getPrevNode() { return &*std::prev(getIterator()); }

  using self_iterator = simple_ilist<DPValue>::iterator;
  using const_self_iterator = simple_ilist<DPValue>::const_iterator;

  enum class LocationType {
    Declare,
    Value,
    Assign,

    End, ///< Marks the end of the concrete types.
    Any, ///< To indicate all LocationTypes in searches.
  };
  /// Classification of the debug-info record that this DPValue represents.
  /// Essentially, "is this a dbg.value or dbg.declare?". dbg.declares are not
  /// currently supported, but it would be trivial to do so.
  LocationType Type;

  /// Marker that this DPValue is linked into.
  DPMarker *Marker = nullptr;

  /// Create a new DPValue representing the intrinsic \p DVI, for example the
  /// assignment represented by a dbg.value.
  DPValue(const DbgVariableIntrinsic *DVI);
  DPValue(const DPValue &DPV);
  /// Directly construct a new DPValue representing a dbg.value intrinsic
  /// assigning \p Location to the DV / Expr / DI variable.
  DPValue(Metadata *Location, DILocalVariable *DV, DIExpression *Expr,
          const DILocation *DI, LocationType Type = LocationType::Value);
  DPValue(Metadata *Value, DILocalVariable *Variable, DIExpression *Expression,
          DIAssignID *AssignID, Metadata *Address,
          DIExpression *AddressExpression, const DILocation *DI);

  static DPValue *createDPVAssign(Value *Val, DILocalVariable *Variable,
                                  DIExpression *Expression,
                                  DIAssignID *AssignID, Value *Address,
                                  DIExpression *AddressExpression,
                                  const DILocation *DI);
  static DPValue *createLinkedDPVAssign(Instruction *LinkedInstr, Value *Val,
                                        DILocalVariable *Variable,
                                        DIExpression *Expression,
                                        Value *Address,
                                        DIExpression *AddressExpression,
                                        const DILocation *DI);

  static DPValue *createDPValue(Value *Location, DILocalVariable *DV,
                                DIExpression *Expr, const DILocation *DI);
  static DPValue *createDPValue(Value *Location, DILocalVariable *DV,
                                DIExpression *Expr, const DILocation *DI,
                                DPValue &InsertBefore);
  static DPValue *createDPVDeclare(Value *Address, DILocalVariable *DV,
                                   DIExpression *Expr, const DILocation *DI);
  static DPValue *createDPVDeclare(Value *Address, DILocalVariable *DV,
                                   DIExpression *Expr, const DILocation *DI,
                                   DPValue &InsertBefore);

  /// Iterator for ValueAsMetadata that internally uses direct pointer iteration
  /// over either a ValueAsMetadata* or a ValueAsMetadata**, dereferencing to the
  /// ValueAsMetadata .
  class location_op_iterator
      : public iterator_facade_base<location_op_iterator,
                                    std::bidirectional_iterator_tag, Value *> {
    PointerUnion<ValueAsMetadata *, ValueAsMetadata **> I;

  public:
    location_op_iterator(ValueAsMetadata *SingleIter) : I(SingleIter) {}
    location_op_iterator(ValueAsMetadata **MultiIter) : I(MultiIter) {}

    location_op_iterator(const location_op_iterator &R) : I(R.I) {}
    location_op_iterator &operator=(const location_op_iterator &R) {
      I = R.I;
      return *this;
    }
    bool operator==(const location_op_iterator &RHS) const {
      return I == RHS.I;
    }
    const Value *operator*() const {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    };
    Value *operator*() {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    }
    location_op_iterator &operator++() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() + 1;
      else
        I = I.get<ValueAsMetadata **>() + 1;
      return *this;
    }
    location_op_iterator &operator--() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() - 1;
      else
        I = I.get<ValueAsMetadata **>() - 1;
      return *this;
    }
  };

  bool isDbgDeclare() { return Type == LocationType::Declare; }
  bool isDbgValue() { return Type == LocationType::Value; }

  /// Get the locations corresponding to the variable referenced by the debug
  /// info intrinsic.  Depending on the intrinsic, this could be the
  /// variable's value or its address.
  iterator_range<location_op_iterator> location_ops() const;

  Value *getVariableLocationOp(unsigned OpIdx) const;

  void replaceVariableLocationOp(Value *OldValue, Value *NewValue,
                                 bool AllowEmpty = false);
  void replaceVariableLocationOp(unsigned OpIdx, Value *NewValue);
  /// Adding a new location operand will always result in this intrinsic using
  /// an ArgList, and must always be accompanied by a new expression that uses
  /// the new operand.
  void addVariableLocationOps(ArrayRef<Value *> NewValues,
                              DIExpression *NewExpr);

  void setVariable(DILocalVariable *NewVar) { Variable = NewVar; }

  void setExpression(DIExpression *NewExpr) { Expression = NewExpr; }

  unsigned getNumVariableLocationOps() const;

  bool hasArgList() const { return isa<DIArgList>(getRawLocation()); }
  /// Returns true if this DPValue has no empty MDNodes in its location list.
  bool hasValidLocation() const { return getVariableLocationOp(0) != nullptr; }

  /// Does this describe the address of a local variable. True for dbg.addr
  /// and dbg.declare, but not dbg.value, which describes its value.
  bool isAddressOfVariable() const { return Type == LocationType::Declare; }
  LocationType getType() const { return Type; }

  DebugLoc getDebugLoc() const { return DbgLoc; }
  void setDebugLoc(DebugLoc Loc) { DbgLoc = std::move(Loc); }

  void setKillLocation();
  bool isKillLocation() const;

  DILocalVariable *getVariable() const { return Variable; }

  DIExpression *getExpression() const { return Expression; }

  /// Returns the metadata operand for the first location description. i.e.,
  /// dbg intrinsic dbg.value,declare operand and dbg.assign 1st location
  /// operand (the "value componenet"). Note the operand (singular) may be
  /// a DIArgList which is a list of values.
  Metadata *getRawLocation() const { return DebugValues[0]; }

  Value *getValue(unsigned OpIdx = 0) const {
    return getVariableLocationOp(OpIdx);
  }

  /// Use of this should generally be avoided; instead,
  /// replaceVariableLocationOp and addVariableLocationOps should be used where
  /// possible to avoid creating invalid state.
  void setRawLocation(Metadata *NewLocation) {
    assert(
        (isa<ValueAsMetadata>(NewLocation) || isa<DIArgList>(NewLocation) ||
         isa<MDNode>(NewLocation)) &&
        "Location for a DPValue must be either ValueAsMetadata or DIArgList");
    resetDebugValue(0, NewLocation);
  }

  /// Get the size (in bits) of the variable, or fragment of the variable that
  /// is described.
  std::optional<uint64_t> getFragmentSizeInBits() const;

  bool isEquivalentTo(const DPValue &Other) {
    return std::tie(Type, DebugValues, Variable, Expression, DbgLoc) ==
           std::tie(Other.Type, Other.DebugValues, Other.Variable,
                    Other.Expression, Other.DbgLoc);
  }
  // Matches the definition of the Instruction version, equivalent to above but
  // without checking DbgLoc.
  bool isIdenticalToWhenDefined(const DPValue &Other) {
    return std::tie(Type, DebugValues, Variable, Expression) ==
           std::tie(Other.Type, Other.DebugValues, Other.Variable,
                    Other.Expression);
  }

  /// @name DbgAssign Methods
  /// @{
  bool isDbgAssign() const { return getType() == LocationType::Assign; }

  Value *getAddress() const;
  Metadata *getRawAddress() const {
    return isDbgAssign() ? DebugValues[1] : DebugValues[0];
  }
  Metadata *getRawAssignID() const { return DebugValues[2]; }
  DIAssignID *getAssignID() const;
  DIExpression *getAddressExpression() const { return AddressExpression; }
  void setAddressExpression(DIExpression *NewExpr) {
    AddressExpression = NewExpr;
  }
  void setAssignId(DIAssignID *New);
  void setAddress(Value *V) { resetDebugValue(1, ValueAsMetadata::get(V)); }
  /// Kill the address component.
  void setKillAddress();
  /// Check whether this kills the address component. This doesn't take into
  /// account the position of the intrinsic, therefore a returned value of false
  /// does not guarentee the address is a valid location for the variable at the
  /// intrinsic's position in IR.
  bool isKillAddress() const;

  /// @}

  DPValue *clone() const;
  /// Convert this DPValue back into a dbg.value intrinsic.
  /// \p InsertBefore Optional position to insert this intrinsic.
  /// \returns A new dbg.value intrinsic representiung this DPValue.
  DbgVariableIntrinsic *createDebugIntrinsic(Module *M,
                                             Instruction *InsertBefore) const;
  void setMarker(DPMarker *M) { Marker = M; }

  DPMarker *getMarker() { return Marker; }
  const DPMarker *getMarker() const { return Marker; }

  BasicBlock *getBlock();
  const BasicBlock *getBlock() const;

  Function *getFunction();
  const Function *getFunction() const;

  Module *getModule();
  const Module *getModule() const;

  LLVMContext &getContext();
  const LLVMContext &getContext() const;

  /// Insert this DPValue prior to \p InsertBefore. Must not be called if this
  /// is already contained in a DPMarker.
  void insertBefore(DPValue *InsertBefore);
  void insertAfter(DPValue *InsertAfter);
  void moveBefore(DPValue *MoveBefore);
  void moveAfter(DPValue *MoveAfter);

  void print(raw_ostream &O, bool IsForDebug = false) const;
  void print(raw_ostream &ROS, ModuleSlotTracker &MST, bool IsForDebug) const;
};

/// Per-instruction record of debug-info. If an Instruction is the position of
/// some debugging information, it points at a DPMarker storing that info. Each
/// marker points back at the instruction that owns it. Various utilities are
/// provided for manipulating the DPValues contained within this marker.
///
/// This class has a rough surface area, because it's needed to preserve the one
/// arefact that we can't yet eliminate from the intrinsic / dbg.value
/// debug-info design: the order of DPValues/records is significant, and
/// duplicates can exist. Thus, if one has a run of debug-info records such as:
///    dbg.value(...
///    %foo = barinst
///    dbg.value(...
/// and remove barinst, then the dbg.values must be preserved in the correct
/// order. Hence, the use of iterators to select positions to insert things
/// into, or the occasional InsertAtHead parameter indicating that new records
/// should go at the start of the list.
///
/// There are only five or six places in LLVM that truly rely on this ordering,
/// which we can improve in the future. Additionally, many improvements in the
/// way that debug-info is stored can be achieved in this class, at a future
/// date.
class DPMarker {
public:
  DPMarker() {}
  /// Link back to the Instruction that owns this marker. Can be null during
  /// operations that move a marker from one instruction to another.
  Instruction *MarkedInstr = nullptr;

  /// List of DPValues, each recording a single variable assignment, the
  /// equivalent of a dbg.value intrinsic. There is a one-to-one relationship
  /// between each dbg.value in a block and each DPValue once the
  /// representation has been converted, and the ordering of DPValues is
  /// meaningful in the same was a dbg.values.
  simple_ilist<DPValue> StoredDPValues;
  bool empty() const { return StoredDPValues.empty(); }

  const BasicBlock *getParent() const;
  BasicBlock *getParent();

  /// Handle the removal of a marker: the position of debug-info has gone away,
  /// but the stored debug records should not. Drop them onto the next
  /// instruction, or otherwise work out what to do with them.
  void removeMarker();
  void dump() const;

  void removeFromParent();
  void eraseFromParent();

  /// Implement operator<< on DPMarker.
  void print(raw_ostream &O, bool IsForDebug = false) const;
  void print(raw_ostream &ROS, ModuleSlotTracker &MST, bool IsForDebug) const;

  /// Produce a range over all the DPValues in this Marker.
  iterator_range<simple_ilist<DPValue>::iterator> getDbgValueRange();
  iterator_range<simple_ilist<DPValue>::const_iterator>
  getDbgValueRange() const;
  /// Transfer any DPValues from \p Src into this DPMarker. If \p InsertAtHead
  /// is true, place them before existing DPValues, otherwise afterwards.
  void absorbDebugValues(DPMarker &Src, bool InsertAtHead);
  /// Transfer the DPValues in \p Range from \p Src into this DPMarker. If
  /// \p InsertAtHead is true, place them before existing DPValues, otherwise
  // afterwards.
  void absorbDebugValues(iterator_range<DPValue::self_iterator> Range,
                         DPMarker &Src, bool InsertAtHead);
  /// Insert a DPValue into this DPMarker, at the end of the list. If
  /// \p InsertAtHead is true, at the start.
  void insertDPValue(DPValue *New, bool InsertAtHead);
  /// Insert a DPValue prior to a DPValue contained within this marker.
  void insertDPValue(DPValue *New, DPValue *InsertBefore);
  /// Insert a DPValue after a DPValue contained within this marker.
  void insertDPValueAfter(DPValue *New, DPValue *InsertAfter);
  /// Clone all DPMarkers from \p From into this marker. There are numerous
  /// options to customise the source/destination, due to gnarliness, see class
  /// comment.
  /// \p FromHere If non-null, copy from FromHere to the end of From's DPValues
  /// \p InsertAtHead Place the cloned DPValues at the start of StoredDPValues
  /// \returns Range over all the newly cloned DPValues
  iterator_range<simple_ilist<DPValue>::iterator>
  cloneDebugInfoFrom(DPMarker *From,
                     std::optional<simple_ilist<DPValue>::iterator> FromHere,
                     bool InsertAtHead = false);
  /// Erase all DPValues in this DPMarker.
  void dropDPValues();
  /// Erase a single DPValue from this marker. In an ideal future, we would
  /// never erase an assignment in this way, but it's the equivalent to
  /// erasing a dbg.value from a block.
  void dropOneDPValue(DPValue *DPV);

  /// We generally act like all llvm Instructions have a range of DPValues
  /// attached to them, but in reality sometimes we don't allocate the DPMarker
  /// to save time and memory, but still have to return ranges of DPValues. When
  /// we need to describe such an unallocated DPValue range, use this static
  /// markers range instead. This will bite us if someone tries to insert a
  /// DPValue in that range, but they should be using the Official (TM) API for
  /// that.
  static DPMarker EmptyDPMarker;
  static iterator_range<simple_ilist<DPValue>::iterator> getEmptyDPValueRange(){
    return make_range(EmptyDPMarker.StoredDPValues.end(), EmptyDPMarker.StoredDPValues.end());
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const DPMarker &Marker) {
  Marker.print(OS);
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const DPValue &Value) {
  Value.print(OS);
  return OS;
}

/// Inline helper to return a range of DPValues attached to a marker. It needs
/// to be inlined as it's frequently called, but also come after the declaration
/// of DPMarker. Thus: it's pre-declared by users like Instruction, then an
/// inlineable body defined here.
inline iterator_range<simple_ilist<DPValue>::iterator>
getDbgValueRange(DPMarker *DbgMarker) {
  if (!DbgMarker)
    return DPMarker::getEmptyDPValueRange();
  return DbgMarker->getDbgValueRange();
}

} // namespace llvm

#endif // LLVM_IR_DEBUGPROGRAMINSTRUCTION_H
