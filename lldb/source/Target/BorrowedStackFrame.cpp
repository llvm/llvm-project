//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/BorrowedStackFrame.h"

using namespace lldb;
using namespace lldb_private;

char BorrowedStackFrame::ID;

BorrowedStackFrame::BorrowedStackFrame(
    StackFrameSP borrowed_frame_sp, uint32_t new_frame_index,
    std::optional<uint32_t> new_concrete_frame_index)
    : StackFrame(
          borrowed_frame_sp->GetThread(), new_frame_index,
          borrowed_frame_sp->GetConcreteFrameIndex(),
          borrowed_frame_sp->GetRegisterContextSP(),
          borrowed_frame_sp->GetStackID().GetPC(),
          borrowed_frame_sp->GetStackID().GetCallFrameAddressWithoutMetadata(),
          borrowed_frame_sp->m_behaves_like_zeroth_frame,
          &borrowed_frame_sp->GetSymbolContext(eSymbolContextEverything)),
      m_borrowed_frame_sp(borrowed_frame_sp),
      m_new_frame_index(new_frame_index) {
  if (new_concrete_frame_index)
    m_new_concrete_frame_index = *new_concrete_frame_index;
  else
    m_new_concrete_frame_index =
        IsInlined() ? LLDB_INVALID_FRAME_ID : new_frame_index;
}

uint32_t BorrowedStackFrame::GetFrameIndex() const { return m_new_frame_index; }

void BorrowedStackFrame::SetFrameIndex(uint32_t index) {
  m_new_frame_index = index;
}

uint32_t BorrowedStackFrame::GetConcreteFrameIndex() {
  // FIXME: We need to find where the concrete frame into which this frame was
  // inlined landed in the new stack frame list as that is the correct concrete
  // frame index in this
  // stack frame.
  return m_new_concrete_frame_index;
}

StackID &BorrowedStackFrame::GetStackID() {
  return m_borrowed_frame_sp->GetStackID();
}

const Address &BorrowedStackFrame::GetFrameCodeAddress() {
  return m_borrowed_frame_sp->GetFrameCodeAddress();
}

Address BorrowedStackFrame::GetFrameCodeAddressForSymbolication() {
  return m_borrowed_frame_sp->GetFrameCodeAddressForSymbolication();
}

bool BorrowedStackFrame::ChangePC(addr_t pc) {
  return m_borrowed_frame_sp->ChangePC(pc);
}

const SymbolContext &
BorrowedStackFrame::GetSymbolContext(SymbolContextItem resolve_scope) {
  return m_borrowed_frame_sp->GetSymbolContext(resolve_scope);
}

llvm::Error BorrowedStackFrame::GetFrameBaseValue(Scalar &value) {
  return m_borrowed_frame_sp->GetFrameBaseValue(value);
}

DWARFExpressionList *
BorrowedStackFrame::GetFrameBaseExpression(Status *error_ptr) {
  return m_borrowed_frame_sp->GetFrameBaseExpression(error_ptr);
}

Block *BorrowedStackFrame::GetFrameBlock() {
  return m_borrowed_frame_sp->GetFrameBlock();
}

RegisterContextSP BorrowedStackFrame::GetRegisterContext() {
  return m_borrowed_frame_sp->GetRegisterContext();
}

VariableList *BorrowedStackFrame::GetVariableList(bool get_file_globals,
                                                  Status *error_ptr) {
  return m_borrowed_frame_sp->GetVariableList(get_file_globals, error_ptr);
}

VariableListSP
BorrowedStackFrame::GetInScopeVariableList(bool get_file_globals,
                                           bool must_have_valid_location) {
  return m_borrowed_frame_sp->GetInScopeVariableList(get_file_globals,
                                                     must_have_valid_location);
}

ValueObjectSP BorrowedStackFrame::GetValueForVariableExpressionPath(
    llvm::StringRef var_expr, DynamicValueType use_dynamic, uint32_t options,
    VariableSP &var_sp, Status &error, lldb::DILMode mode) {
  return m_borrowed_frame_sp->GetValueForVariableExpressionPath(
      var_expr, use_dynamic, options, var_sp, error, mode);
}

bool BorrowedStackFrame::HasDebugInformation() {
  return m_borrowed_frame_sp->HasDebugInformation();
}

const char *BorrowedStackFrame::Disassemble() {
  return m_borrowed_frame_sp->Disassemble();
}

ValueObjectSP BorrowedStackFrame::GetValueObjectForFrameVariable(
    const VariableSP &variable_sp, DynamicValueType use_dynamic) {
  return m_borrowed_frame_sp->GetValueObjectForFrameVariable(variable_sp,
                                                             use_dynamic);
}

bool BorrowedStackFrame::IsInlined() {
  return m_borrowed_frame_sp->IsInlined();
}

bool BorrowedStackFrame::IsSynthetic() const {
  return m_borrowed_frame_sp->IsSynthetic();
}

bool BorrowedStackFrame::IsHistorical() const {
  return m_borrowed_frame_sp->IsHistorical();
}

bool BorrowedStackFrame::IsArtificial() const {
  return m_borrowed_frame_sp->IsArtificial();
}

bool BorrowedStackFrame::IsHidden() { return m_borrowed_frame_sp->IsHidden(); }

const char *BorrowedStackFrame::GetFunctionName() {
  return m_borrowed_frame_sp->GetFunctionName();
}

const char *BorrowedStackFrame::GetDisplayFunctionName() {
  return m_borrowed_frame_sp->GetDisplayFunctionName();
}

ValueObjectSP BorrowedStackFrame::FindVariable(ConstString name) {
  return m_borrowed_frame_sp->FindVariable(name);
}

SourceLanguage BorrowedStackFrame::GetLanguage() {
  return m_borrowed_frame_sp->GetLanguage();
}

SourceLanguage BorrowedStackFrame::GuessLanguage() {
  return m_borrowed_frame_sp->GuessLanguage();
}

ValueObjectSP BorrowedStackFrame::GuessValueForAddress(addr_t addr) {
  return m_borrowed_frame_sp->GuessValueForAddress(addr);
}

ValueObjectSP
BorrowedStackFrame::GuessValueForRegisterAndOffset(ConstString reg,
                                                   int64_t offset) {
  return m_borrowed_frame_sp->GuessValueForRegisterAndOffset(reg, offset);
}

StructuredData::ObjectSP BorrowedStackFrame::GetLanguageSpecificData() {
  return m_borrowed_frame_sp->GetLanguageSpecificData();
}

RecognizedStackFrameSP BorrowedStackFrame::GetRecognizedFrame() {
  return m_borrowed_frame_sp->GetRecognizedFrame();
}

StackFrameSP BorrowedStackFrame::GetBorrowedFrame() const {
  return m_borrowed_frame_sp;
}

bool BorrowedStackFrame::isA(const void *ClassID) const {
  return ClassID == &ID || StackFrame::isA(ClassID);
}

bool BorrowedStackFrame::classof(const StackFrame *obj) {
  return obj->isA(&ID);
}
