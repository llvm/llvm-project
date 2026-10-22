// RUN: llvm-tblgen -gen-instr-info -I %p/../../include -I %p/../../lib/Target/AMDGPU -I %p/../../lib/Target %s | FileCheck %s

include "AMDGPU.td"

// CHECK-LABEL: #ifdef GET_INSTRINFO_OPERAND_ENUM
// CHECK: namespace llvm::AMDGPU {
// CHECK: enum class OpName : uint8_t {
// CHECK: callee =
// CHECK: LLVM_READONLY int16_t getNamedOperandIdx(uint32_t Opcode, OpName Name);
// CHECK: LLVM_READONLY OpName getOperandIdxName(uint32_t Opcode, int16_t Idx);

// CHECK-LABEL: #ifdef GET_INSTRINFO_NAMED_OPS
// CHECK: LLVM_READONLY OpName getOperandIdxName(uint32_t Opcode, int16_t Idx) {
// CHECK: static constexpr OpName OperandMap[][24] = {
// CHECK: {OpName::src0, OpName::callee, OpName::NUM_OPERAND_NAMES, OpName::NUM_OPERAND_NAMES,
