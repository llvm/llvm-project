//===-- DisassembleRequestHandler.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "RequestHandler.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

using namespace lldb_dap::protocol;

namespace lldb_dap {

/// Disassembles code stored at the provided location.
/// Clients should only call this request if the corresponding capability
/// `supportsDisassembleRequest` is true.
llvm::Expected<DisassembleResponseBody>
DisassembleRequestHandler::Run(const DisassembleArguments &args) const {
  std::vector<DisassembledInstruction> instructions;

  std::optional<lldb::addr_t> addr_opt =
      DecodeMemoryReference(args.memoryReference);
  if (!addr_opt.has_value())
    return llvm::make_error<DAPError>("Malformed memory reference: " +
                                      args.memoryReference);

  lldb::addr_t addr_ptr = *addr_opt;
  addr_ptr += args.instructionOffset.value_or(0);
  lldb::SBAddress addr(addr_ptr, dap.target);
  if (!addr.IsValid())
    return llvm::make_error<DAPError>(
        "Memory reference not found in the current binary.");

  std::string flavor_string;
  const auto target_triple = llvm::StringRef(dap.target.GetTriple());
  // This handles both 32 and 64bit x86 architecture. The logic is duplicated in
  // `CommandObjectDisassemble::CommandOptions::OptionParsingStarting`
  if (target_triple.starts_with("x86")) {
    const lldb::SBStructuredData flavor =
        dap.debugger.GetSetting("target.x86-disassembly-flavor");

    const size_t str_length = flavor.GetStringValue(nullptr, 0);
    if (str_length != 0) {
      flavor_string.resize(str_length + 1);
      flavor.GetStringValue(flavor_string.data(), flavor_string.length());
    }
  }

  lldb::SBInstructionList insts = dap.target.ReadInstructions(
      addr, args.instructionCount, flavor_string.c_str());

  if (!insts.IsValid())
    return llvm::make_error<DAPError>(
        "Failed to find instructions for memory address.");

  const bool resolve_symbols = args.resolveSymbols.value_or(false);
  const auto num_insts = insts.GetSize();
  for (size_t i = 0; i < num_insts; ++i) {
    lldb::SBInstruction inst = insts.GetInstructionAtIndex(i);
    auto addr = inst.GetAddress();
    const auto inst_addr = addr.GetLoadAddress(dap.target);
    const char *m = inst.GetMnemonic(dap.target);
    const char *o = inst.GetOperands(dap.target);
    const char *c = inst.GetComment(dap.target);
    auto d = inst.GetData(dap.target);

    std::string bytes;
    llvm::raw_string_ostream sb(bytes);
    for (unsigned i = 0; i < inst.GetByteSize(); i++) {
      lldb::SBError error;
      uint8_t b = d.GetUnsignedInt8(error, i);
      if (error.Success()) {
        sb << llvm::format("%2.2x ", b);
      }
    }

    DisassembledInstruction disassembled_inst;
    disassembled_inst.address = inst_addr;
    disassembled_inst.instructionBytes =
        bytes.size() > 0 ? bytes.substr(0, bytes.size() - 1) : "";

    std::string instruction;
    llvm::raw_string_ostream si(instruction);

    lldb::SBSymbol symbol = addr.GetSymbol();
    // Only add the symbol on the first line of the function.
    if (symbol.IsValid() && symbol.GetStartAddress() == addr) {
      // If we have a valid symbol, append it as a label prefix for the first
      // instruction. This is so you can see the start of a function/callsite
      // in the assembly, at the moment VS Code (1.80) does not visualize the
      // symbol associated with the assembly instruction.
      si << (symbol.GetMangledName() != nullptr ? symbol.GetMangledName()
                                                : symbol.GetName())
         << ": ";

      if (resolve_symbols)
        disassembled_inst.symbol = symbol.GetDisplayName();
    }

    si << llvm::formatv("{0,7} {1,12}", m, o);
    if (c && c[0]) {
      si << " ; " << c;
    }

    disassembled_inst.instruction = instruction;

    auto line_entry = addr.GetLineEntry();
    // If the line number is 0 then the entry represents a compiler generated
    // location.
    if (line_entry.GetStartAddress() == addr && line_entry.IsValid() &&
        line_entry.GetFileSpec().IsValid() && line_entry.GetLine() != 0) {
      auto source = CreateSource(line_entry);
      disassembled_inst.location = std::move(source);

      const auto line = line_entry.GetLine();
      if (line != 0 && line != LLDB_INVALID_LINE_NUMBER)
        disassembled_inst.line = line;

      const auto column = line_entry.GetColumn();
      if (column != 0 && column != LLDB_INVALID_COLUMN_NUMBER)
        disassembled_inst.column = column;

      auto end_line_entry = line_entry.GetEndAddress().GetLineEntry();
      if (end_line_entry.IsValid() &&
          end_line_entry.GetFileSpec() == line_entry.GetFileSpec()) {
        const auto end_line = end_line_entry.GetLine();
        if (end_line != 0 && end_line != LLDB_INVALID_LINE_NUMBER &&
            end_line != line) {
          disassembled_inst.endLine = end_line;

          const auto end_column = end_line_entry.GetColumn();
          if (end_column != 0 && end_column != LLDB_INVALID_COLUMN_NUMBER &&
              end_column != column)
            disassembled_inst.endColumn = end_column - 1;
        }
      }
    }

    instructions.push_back(std::move(disassembled_inst));
  }

  return DisassembleResponseBody{std::move(instructions)};
}

} // namespace lldb_dap
