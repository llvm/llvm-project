#ifndef LLVM_TOOLS_LLVM_MC_CFI_STATE_H
#define LLVM_TOOLS_LLVM_MC_CFI_STATE_H

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include <cstdint>
#include <optional>
namespace llvm {

using DWARFRegType = uint32_t;

class UnwindInfoHistory {
public:
  UnwindInfoHistory(MCContext *Context) : Context(Context) {};

  std::optional<dwarf::UnwindTable::const_iterator> getCurrentUnwindRow() const;
  void update(const MCCFIInstruction &CFIDirective);

private:
  MCContext *Context;
  dwarf::UnwindTable Table;

  std::optional<dwarf::CFIProgram> convert(MCCFIInstruction CFIDirective);
};
} // namespace llvm

#endif