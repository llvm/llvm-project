#ifndef LLVM_TOOLS_LLVM_MC_CFI_STATE_H
#define LLVM_TOOLS_LLVM_MC_CFI_STATE_H

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include <optional>
namespace llvm {

class UnwindInfoHistory {
public:
  using DWARFRegType = uint32_t;

  UnwindInfoHistory(MCContext *Context) : Context(Context) {};

  static std::optional<DWARFRegType>
  getReferenceRegisterForUnwindInfoOfRegister(
      const dwarf::UnwindTable::const_iterator &UnwindRow, DWARFRegType Reg);

  std::optional<dwarf::UnwindTable::const_iterator> getCurrentUnwindRow() const;
  void update(const MCCFIInstruction &CFIDirective);

private:
  MCContext *Context;
  dwarf::UnwindTable Table;

  std::optional<dwarf::CFIProgram> convert(MCCFIInstruction CFIDirective);
};
} // namespace llvm

#endif