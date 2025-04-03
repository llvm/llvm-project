#include "PEMCAsmInfo.h"

namespace llvm{

PEMCAsmInfo :: PEMCAsmInfo(const Triple &TT)
{
     // 设置目标特定的汇编信息,指定汇编的格式信息
     
     CodePointerSize = 4; // 假设目标是32位架构
     CalleeSaveStackSlotSize = 4;
     IsLittleEndian = true;
     StackGrowsUp = false;
     HasSubsectionsViaSymbols = true;
     SupportsDebugInformation = true;
     ExceptionsType = ExceptionHandling::DwarfCFI;
}

}