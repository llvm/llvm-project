//===-- llvm/CodeGen/JITCodeEmitter.h - Code emission ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstract interface that is used by the machine code
// emission framework to output the code.  This allows machine code emission to
// be separated from concerns such as resolution of call targets, and where the
// machine code will be written (memory or disk, f.e.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_JITCODEEMITTER_H
#define LLVM_CODEGEN_JITCODEEMITTER_H

#include <string>
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class MachineBasicBlock;
class MachineConstantPool;
class MachineJumpTableInfo;
class MachineFunction;
class MachineModuleInfo;
class MachineRelocation;
class Value;
class GlobalValue;
class Function;
  
/// JITCodeEmitter - This class defines two sorts of methods: those for
/// emitting the actual bytes of machine code, and those for emitting auxiliary
/// structures, such as jump tables, relocations, etc.
///
/// Emission of machine code is complicated by the fact that we don't (in
/// general) know the size of the machine code that we're about to emit before
/// we emit it.  As such, we preallocate a certain amount of memory, and set the
/// BufferBegin/BufferEnd pointers to the start and end of the buffer.  As we
/// emit machine instructions, we advance the CurBufferPtr to indicate the
/// location of the next byte to emit.  In the case of a buffer overflow (we
/// need to emit more machine code than we have allocated space for), the
/// CurBufferPtr will saturate to BufferEnd and ignore stores.  Once the entire
/// function has been emitted, the overflow condition is checked, and if it has
/// occurred, more memory is allocated, and we reemit the code into it.
/// 
class JITCodeEmitter : public MachineCodeEmitter {
  virtual void anchor();
public:
  virtual ~JITCodeEmitter() {}

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.  This initializes the BufferBegin/End/Ptr
  /// fields.
  ///
  virtual void startFunction(MachineFunction &F) = 0;

  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.  If a buffer overflow has occurred, this method
  /// returns true (the callee is required to try again), otherwise it returns
  /// false.
  ///
  virtual bool finishFunction(MachineFunction &F) = 0;
  
  /// allocIndirectGV - Allocates and fills storage for an indirect
  /// GlobalValue, and returns the address.
  virtual void *allocIndirectGV(const GlobalValue *GV,
                                const uint8_t *Buffer, size_t Size,
                                unsigned Alignment) = 0;

  /// emitByte - This callback is invoked when a byte needs to be written to the
  /// output stream.
  ///
  void emitByte(uint8_t B) {
    if (CurBufferPtr != BufferEnd)
      *CurBufferPtr++ = B;
  }

  /// emitWordLE - This callback is invoked when a 32-bit word needs to be
  /// written to the output stream in little-endian format.
  ///
  void emitWordLE(uint32_t W) {
    if (4 <= BufferEnd-CurBufferPtr) {
      *CurBufferPtr++ = (uint8_t)(W >>  0);
      *CurBufferPtr++ = (uint8_t)(W >>  8);
      *CurBufferPtr++ = (uint8_t)(W >> 16);
      *CurBufferPtr++ = (uint8_t)(W >> 24);
    } else {
      CurBufferPtr = BufferEnd;
    }
  }
  
  /// emitWordBE - This callback is invoked when a 32-bit word needs to be
  /// written to the output stream in big-endian format.
  ///
  void emitWordBE(uint32_t W) {
    if (4 <= BufferEnd-CurBufferPtr) {
      *CurBufferPtr++ = (uint8_t)(W >> 24);
      *CurBufferPtr++ = (uint8_t)(W >> 16);
      *CurBufferPtr++ = (uint8_t)(W >>  8);
      *CurBufferPtr++ = (uint8_t)(W >>  0);
    } else {
      CurBufferPtr = BufferEnd;
    }
  }

  /// emitDWordLE - This callback is invoked when a 64-bit word needs to be
  /// written to the output stream in little-endian format.
  ///
  void emitDWordLE(uint64_t W) {
    if (8 <= BufferEnd-CurBufferPtr) {
      *CurBufferPtr++ = (uint8_t)(W >>  0);
      *CurBufferPtr++ = (uint8_t)(W >>  8);
      *CurBufferPtr++ = (uint8_t)(W >> 16);
      *CurBufferPtr++ = (uint8_t)(W >> 24);
      *CurBufferPtr++ = (uint8_t)(W >> 32);
      *CurBufferPtr++ = (uint8_t)(W >> 40);
      *CurBufferPtr++ = (uint8_t)(W >> 48);
      *CurBufferPtr++ = (uint8_t)(W >> 56);
    } else {
      CurBufferPtr = BufferEnd;
    }
  }
  
  /// emitDWordBE - This callback is invoked when a 64-bit word needs to be
  /// written to the output stream in big-endian format.
  ///
  void emitDWordBE(uint64_t W) {
    if (8 <= BufferEnd-CurBufferPtr) {
      *CurBufferPtr++ = (uint8_t)(W >> 56);
      *CurBufferPtr++ = (uint8_t)(W >> 48);
      *CurBufferPtr++ = (uint8_t)(W >> 40);
      *CurBufferPtr++ = (uint8_t)(W >> 32);
      *CurBufferPtr++ = (uint8_t)(W >> 24);
      *CurBufferPtr++ = (uint8_t)(W >> 16);
      *CurBufferPtr++ = (uint8_t)(W >>  8);
      *CurBufferPtr++ = (uint8_t)(W >>  0);
    } else {
      CurBufferPtr = BufferEnd;
    }
  }

  /// emitAlignment - Move the CurBufferPtr pointer up to the specified
  /// alignment (saturated to BufferEnd of course).
  void emitAlignment(unsigned Alignment) {
    if (Alignment == 0) Alignment = 1;
    uint8_t *NewPtr = (uint8_t*)RoundUpToAlignment((uintptr_t)CurBufferPtr,
                                                   Alignment);
    CurBufferPtr = std::min(NewPtr, BufferEnd);
  }

  /// emitAlignmentWithFill - Similar to emitAlignment, except that the
  /// extra bytes are filled with the provided byte.
  void emitAlignmentWithFill(unsigned Alignment, uint8_t Fill) {
    if (Alignment == 0) Alignment = 1;
    uint8_t *NewPtr = (uint8_t*)RoundUpToAlignment((uintptr_t)CurBufferPtr,
                                                   Alignment);
    // Fail if we don't have room.
    if (NewPtr > BufferEnd) {
      CurBufferPtr = BufferEnd;
      return;
    }
    while (CurBufferPtr < NewPtr) {
      *CurBufferPtr++ = Fill;
    }
  }

  /// emitULEB128Bytes - This callback is invoked when a ULEB128 needs to be
  /// written to the output stream.
  void emitULEB128Bytes(uint64_t Value, unsigned PadTo = 0) {
    do {
      uint8_t Byte = Value & 0x7f;
      Value >>= 7;
      if (Value || PadTo != 0) Byte |= 0x80;
      emitByte(Byte);
    } while (Value);

    if (PadTo) {
      do {
        uint8_t Byte = (PadTo > 1) ? 0x80 : 0x0;
        emitByte(Byte);
      } while (--PadTo);
    }
  }
  
  /// emitSLEB128Bytes - This callback is invoked when a SLEB128 needs to be
  /// written to the output stream.
  void emitSLEB128Bytes(int64_t Value) {
    int32_t Sign = Value >> (8 * sizeof(Value) - 1);
    bool IsMore;
  
    do {
      uint8_t Byte = Value & 0x7f;
      Value >>= 7;
      IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
      if (IsMore) Byte |= 0x80;
      emitByte(Byte);
    } while (IsMore);
  }

  /// emitString - This callback is invoked when a String needs to be
  /// written to the output stream.
  void emitString(const std::string &String) {
    for (unsigned i = 0, N = static_cast<unsigned>(String.size());
         i < N; ++i) {
      uint8_t C = String[i];
      emitByte(C);
    }
    emitByte(0);
  }
  
  /// emitInt32 - Emit a int32 directive.
  void emitInt32(uint32_t Value) {
    if (4 <= BufferEnd-CurBufferPtr) {
      *((uint32_t*)CurBufferPtr) = Value;
      CurBufferPtr += 4;
    } else {
      CurBufferPtr = BufferEnd;
    }
  }

  /// emitInt64 - Emit a int64 directive.
  void emitInt64(uint64_t Value) {
    if (8 <= BufferEnd-CurBufferPtr) {
      *((uint64_t*)CurBufferPtr) = Value;
      CurBufferPtr += 8;
    } else {
      CurBufferPtr = BufferEnd;
    }
  }
  
  /// emitInt32At - Emit the Int32 Value in Addr.
  void emitInt32At(uintptr_t *Addr, uintptr_t Value) {
    if (Addr >= (uintptr_t*)BufferBegin && Addr < (uintptr_t*)BufferEnd)
      (*(uint32_t*)Addr) = (uint32_t)Value;
  }
  
  /// emitInt64At - Emit the Int64 Value in Addr.
  void emitInt64At(uintptr_t *Addr, uintptr_t Value) {
    if (Addr >= (uintptr_t*)BufferBegin && Addr < (uintptr_t*)BufferEnd)
      (*(uint64_t*)Addr) = (uint64_t)Value;
  }
  
  
  /// emitLabel - Emits a label
  virtual void emitLabel(MCSymbol *Label) = 0;

  /// allocateSpace - Allocate a block of space in the current output buffer,
  /// returning null (and setting conditions to indicate buffer overflow) on
  /// failure.  Alignment is the alignment in bytes of the buffer desired.
  virtual void *allocateSpace(uintptr_t Size, unsigned Alignment) {
    emitAlignment(Alignment);
    void *Result;
    
    // Check for buffer overflow.
    if (Size >= (uintptr_t)(BufferEnd-CurBufferPtr)) {
      CurBufferPtr = BufferEnd;
      Result = 0;
    } else {
      // Allocate the space.
      Result = CurBufferPtr;
      CurBufferPtr += Size;
    }
    
    return Result;
  }

  /// allocateGlobal - Allocate memory for a global.  Unlike allocateSpace,
  /// this method does not allocate memory in the current output buffer,
  /// because a global may live longer than the current function.
  virtual void *allocateGlobal(uintptr_t Size, unsigned Alignment) = 0;

  /// StartMachineBasicBlock - This should be called by the target when a new
  /// basic block is about to be emitted.  This way the MCE knows where the
  /// start of the block is, and can implement getMachineBasicBlockAddress.
  virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) = 0;
  
  /// getCurrentPCValue - This returns the address that the next emitted byte
  /// will be output to.
  ///
  virtual uintptr_t getCurrentPCValue() const {
    return (uintptr_t)CurBufferPtr;
  }

  /// getCurrentPCOffset - Return the offset from the start of the emitted
  /// buffer that we are currently writing to.
  uintptr_t getCurrentPCOffset() const {
    return CurBufferPtr-BufferBegin;
  }

  /// earlyResolveAddresses - True if the code emitter can use symbol addresses 
  /// during code emission time. The JIT is capable of doing this because it
  /// creates jump tables or constant pools in memory on the fly while the
  /// object code emitters rely on a linker to have real addresses and should
  /// use relocations instead.
  bool earlyResolveAddresses() const { return true; }

  /// addRelocation - Whenever a relocatable address is needed, it should be
  /// noted with this interface.
  virtual void addRelocation(const MachineRelocation &MR) = 0;
  
  /// FIXME: These should all be handled with relocations!
  
  /// getConstantPoolEntryAddress - Return the address of the 'Index' entry in
  /// the constant pool that was last emitted with the emitConstantPool method.
  ///
  virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const = 0;

  /// getJumpTableEntryAddress - Return the address of the jump table with index
  /// 'Index' in the function that last called initJumpTableInfo.
  ///
  virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const = 0;
  
  /// getMachineBasicBlockAddress - Return the address of the specified
  /// MachineBasicBlock, only usable after the label for the MBB has been
  /// emitted.
  ///
  virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const= 0;

  /// getLabelAddress - Return the address of the specified Label, only usable
  /// after the Label has been emitted.
  ///
  virtual uintptr_t getLabelAddress(MCSymbol *Label) const = 0;
  
  /// Specifies the MachineModuleInfo object. This is used for exception handling
  /// purposes.
  virtual void setModuleInfo(MachineModuleInfo* Info) = 0;

  /// getLabelLocations - Return the label locations map of the label IDs to
  /// their address.
  virtual DenseMap<MCSymbol*, uintptr_t> *getLabelLocations() { return 0; }
};

} // End llvm namespace

#endif
