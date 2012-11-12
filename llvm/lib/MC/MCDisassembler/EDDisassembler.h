//===-- EDDisassembler.h - LLVM Enhanced Disassembler -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Enhanced Disassembly library's
// disassembler class.  The disassembler is responsible for vending individual
// instructions according to a given architecture and disassembly syntax.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDDISASSEMBLER_H
#define LLVM_EDDISASSEMBLER_H

#include "EDInfo.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mutex.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class AsmLexer;
class AsmParser;
class AsmToken;
class MCContext;
class MCAsmInfo;
class MCAsmLexer;
class MCDisassembler;
class MCInst;
class MCInstPrinter;
class MCInstrInfo;
class MCParsedAsmOperand;
class MCRegisterInfo;
class MCStreamer;
class MCSubtargetInfo;
class MCTargetAsmLexer;
class MCTargetAsmParser;
template <typename T> class SmallVectorImpl;
class SourceMgr;
class Target;

struct EDInstInfo;
struct EDInst;
struct EDOperand;
struct EDToken;

typedef int (*EDByteReaderCallback)(uint8_t *byte, uint64_t address, void *arg);

/// EDDisassembler - Encapsulates a disassembler for a single architecture and
///   disassembly syntax.  Also manages the static disassembler registry.
struct EDDisassembler {
  typedef enum {
    /*! @constant kEDAssemblySyntaxX86Intel Intel syntax for i386 and x86_64. */
    kEDAssemblySyntaxX86Intel  = 0,
    /*! @constant kEDAssemblySyntaxX86ATT AT&T syntax for i386 and x86_64. */
    kEDAssemblySyntaxX86ATT    = 1,
    kEDAssemblySyntaxARMUAL    = 2
  } AssemblySyntax;
  
  
  ////////////////////
  // Static members //
  ////////////////////
  
  /// CPUKey - Encapsulates the descriptor of an architecture/disassembly-syntax
  ///   pair
  struct CPUKey {
    /// The architecture type
    std::string Triple;
    
    /// The assembly syntax
    AssemblySyntax Syntax;
    
    /// operator== - Equality operator
    bool operator==(const CPUKey &key) const {
      return (Triple == key.Triple &&
              Syntax == key.Syntax);
    }
    
    /// operator< - Less-than operator
    bool operator<(const CPUKey &key) const {
      return ((Triple < key.Triple) ||
              ((Triple == key.Triple) && Syntax < (key.Syntax)));
    }
  };
  
  typedef std::map<CPUKey, EDDisassembler*> DisassemblerMap_t;
  
  /// A map from disassembler specifications to disassemblers.  Populated
  ///   lazily.
  static DisassemblerMap_t sDisassemblers;

  /// getDisassembler - Returns the specified disassemble, or NULL on failure
  ///
  /// @arg arch   - The desired architecture
  /// @arg syntax - The desired disassembly syntax
  static EDDisassembler *getDisassembler(llvm::Triple::ArchType arch,
                                         AssemblySyntax syntax);
  
  /// getDisassembler - Returns the disassembler for a given combination of
  ///   CPU type, CPU subtype, and assembly syntax, or NULL on failure
  ///
  /// @arg str    - The string representation of the architecture triple, e.g.,
  ///               "x86_64-apple-darwin"
  /// @arg syntax - The disassembly syntax for the required disassembler
  static EDDisassembler *getDisassembler(llvm::StringRef str,
                                         AssemblySyntax syntax);
  
  ////////////////////////
  // Per-object members //
  ////////////////////////
  
  /// True only if the object has been successfully initialized
  bool Valid;
  /// True if the disassembler can provide semantic information
  bool HasSemantics;
  
  /// The stream to write errors to
  llvm::raw_ostream &ErrorStream;

  /// The triple/syntax pair for the current architecture
  CPUKey Key;
  /// The Triple fur the current architecture
  Triple TgtTriple;
  /// The LLVM target corresponding to the disassembler
  const llvm::Target *Tgt;
  /// The assembly information for the target architecture
  llvm::OwningPtr<const llvm::MCAsmInfo> AsmInfo;
  /// The subtarget information for the target architecture
  llvm::OwningPtr<const llvm::MCSubtargetInfo> STI;
  // The instruction information for the target architecture.
  llvm::OwningPtr<const llvm::MCInstrInfo> MII;
  // The register information for the target architecture.
  llvm::OwningPtr<const llvm::MCRegisterInfo> MRI;
  /// The disassembler for the target architecture
  llvm::OwningPtr<const llvm::MCDisassembler> Disassembler;
  /// The output string for the instruction printer; must be guarded with 
  ///   PrinterMutex
  llvm::OwningPtr<std::string> InstString;
  /// The output stream for the disassembler; must be guarded with
  ///   PrinterMutex
  llvm::OwningPtr<llvm::raw_string_ostream> InstStream;
  /// The instruction printer for the target architecture; must be guarded with
  ///   PrinterMutex when printing
  llvm::OwningPtr<llvm::MCInstPrinter> InstPrinter;
  /// The mutex that guards the instruction printer's printing functions, which
  ///   use a shared stream
  llvm::sys::Mutex PrinterMutex;
  /// The array of instruction information provided by the TableGen backend for
  ///   the target architecture
  const llvm::EDInstInfo *InstInfos;
  /// The target-specific lexer for use in tokenizing strings, in
  ///   target-independent and target-specific portions
  llvm::OwningPtr<llvm::AsmLexer> GenericAsmLexer;
  llvm::OwningPtr<llvm::MCTargetAsmLexer> SpecificAsmLexer;
  /// The guard for the above
  llvm::sys::Mutex ParserMutex;
  /// The LLVM number used for the target disassembly syntax variant
  int LLVMSyntaxVariant;
    
  typedef std::vector<std::string> regvec_t;
  typedef std::map<std::string, unsigned> regrmap_t;
  
  /// A vector of registers for quick mapping from LLVM register IDs to names
  regvec_t RegVec;
  /// A map of registers for quick mapping from register names to LLVM IDs
  regrmap_t RegRMap;
  
  /// A set of register IDs for aliases of the stack pointer for the current
  ///   architecture
  std::set<unsigned> stackPointers;
  /// A set of register IDs for aliases of the program counter for the current
  ///   architecture
  std::set<unsigned> programCounters;
  
  /// Constructor - initializes a disassembler with all the necessary objects,
  ///   which come pre-allocated from the registry accessor function
  ///
  /// @arg key                - the architecture and disassembly syntax for the 
  ///                           disassembler
  EDDisassembler(CPUKey& key);
  
  /// valid - reports whether there was a failure in the constructor.
  bool valid() {
    return Valid;
  }
  
  /// hasSemantics - reports whether the disassembler can provide operands and
  ///   tokens.
  bool hasSemantics() {
    return HasSemantics;
  }
  
  ~EDDisassembler();
  
  /// createInst - creates and returns an instruction given a callback and
  ///   memory address, or NULL on failure
  ///
  /// @arg byteReader - A callback function that provides machine code bytes
  /// @arg address    - The address of the first byte of the instruction,
  ///                   suitable for passing to byteReader
  /// @arg arg        - An opaque argument for byteReader
  EDInst *createInst(EDByteReaderCallback byteReader, 
                     uint64_t address, 
                     void *arg);

  /// initMaps - initializes regVec and regRMap using the provided register
  ///   info
  ///
  /// @arg registerInfo - the register information to use as a source
  void initMaps(const llvm::MCRegisterInfo &registerInfo);
  /// nameWithRegisterID - Returns the name (owned by the EDDisassembler) of a 
  ///   register for a given register ID, or NULL on failure
  ///
  /// @arg registerID - the ID of the register to be queried
  const char *nameWithRegisterID(unsigned registerID) const;
  /// registerIDWithName - Returns the ID of a register for a given register
  ///   name, or (unsigned)-1 on failure
  ///
  /// @arg name - The name of the register
  unsigned registerIDWithName(const char *name) const;
  
  /// registerIsStackPointer - reports whether a register ID is an alias for the
  ///   stack pointer register
  ///
  /// @arg registerID - The LLVM register ID
  bool registerIsStackPointer(unsigned registerID);
  /// registerIsStackPointer - reports whether a register ID is an alias for the
  ///   stack pointer register
  ///
  /// @arg registerID - The LLVM register ID
  bool registerIsProgramCounter(unsigned registerID);
  
  /// printInst - prints an MCInst to a string, returning 0 on success, or -1
  ///   otherwise
  ///
  /// @arg str  - A reference to a string which is filled in with the string
  ///             representation of the instruction
  /// @arg inst - A reference to the MCInst to be printed
  int printInst(std::string& str,
                llvm::MCInst& inst);
  
  /// parseInst - extracts operands and tokens from a string for use in
  ///   tokenizing the string.  Returns 0 on success, or -1 otherwise.
  ///
  /// @arg operands - A reference to a vector that will be filled in with the
  ///                 parsed operands
  /// @arg tokens   - A reference to a vector that will be filled in with the
  ///                 tokens
  /// @arg str      - The string representation of the instruction
  int parseInst(llvm::SmallVectorImpl<llvm::MCParsedAsmOperand*> &operands,
                llvm::SmallVectorImpl<llvm::AsmToken> &tokens,
                const std::string &str);
  
  /// llvmSyntaxVariant - returns the LLVM syntax variant for this disassembler
  int llvmSyntaxVariant() const;  
};

} // end namespace llvm

#endif
