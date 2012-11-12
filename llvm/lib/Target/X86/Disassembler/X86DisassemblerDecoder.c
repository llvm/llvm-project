/*===-- X86DisassemblerDecoder.c - Disassembler decoder ------------*- C -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*
 *
 * This file is part of the X86 Disassembler.
 * It contains the implementation of the instruction decoder.
 * Documentation for the disassembler can be found in X86Disassembler.h.
 *
 *===----------------------------------------------------------------------===*/

#include <stdarg.h>   /* for va_*()       */
#include <stdio.h>    /* for vsnprintf()  */
#include <stdlib.h>   /* for exit()       */
#include <string.h>   /* for memset()     */

#include "X86DisassemblerDecoder.h"

#include "X86GenDisassemblerTables.inc"

#define TRUE  1
#define FALSE 0

typedef int8_t bool;

#ifndef NDEBUG
#define debug(s) do { x86DisassemblerDebug(__FILE__, __LINE__, s); } while (0)
#else
#define debug(s) do { } while (0)
#endif


/*
 * contextForAttrs - Client for the instruction context table.  Takes a set of
 *   attributes and returns the appropriate decode context.
 *
 * @param attrMask  - Attributes, from the enumeration attributeBits.
 * @return          - The InstructionContext to use when looking up an
 *                    an instruction with these attributes.
 */
static InstructionContext contextForAttrs(uint8_t attrMask) {
  return CONTEXTS_SYM[attrMask];
}

/*
 * modRMRequired - Reads the appropriate instruction table to determine whether
 *   the ModR/M byte is required to decode a particular instruction.
 *
 * @param type        - The opcode type (i.e., how many bytes it has).
 * @param insnContext - The context for the instruction, as returned by
 *                      contextForAttrs.
 * @param opcode      - The last byte of the instruction's opcode, not counting
 *                      ModR/M extensions and escapes.
 * @return            - TRUE if the ModR/M byte is required, FALSE otherwise.
 */
static int modRMRequired(OpcodeType type,
                         InstructionContext insnContext,
                         uint8_t opcode) {
  const struct ContextDecision* decision = 0;
  
  switch (type) {
  case ONEBYTE:
    decision = &ONEBYTE_SYM;
    break;
  case TWOBYTE:
    decision = &TWOBYTE_SYM;
    break;
  case THREEBYTE_38:
    decision = &THREEBYTE38_SYM;
    break;
  case THREEBYTE_3A:
    decision = &THREEBYTE3A_SYM;
    break;
  case THREEBYTE_A6:
    decision = &THREEBYTEA6_SYM;
    break;
  case THREEBYTE_A7:
    decision = &THREEBYTEA7_SYM;
    break;
  }

  return decision->opcodeDecisions[insnContext].modRMDecisions[opcode].
    modrm_type != MODRM_ONEENTRY;
}

/*
 * decode - Reads the appropriate instruction table to obtain the unique ID of
 *   an instruction.
 *
 * @param type        - See modRMRequired().
 * @param insnContext - See modRMRequired().
 * @param opcode      - See modRMRequired().
 * @param modRM       - The ModR/M byte if required, or any value if not.
 * @return            - The UID of the instruction, or 0 on failure.
 */
static InstrUID decode(OpcodeType type,
                       InstructionContext insnContext,
                       uint8_t opcode,
                       uint8_t modRM) {
  const struct ModRMDecision* dec = 0;
  
  switch (type) {
  case ONEBYTE:
    dec = &ONEBYTE_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case TWOBYTE:
    dec = &TWOBYTE_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_38:
    dec = &THREEBYTE38_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_3A:
    dec = &THREEBYTE3A_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_A6:
    dec = &THREEBYTEA6_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  case THREEBYTE_A7:
    dec = &THREEBYTEA7_SYM.opcodeDecisions[insnContext].modRMDecisions[opcode];
    break;
  }
  
  switch (dec->modrm_type) {
  default:
    debug("Corrupt table!  Unknown modrm_type");
    return 0;
  case MODRM_ONEENTRY:
    return modRMTable[dec->instructionIDs];
  case MODRM_SPLITRM:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs+1];
    return modRMTable[dec->instructionIDs];
  case MODRM_SPLITREG:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs+((modRM & 0x38) >> 3)+8];
    return modRMTable[dec->instructionIDs+((modRM & 0x38) >> 3)];
  case MODRM_SPLITMISC:
    if (modFromModRM(modRM) == 0x3)
      return modRMTable[dec->instructionIDs+(modRM & 0x3f)+8];
    return modRMTable[dec->instructionIDs+((modRM & 0x38) >> 3)];
  case MODRM_FULL:
    return modRMTable[dec->instructionIDs+modRM];
  }
}

/*
 * specifierForUID - Given a UID, returns the name and operand specification for
 *   that instruction.
 *
 * @param uid - The unique ID for the instruction.  This should be returned by
 *              decode(); specifierForUID will not check bounds.
 * @return    - A pointer to the specification for that instruction.
 */
static const struct InstructionSpecifier *specifierForUID(InstrUID uid) {
  return &INSTRUCTIONS_SYM[uid];
}

/*
 * consumeByte - Uses the reader function provided by the user to consume one
 *   byte from the instruction's memory and advance the cursor.
 *
 * @param insn  - The instruction with the reader function to use.  The cursor
 *                for this instruction is advanced.
 * @param byte  - A pointer to a pre-allocated memory buffer to be populated
 *                with the data read.
 * @return      - 0 if the read was successful; nonzero otherwise.
 */
static int consumeByte(struct InternalInstruction* insn, uint8_t* byte) {
  int ret = insn->reader(insn->readerArg, byte, insn->readerCursor);
  
  if (!ret)
    ++(insn->readerCursor);
  
  return ret;
}

/*
 * lookAtByte - Like consumeByte, but does not advance the cursor.
 *
 * @param insn  - See consumeByte().
 * @param byte  - See consumeByte().
 * @return      - See consumeByte().
 */
static int lookAtByte(struct InternalInstruction* insn, uint8_t* byte) {
  return insn->reader(insn->readerArg, byte, insn->readerCursor);
}

static void unconsumeByte(struct InternalInstruction* insn) {
  insn->readerCursor--;
}

#define CONSUME_FUNC(name, type)                                  \
  static int name(struct InternalInstruction* insn, type* ptr) {  \
    type combined = 0;                                            \
    unsigned offset;                                              \
    for (offset = 0; offset < sizeof(type); ++offset) {           \
      uint8_t byte;                                               \
      int ret = insn->reader(insn->readerArg,                     \
                             &byte,                               \
                             insn->readerCursor + offset);        \
      if (ret)                                                    \
        return ret;                                               \
      combined = combined | ((uint64_t)byte << (offset * 8));     \
    }                                                             \
    *ptr = combined;                                              \
    insn->readerCursor += sizeof(type);                           \
    return 0;                                                     \
  }

/*
 * consume* - Use the reader function provided by the user to consume data
 *   values of various sizes from the instruction's memory and advance the
 *   cursor appropriately.  These readers perform endian conversion.
 *
 * @param insn    - See consumeByte().
 * @param ptr     - A pointer to a pre-allocated memory of appropriate size to
 *                  be populated with the data read.
 * @return        - See consumeByte().
 */
CONSUME_FUNC(consumeInt8, int8_t)
CONSUME_FUNC(consumeInt16, int16_t)
CONSUME_FUNC(consumeInt32, int32_t)
CONSUME_FUNC(consumeUInt16, uint16_t)
CONSUME_FUNC(consumeUInt32, uint32_t)
CONSUME_FUNC(consumeUInt64, uint64_t)

/*
 * dbgprintf - Uses the logging function provided by the user to log a single
 *   message, typically without a carriage-return.
 *
 * @param insn    - The instruction containing the logging function.
 * @param format  - See printf().
 * @param ...     - See printf().
 */
static void dbgprintf(struct InternalInstruction* insn,
                      const char* format,
                      ...) {  
  char buffer[256];
  va_list ap;
  
  if (!insn->dlog)
    return;
    
  va_start(ap, format);
  (void)vsnprintf(buffer, sizeof(buffer), format, ap);
  va_end(ap);
  
  insn->dlog(insn->dlogArg, buffer);
  
  return;
}

/*
 * setPrefixPresent - Marks that a particular prefix is present at a particular
 *   location.
 *
 * @param insn      - The instruction to be marked as having the prefix.
 * @param prefix    - The prefix that is present.
 * @param location  - The location where the prefix is located (in the address
 *                    space of the instruction's reader).
 */
static void setPrefixPresent(struct InternalInstruction* insn,
                                    uint8_t prefix,
                                    uint64_t location)
{
  insn->prefixPresent[prefix] = 1;
  insn->prefixLocations[prefix] = location;
}

/*
 * isPrefixAtLocation - Queries an instruction to determine whether a prefix is
 *   present at a given location.
 *
 * @param insn      - The instruction to be queried.
 * @param prefix    - The prefix.
 * @param location  - The location to query.
 * @return          - Whether the prefix is at that location.
 */
static BOOL isPrefixAtLocation(struct InternalInstruction* insn,
                               uint8_t prefix,
                               uint64_t location)
{
  if (insn->prefixPresent[prefix] == 1 &&
     insn->prefixLocations[prefix] == location)
    return TRUE;
  else
    return FALSE;
}

/*
 * readPrefixes - Consumes all of an instruction's prefix bytes, and marks the
 *   instruction as having them.  Also sets the instruction's default operand,
 *   address, and other relevant data sizes to report operands correctly.
 *
 * @param insn  - The instruction whose prefixes are to be read.
 * @return      - 0 if the instruction could be read until the end of the prefix
 *                bytes, and no prefixes conflicted; nonzero otherwise.
 */
static int readPrefixes(struct InternalInstruction* insn) {
  BOOL isPrefix = TRUE;
  BOOL prefixGroups[4] = { FALSE };
  uint64_t prefixLocation;
  uint8_t byte = 0;
  
  BOOL hasAdSize = FALSE;
  BOOL hasOpSize = FALSE;
  
  dbgprintf(insn, "readPrefixes()");
    
  while (isPrefix) {
    prefixLocation = insn->readerCursor;
    
    if (consumeByte(insn, &byte))
      return -1;

    /*
     * If the first byte is a LOCK prefix break and let it be disassembled
     * as a lock "instruction", by creating an <MCInst #xxxx LOCK_PREFIX>.
     * FIXME there is currently no way to get the disassembler to print the
     * lock prefix if it is not the first byte.
     */
    if (insn->readerCursor - 1 == insn->startLocation && byte == 0xf0)
      break;
    
    switch (byte) {
    case 0xf0:  /* LOCK */
    case 0xf2:  /* REPNE/REPNZ */
    case 0xf3:  /* REP or REPE/REPZ */
      if (prefixGroups[0])
        dbgprintf(insn, "Redundant Group 1 prefix");
      prefixGroups[0] = TRUE;
      setPrefixPresent(insn, byte, prefixLocation);
      break;
    case 0x2e:  /* CS segment override -OR- Branch not taken */
    case 0x36:  /* SS segment override -OR- Branch taken */
    case 0x3e:  /* DS segment override */
    case 0x26:  /* ES segment override */
    case 0x64:  /* FS segment override */
    case 0x65:  /* GS segment override */
      switch (byte) {
      case 0x2e:
        insn->segmentOverride = SEG_OVERRIDE_CS;
        break;
      case 0x36:
        insn->segmentOverride = SEG_OVERRIDE_SS;
        break;
      case 0x3e:
        insn->segmentOverride = SEG_OVERRIDE_DS;
        break;
      case 0x26:
        insn->segmentOverride = SEG_OVERRIDE_ES;
        break;
      case 0x64:
        insn->segmentOverride = SEG_OVERRIDE_FS;
        break;
      case 0x65:
        insn->segmentOverride = SEG_OVERRIDE_GS;
        break;
      default:
        debug("Unhandled override");
        return -1;
      }
      if (prefixGroups[1])
        dbgprintf(insn, "Redundant Group 2 prefix");
      prefixGroups[1] = TRUE;
      setPrefixPresent(insn, byte, prefixLocation);
      break;
    case 0x66:  /* Operand-size override */
      if (prefixGroups[2])
        dbgprintf(insn, "Redundant Group 3 prefix");
      prefixGroups[2] = TRUE;
      hasOpSize = TRUE;
      setPrefixPresent(insn, byte, prefixLocation);
      break;
    case 0x67:  /* Address-size override */
      if (prefixGroups[3])
        dbgprintf(insn, "Redundant Group 4 prefix");
      prefixGroups[3] = TRUE;
      hasAdSize = TRUE;
      setPrefixPresent(insn, byte, prefixLocation);
      break;
    default:    /* Not a prefix byte */
      isPrefix = FALSE;
      break;
    }
    
    if (isPrefix)
      dbgprintf(insn, "Found prefix 0x%hhx", byte);
  }
    
  insn->vexSize = 0;
  
  if (byte == 0xc4) {
    uint8_t byte1;
      
    if (lookAtByte(insn, &byte1)) {
      dbgprintf(insn, "Couldn't read second byte of VEX");
      return -1;
    }
    
    if (insn->mode == MODE_64BIT || (byte1 & 0xc0) == 0xc0) {
      insn->vexSize = 3;
      insn->necessaryPrefixLocation = insn->readerCursor - 1;
    }
    else {
      unconsumeByte(insn);
      insn->necessaryPrefixLocation = insn->readerCursor - 1;
    }
    
    if (insn->vexSize == 3) {
      insn->vexPrefix[0] = byte;
      consumeByte(insn, &insn->vexPrefix[1]);
      consumeByte(insn, &insn->vexPrefix[2]);

      /* We simulate the REX prefix for simplicity's sake */
   
      if (insn->mode == MODE_64BIT) {
        insn->rexPrefix = 0x40 
                        | (wFromVEX3of3(insn->vexPrefix[2]) << 3)
                        | (rFromVEX2of3(insn->vexPrefix[1]) << 2)
                        | (xFromVEX2of3(insn->vexPrefix[1]) << 1)
                        | (bFromVEX2of3(insn->vexPrefix[1]) << 0);
      }
    
      switch (ppFromVEX3of3(insn->vexPrefix[2]))
      {
      default:
        break;
      case VEX_PREFIX_66:
        hasOpSize = TRUE;      
        break;
      }
    
      dbgprintf(insn, "Found VEX prefix 0x%hhx 0x%hhx 0x%hhx", insn->vexPrefix[0], insn->vexPrefix[1], insn->vexPrefix[2]);
    }
  }
  else if (byte == 0xc5) {
    uint8_t byte1;
    
    if (lookAtByte(insn, &byte1)) {
      dbgprintf(insn, "Couldn't read second byte of VEX");
      return -1;
    }
      
    if (insn->mode == MODE_64BIT || (byte1 & 0xc0) == 0xc0) {
      insn->vexSize = 2;
    }
    else {
      unconsumeByte(insn);
    }
    
    if (insn->vexSize == 2) {
      insn->vexPrefix[0] = byte;
      consumeByte(insn, &insn->vexPrefix[1]);
        
      if (insn->mode == MODE_64BIT) {
        insn->rexPrefix = 0x40 
                        | (rFromVEX2of2(insn->vexPrefix[1]) << 2);
      }
        
      switch (ppFromVEX2of2(insn->vexPrefix[1]))
      {
      default:
        break;
      case VEX_PREFIX_66:
        hasOpSize = TRUE;      
        break;
      }
         
      dbgprintf(insn, "Found VEX prefix 0x%hhx 0x%hhx", insn->vexPrefix[0], insn->vexPrefix[1]);
    }
  }
  else {
    if (insn->mode == MODE_64BIT) {
      if ((byte & 0xf0) == 0x40) {
        uint8_t opcodeByte;
          
        if (lookAtByte(insn, &opcodeByte) || ((opcodeByte & 0xf0) == 0x40)) {
          dbgprintf(insn, "Redundant REX prefix");
          return -1;
        }
          
        insn->rexPrefix = byte;
        insn->necessaryPrefixLocation = insn->readerCursor - 2;
          
        dbgprintf(insn, "Found REX prefix 0x%hhx", byte);
      } else {                
        unconsumeByte(insn);
        insn->necessaryPrefixLocation = insn->readerCursor - 1;
      }
    } else {
      unconsumeByte(insn);
      insn->necessaryPrefixLocation = insn->readerCursor - 1;
    }
  }

  if (insn->mode == MODE_16BIT) {
    insn->registerSize       = (hasOpSize ? 4 : 2);
    insn->addressSize        = (hasAdSize ? 4 : 2);
    insn->displacementSize   = (hasAdSize ? 4 : 2);
    insn->immediateSize      = (hasOpSize ? 4 : 2);
  } else if (insn->mode == MODE_32BIT) {
    insn->registerSize       = (hasOpSize ? 2 : 4);
    insn->addressSize        = (hasAdSize ? 2 : 4);
    insn->displacementSize   = (hasAdSize ? 2 : 4);
    insn->immediateSize      = (hasOpSize ? 2 : 4);
  } else if (insn->mode == MODE_64BIT) {
    if (insn->rexPrefix && wFromREX(insn->rexPrefix)) {
      insn->registerSize       = 8;
      insn->addressSize        = (hasAdSize ? 4 : 8);
      insn->displacementSize   = 4;
      insn->immediateSize      = 4;
    } else if (insn->rexPrefix) {
      insn->registerSize       = (hasOpSize ? 2 : 4);
      insn->addressSize        = (hasAdSize ? 4 : 8);
      insn->displacementSize   = (hasOpSize ? 2 : 4);
      insn->immediateSize      = (hasOpSize ? 2 : 4);
    } else {
      insn->registerSize       = (hasOpSize ? 2 : 4);
      insn->addressSize        = (hasAdSize ? 4 : 8);
      insn->displacementSize   = (hasOpSize ? 2 : 4);
      insn->immediateSize      = (hasOpSize ? 2 : 4);
    }
  }
  
  return 0;
}

/*
 * readOpcode - Reads the opcode (excepting the ModR/M byte in the case of
 *   extended or escape opcodes).
 *
 * @param insn  - The instruction whose opcode is to be read.
 * @return      - 0 if the opcode could be read successfully; nonzero otherwise.
 */
static int readOpcode(struct InternalInstruction* insn) {  
  /* Determine the length of the primary opcode */
  
  uint8_t current;
  
  dbgprintf(insn, "readOpcode()");
  
  insn->opcodeType = ONEBYTE;
    
  if (insn->vexSize == 3)
  {
    switch (mmmmmFromVEX2of3(insn->vexPrefix[1]))
    {
    default:
      dbgprintf(insn, "Unhandled m-mmmm field for instruction (0x%hhx)", mmmmmFromVEX2of3(insn->vexPrefix[1]));
      return -1;      
    case 0:
      break;
    case VEX_LOB_0F:
      insn->twoByteEscape = 0x0f;
      insn->opcodeType = TWOBYTE;
      return consumeByte(insn, &insn->opcode);
    case VEX_LOB_0F38:
      insn->twoByteEscape = 0x0f;
      insn->threeByteEscape = 0x38;
      insn->opcodeType = THREEBYTE_38;
      return consumeByte(insn, &insn->opcode);
    case VEX_LOB_0F3A:    
      insn->twoByteEscape = 0x0f;
      insn->threeByteEscape = 0x3a;
      insn->opcodeType = THREEBYTE_3A;
      return consumeByte(insn, &insn->opcode);
    }
  }
  else if (insn->vexSize == 2)
  {
    insn->twoByteEscape = 0x0f;
    insn->opcodeType = TWOBYTE;
    return consumeByte(insn, &insn->opcode);
  }
    
  if (consumeByte(insn, &current))
    return -1;
  
  if (current == 0x0f) {
    dbgprintf(insn, "Found a two-byte escape prefix (0x%hhx)", current);
    
    insn->twoByteEscape = current;
    
    if (consumeByte(insn, &current))
      return -1;
    
    if (current == 0x38) {
      dbgprintf(insn, "Found a three-byte escape prefix (0x%hhx)", current);
      
      insn->threeByteEscape = current;
      
      if (consumeByte(insn, &current))
        return -1;
      
      insn->opcodeType = THREEBYTE_38;
    } else if (current == 0x3a) {
      dbgprintf(insn, "Found a three-byte escape prefix (0x%hhx)", current);
      
      insn->threeByteEscape = current;
      
      if (consumeByte(insn, &current))
        return -1;
      
      insn->opcodeType = THREEBYTE_3A;
    } else if (current == 0xa6) {
      dbgprintf(insn, "Found a three-byte escape prefix (0x%hhx)", current);
      
      insn->threeByteEscape = current;
      
      if (consumeByte(insn, &current))
        return -1;
      
      insn->opcodeType = THREEBYTE_A6;
    } else if (current == 0xa7) {
      dbgprintf(insn, "Found a three-byte escape prefix (0x%hhx)", current);
      
      insn->threeByteEscape = current;
      
      if (consumeByte(insn, &current))
        return -1;
      
      insn->opcodeType = THREEBYTE_A7;
    } else {
      dbgprintf(insn, "Didn't find a three-byte escape prefix");
      
      insn->opcodeType = TWOBYTE;
    }
  }
  
  /*
   * At this point we have consumed the full opcode.
   * Anything we consume from here on must be unconsumed.
   */
  
  insn->opcode = current;
  
  return 0;
}

static int readModRM(struct InternalInstruction* insn);

/*
 * getIDWithAttrMask - Determines the ID of an instruction, consuming
 *   the ModR/M byte as appropriate for extended and escape opcodes,
 *   and using a supplied attribute mask.
 *
 * @param instructionID - A pointer whose target is filled in with the ID of the
 *                        instruction.
 * @param insn          - The instruction whose ID is to be determined.
 * @param attrMask      - The attribute mask to search.
 * @return              - 0 if the ModR/M could be read when needed or was not
 *                        needed; nonzero otherwise.
 */
static int getIDWithAttrMask(uint16_t* instructionID,
                             struct InternalInstruction* insn,
                             uint8_t attrMask) {
  BOOL hasModRMExtension;
  
  uint8_t instructionClass;

  instructionClass = contextForAttrs(attrMask);
  
  hasModRMExtension = modRMRequired(insn->opcodeType,
                                    instructionClass,
                                    insn->opcode);
  
  if (hasModRMExtension) {
    if (readModRM(insn))
      return -1;
    
    *instructionID = decode(insn->opcodeType,
                            instructionClass,
                            insn->opcode,
                            insn->modRM);
  } else {
    *instructionID = decode(insn->opcodeType,
                            instructionClass,
                            insn->opcode,
                            0);
  }
      
  return 0;
}

/*
 * is16BitEquivalent - Determines whether two instruction names refer to
 * equivalent instructions but one is 16-bit whereas the other is not.
 *
 * @param orig  - The instruction that is not 16-bit
 * @param equiv - The instruction that is 16-bit
 */
static BOOL is16BitEquivalent(const char* orig, const char* equiv) {
  off_t i;
  
  for (i = 0;; i++) {
    if (orig[i] == '\0' && equiv[i] == '\0')
      return TRUE;
    if (orig[i] == '\0' || equiv[i] == '\0')
      return FALSE;
    if (orig[i] != equiv[i]) {
      if ((orig[i] == 'Q' || orig[i] == 'L') && equiv[i] == 'W')
        continue;
      if ((orig[i] == '6' || orig[i] == '3') && equiv[i] == '1')
        continue;
      if ((orig[i] == '4' || orig[i] == '2') && equiv[i] == '6')
        continue;
      return FALSE;
    }
  }
}

/*
 * getID - Determines the ID of an instruction, consuming the ModR/M byte as 
 *   appropriate for extended and escape opcodes.  Determines the attributes and 
 *   context for the instruction before doing so.
 *
 * @param insn  - The instruction whose ID is to be determined.
 * @return      - 0 if the ModR/M could be read when needed or was not needed;
 *                nonzero otherwise.
 */
static int getID(struct InternalInstruction* insn, const void *miiArg) {
  uint8_t attrMask;
  uint16_t instructionID;
  
  dbgprintf(insn, "getID()");
    
  attrMask = ATTR_NONE;

  if (insn->mode == MODE_64BIT)
    attrMask |= ATTR_64BIT;
    
  if (insn->vexSize) {
    attrMask |= ATTR_VEX;

    if (insn->vexSize == 3) {
      switch (ppFromVEX3of3(insn->vexPrefix[2])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;    
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }
    
      if (lFromVEX3of3(insn->vexPrefix[2]))
        attrMask |= ATTR_VEXL;
    }
    else if (insn->vexSize == 2) {
      switch (ppFromVEX2of2(insn->vexPrefix[1])) {
      case VEX_PREFIX_66:
        attrMask |= ATTR_OPSIZE;    
        break;
      case VEX_PREFIX_F3:
        attrMask |= ATTR_XS;
        break;
      case VEX_PREFIX_F2:
        attrMask |= ATTR_XD;
        break;
      }
    
      if (lFromVEX2of2(insn->vexPrefix[1]))
        attrMask |= ATTR_VEXL;
    }
    else {
      return -1;
    }
  }
  else {
    if (isPrefixAtLocation(insn, 0x66, insn->necessaryPrefixLocation))
      attrMask |= ATTR_OPSIZE;
    else if (isPrefixAtLocation(insn, 0x67, insn->necessaryPrefixLocation))
      attrMask |= ATTR_ADSIZE;
    else if (isPrefixAtLocation(insn, 0xf3, insn->necessaryPrefixLocation))
      attrMask |= ATTR_XS;
    else if (isPrefixAtLocation(insn, 0xf2, insn->necessaryPrefixLocation))
      attrMask |= ATTR_XD;
  }

  if (insn->rexPrefix & 0x08)
    attrMask |= ATTR_REXW;

  if (getIDWithAttrMask(&instructionID, insn, attrMask))
    return -1;

  /* The following clauses compensate for limitations of the tables. */

  if ((attrMask & ATTR_VEXL) && (attrMask & ATTR_REXW) &&
      !(attrMask & ATTR_OPSIZE)) {
    /*
     * Some VEX instructions ignore the L-bit, but use the W-bit. Normally L-bit
     * has precedence since there are no L-bit with W-bit entries in the tables.
     * So if the L-bit isn't significant we should use the W-bit instead.
     * We only need to do this if the instruction doesn't specify OpSize since
     * there is a VEX_L_W_OPSIZE table.
     */

    const struct InstructionSpecifier *spec;
    uint16_t instructionIDWithWBit;
    const struct InstructionSpecifier *specWithWBit;

    spec = specifierForUID(instructionID);

    if (getIDWithAttrMask(&instructionIDWithWBit,
                          insn,
                          (attrMask & (~ATTR_VEXL)) | ATTR_REXW)) {
      insn->instructionID = instructionID;
      insn->spec = spec;
      return 0;
    }

    specWithWBit = specifierForUID(instructionIDWithWBit);

    if (instructionID != instructionIDWithWBit) {
      insn->instructionID = instructionIDWithWBit;
      insn->spec = specWithWBit;
    } else {
      insn->instructionID = instructionID;
      insn->spec = spec;
    }
    return 0;
  }

  if (insn->prefixPresent[0x66] && !(attrMask & ATTR_OPSIZE)) {
    /*
     * The instruction tables make no distinction between instructions that
     * allow OpSize anywhere (i.e., 16-bit operations) and that need it in a
     * particular spot (i.e., many MMX operations).  In general we're
     * conservative, but in the specific case where OpSize is present but not
     * in the right place we check if there's a 16-bit operation.
     */
    
    const struct InstructionSpecifier *spec;
    uint16_t instructionIDWithOpsize;
    const char *specName, *specWithOpSizeName;
    
    spec = specifierForUID(instructionID);
    
    if (getIDWithAttrMask(&instructionIDWithOpsize,
                          insn,
                          attrMask | ATTR_OPSIZE)) {
      /* 
       * ModRM required with OpSize but not present; give up and return version
       * without OpSize set
       */
      
      insn->instructionID = instructionID;
      insn->spec = spec;
      return 0;
    }
    
    specName = x86DisassemblerGetInstrName(instructionID, miiArg);
    specWithOpSizeName =
      x86DisassemblerGetInstrName(instructionIDWithOpsize, miiArg);

    if (is16BitEquivalent(specName, specWithOpSizeName)) {
      insn->instructionID = instructionIDWithOpsize;
      insn->spec = specifierForUID(instructionIDWithOpsize);
    } else {
      insn->instructionID = instructionID;
      insn->spec = spec;
    }
    return 0;
  }

  if (insn->opcodeType == ONEBYTE && insn->opcode == 0x90 &&
      insn->rexPrefix & 0x01) {
    /*
     * NOOP shouldn't decode as NOOP if REX.b is set. Instead
     * it should decode as XCHG %r8, %eax.
     */

    const struct InstructionSpecifier *spec;
    uint16_t instructionIDWithNewOpcode;
    const struct InstructionSpecifier *specWithNewOpcode;

    spec = specifierForUID(instructionID);
    
    /* Borrow opcode from one of the other XCHGar opcodes */
    insn->opcode = 0x91;
   
    if (getIDWithAttrMask(&instructionIDWithNewOpcode,
                          insn,
                          attrMask)) {
      insn->opcode = 0x90;

      insn->instructionID = instructionID;
      insn->spec = spec;
      return 0;
    }

    specWithNewOpcode = specifierForUID(instructionIDWithNewOpcode);

    /* Change back */
    insn->opcode = 0x90;

    insn->instructionID = instructionIDWithNewOpcode;
    insn->spec = specWithNewOpcode;

    return 0;
  }
  
  insn->instructionID = instructionID;
  insn->spec = specifierForUID(insn->instructionID);
  
  return 0;
}

/*
 * readSIB - Consumes the SIB byte to determine addressing information for an
 *   instruction.
 *
 * @param insn  - The instruction whose SIB byte is to be read.
 * @return      - 0 if the SIB byte was successfully read; nonzero otherwise.
 */
static int readSIB(struct InternalInstruction* insn) {
  SIBIndex sibIndexBase = 0;
  SIBBase sibBaseBase = 0;
  uint8_t index, base;
  
  dbgprintf(insn, "readSIB()");
  
  if (insn->consumedSIB)
    return 0;
  
  insn->consumedSIB = TRUE;
  
  switch (insn->addressSize) {
  case 2:
    dbgprintf(insn, "SIB-based addressing doesn't work in 16-bit mode");
    return -1;
    break;
  case 4:
    sibIndexBase = SIB_INDEX_EAX;
    sibBaseBase = SIB_BASE_EAX;
    break;
  case 8:
    sibIndexBase = SIB_INDEX_RAX;
    sibBaseBase = SIB_BASE_RAX;
    break;
  }

  if (consumeByte(insn, &insn->sib))
    return -1;
  
  index = indexFromSIB(insn->sib) | (xFromREX(insn->rexPrefix) << 3);
  
  switch (index) {
  case 0x4:
    insn->sibIndex = SIB_INDEX_NONE;
    break;
  default:
    insn->sibIndex = (SIBIndex)(sibIndexBase + index);
    if (insn->sibIndex == SIB_INDEX_sib ||
        insn->sibIndex == SIB_INDEX_sib64)
      insn->sibIndex = SIB_INDEX_NONE;
    break;
  }
  
  switch (scaleFromSIB(insn->sib)) {
  case 0:
    insn->sibScale = 1;
    break;
  case 1:
    insn->sibScale = 2;
    break;
  case 2:
    insn->sibScale = 4;
    break;
  case 3:
    insn->sibScale = 8;
    break;
  }
  
  base = baseFromSIB(insn->sib) | (bFromREX(insn->rexPrefix) << 3);
  
  switch (base) {
  case 0x5:
    switch (modFromModRM(insn->modRM)) {
    case 0x0:
      insn->eaDisplacement = EA_DISP_32;
      insn->sibBase = SIB_BASE_NONE;
      break;
    case 0x1:
      insn->eaDisplacement = EA_DISP_8;
      insn->sibBase = (insn->addressSize == 4 ? 
                       SIB_BASE_EBP : SIB_BASE_RBP);
      break;
    case 0x2:
      insn->eaDisplacement = EA_DISP_32;
      insn->sibBase = (insn->addressSize == 4 ? 
                       SIB_BASE_EBP : SIB_BASE_RBP);
      break;
    case 0x3:
      debug("Cannot have Mod = 0b11 and a SIB byte");
      return -1;
    }
    break;
  default:
    insn->sibBase = (SIBBase)(sibBaseBase + base);
    break;
  }
  
  return 0;
}

/*
 * readDisplacement - Consumes the displacement of an instruction.
 *
 * @param insn  - The instruction whose displacement is to be read.
 * @return      - 0 if the displacement byte was successfully read; nonzero 
 *                otherwise.
 */
static int readDisplacement(struct InternalInstruction* insn) {  
  int8_t d8;
  int16_t d16;
  int32_t d32;
  
  dbgprintf(insn, "readDisplacement()");
  
  if (insn->consumedDisplacement)
    return 0;
  
  insn->consumedDisplacement = TRUE;
  insn->displacementOffset = insn->readerCursor - insn->startLocation;
  
  switch (insn->eaDisplacement) {
  case EA_DISP_NONE:
    insn->consumedDisplacement = FALSE;
    break;
  case EA_DISP_8:
    if (consumeInt8(insn, &d8))
      return -1;
    insn->displacement = d8;
    break;
  case EA_DISP_16:
    if (consumeInt16(insn, &d16))
      return -1;
    insn->displacement = d16;
    break;
  case EA_DISP_32:
    if (consumeInt32(insn, &d32))
      return -1;
    insn->displacement = d32;
    break;
  }
  
  insn->consumedDisplacement = TRUE;
  return 0;
}

/*
 * readModRM - Consumes all addressing information (ModR/M byte, SIB byte, and
 *   displacement) for an instruction and interprets it.
 *
 * @param insn  - The instruction whose addressing information is to be read.
 * @return      - 0 if the information was successfully read; nonzero otherwise.
 */
static int readModRM(struct InternalInstruction* insn) {  
  uint8_t mod, rm, reg;
  
  dbgprintf(insn, "readModRM()");
  
  if (insn->consumedModRM)
    return 0;
  
  if (consumeByte(insn, &insn->modRM))
    return -1;
  insn->consumedModRM = TRUE;
  
  mod     = modFromModRM(insn->modRM);
  rm      = rmFromModRM(insn->modRM);
  reg     = regFromModRM(insn->modRM);
  
  /*
   * This goes by insn->registerSize to pick the correct register, which messes
   * up if we're using (say) XMM or 8-bit register operands.  That gets fixed in
   * fixupReg().
   */
  switch (insn->registerSize) {
  case 2:
    insn->regBase = MODRM_REG_AX;
    insn->eaRegBase = EA_REG_AX;
    break;
  case 4:
    insn->regBase = MODRM_REG_EAX;
    insn->eaRegBase = EA_REG_EAX;
    break;
  case 8:
    insn->regBase = MODRM_REG_RAX;
    insn->eaRegBase = EA_REG_RAX;
    break;
  }
  
  reg |= rFromREX(insn->rexPrefix) << 3;
  rm  |= bFromREX(insn->rexPrefix) << 3;
  
  insn->reg = (Reg)(insn->regBase + reg);
  
  switch (insn->addressSize) {
  case 2:
    insn->eaBaseBase = EA_BASE_BX_SI;
     
    switch (mod) {
    case 0x0:
      if (rm == 0x6) {
        insn->eaBase = EA_BASE_NONE;
        insn->eaDisplacement = EA_DISP_16;
        if (readDisplacement(insn))
          return -1;
      } else {
        insn->eaBase = (EABase)(insn->eaBaseBase + rm);
        insn->eaDisplacement = EA_DISP_NONE;
      }
      break;
    case 0x1:
      insn->eaBase = (EABase)(insn->eaBaseBase + rm);
      insn->eaDisplacement = EA_DISP_8;
      if (readDisplacement(insn))
        return -1;
      break;
    case 0x2:
      insn->eaBase = (EABase)(insn->eaBaseBase + rm);
      insn->eaDisplacement = EA_DISP_16;
      if (readDisplacement(insn))
        return -1;
      break;
    case 0x3:
      insn->eaBase = (EABase)(insn->eaRegBase + rm);
      if (readDisplacement(insn))
        return -1;
      break;
    }
    break;
  case 4:
  case 8:
    insn->eaBaseBase = (insn->addressSize == 4 ? EA_BASE_EAX : EA_BASE_RAX);
    
    switch (mod) {
    case 0x0:
      insn->eaDisplacement = EA_DISP_NONE; /* readSIB may override this */
      switch (rm) {
      case 0x4:
      case 0xc:   /* in case REXW.b is set */
        insn->eaBase = (insn->addressSize == 4 ? 
                        EA_BASE_sib : EA_BASE_sib64);
        readSIB(insn);
        if (readDisplacement(insn))
          return -1;
        break;
      case 0x5:
        insn->eaBase = EA_BASE_NONE;
        insn->eaDisplacement = EA_DISP_32;
        if (readDisplacement(insn))
          return -1;
        break;
      default:
        insn->eaBase = (EABase)(insn->eaBaseBase + rm);
        break;
      }
      break;
    case 0x1:
    case 0x2:
      insn->eaDisplacement = (mod == 0x1 ? EA_DISP_8 : EA_DISP_32);
      switch (rm) {
      case 0x4:
      case 0xc:   /* in case REXW.b is set */
        insn->eaBase = EA_BASE_sib;
        readSIB(insn);
        if (readDisplacement(insn))
          return -1;
        break;
      default:
        insn->eaBase = (EABase)(insn->eaBaseBase + rm);
        if (readDisplacement(insn))
          return -1;
        break;
      }
      break;
    case 0x3:
      insn->eaDisplacement = EA_DISP_NONE;
      insn->eaBase = (EABase)(insn->eaRegBase + rm);
      break;
    }
    break;
  } /* switch (insn->addressSize) */
  
  return 0;
}

#define GENERIC_FIXUP_FUNC(name, base, prefix)            \
  static uint8_t name(struct InternalInstruction *insn,   \
                      OperandType type,                   \
                      uint8_t index,                      \
                      uint8_t *valid) {                   \
    *valid = 1;                                           \
    switch (type) {                                       \
    default:                                              \
      debug("Unhandled register type");                   \
      *valid = 0;                                         \
      return 0;                                           \
    case TYPE_Rv:                                         \
      return base + index;                                \
    case TYPE_R8:                                         \
      if (insn->rexPrefix &&                              \
         index >= 4 && index <= 7) {                      \
        return prefix##_SPL + (index - 4);                \
      } else {                                            \
        return prefix##_AL + index;                       \
      }                                                   \
    case TYPE_R16:                                        \
      return prefix##_AX + index;                         \
    case TYPE_R32:                                        \
      return prefix##_EAX + index;                        \
    case TYPE_R64:                                        \
      return prefix##_RAX + index;                        \
    case TYPE_XMM256:                                     \
      return prefix##_YMM0 + index;                       \
    case TYPE_XMM128:                                     \
    case TYPE_XMM64:                                      \
    case TYPE_XMM32:                                      \
    case TYPE_XMM:                                        \
      return prefix##_XMM0 + index;                       \
    case TYPE_MM64:                                       \
    case TYPE_MM32:                                       \
    case TYPE_MM:                                         \
      if (index > 7)                                      \
        *valid = 0;                                       \
      return prefix##_MM0 + index;                        \
    case TYPE_SEGMENTREG:                                 \
      if (index > 5)                                      \
        *valid = 0;                                       \
      return prefix##_ES + index;                         \
    case TYPE_DEBUGREG:                                   \
      if (index > 7)                                      \
        *valid = 0;                                       \
      return prefix##_DR0 + index;                        \
    case TYPE_CONTROLREG:                                 \
      if (index > 8)                                      \
        *valid = 0;                                       \
      return prefix##_CR0 + index;                        \
    }                                                     \
  }

/*
 * fixup*Value - Consults an operand type to determine the meaning of the
 *   reg or R/M field.  If the operand is an XMM operand, for example, an
 *   operand would be XMM0 instead of AX, which readModRM() would otherwise
 *   misinterpret it as.
 *
 * @param insn  - The instruction containing the operand.
 * @param type  - The operand type.
 * @param index - The existing value of the field as reported by readModRM().
 * @param valid - The address of a uint8_t.  The target is set to 1 if the
 *                field is valid for the register class; 0 if not.
 * @return      - The proper value.
 */
GENERIC_FIXUP_FUNC(fixupRegValue, insn->regBase,    MODRM_REG)
GENERIC_FIXUP_FUNC(fixupRMValue,  insn->eaRegBase,  EA_REG)

/*
 * fixupReg - Consults an operand specifier to determine which of the
 *   fixup*Value functions to use in correcting readModRM()'ss interpretation.
 *
 * @param insn  - See fixup*Value().
 * @param op    - The operand specifier.
 * @return      - 0 if fixup was successful; -1 if the register returned was
 *                invalid for its class.
 */
static int fixupReg(struct InternalInstruction *insn, 
                    const struct OperandSpecifier *op) {
  uint8_t valid;
  
  dbgprintf(insn, "fixupReg()");
  
  switch ((OperandEncoding)op->encoding) {
  default:
    debug("Expected a REG or R/M encoding in fixupReg");
    return -1;
  case ENCODING_VVVV:
    insn->vvvv = (Reg)fixupRegValue(insn,
                                    (OperandType)op->type,
                                    insn->vvvv,
                                    &valid);
    if (!valid)
      return -1;
    break;
  case ENCODING_REG:
    insn->reg = (Reg)fixupRegValue(insn,
                                   (OperandType)op->type,
                                   insn->reg - insn->regBase,
                                   &valid);
    if (!valid)
      return -1;
    break;
  case ENCODING_RM:
    if (insn->eaBase >= insn->eaRegBase) {
      insn->eaBase = (EABase)fixupRMValue(insn,
                                          (OperandType)op->type,
                                          insn->eaBase - insn->eaRegBase,
                                          &valid);
      if (!valid)
        return -1;
    }
    break;
  }
  
  return 0;
}

/*
 * readOpcodeModifier - Reads an operand from the opcode field of an 
 *   instruction.  Handles AddRegFrm instructions.
 *
 * @param insn    - The instruction whose opcode field is to be read.
 * @param inModRM - Indicates that the opcode field is to be read from the
 *                  ModR/M extension; useful for escape opcodes
 * @return        - 0 on success; nonzero otherwise.
 */
static int readOpcodeModifier(struct InternalInstruction* insn) {
  dbgprintf(insn, "readOpcodeModifier()");
  
  if (insn->consumedOpcodeModifier)
    return 0;
  
  insn->consumedOpcodeModifier = TRUE;
  
  switch (insn->spec->modifierType) {
  default:
    debug("Unknown modifier type.");
    return -1;
  case MODIFIER_NONE:
    debug("No modifier but an operand expects one.");
    return -1;
  case MODIFIER_OPCODE:
    insn->opcodeModifier = insn->opcode - insn->spec->modifierBase;
    return 0;
  case MODIFIER_MODRM:
    insn->opcodeModifier = insn->modRM - insn->spec->modifierBase;
    return 0;
  }  
}

/*
 * readOpcodeRegister - Reads an operand from the opcode field of an 
 *   instruction and interprets it appropriately given the operand width.
 *   Handles AddRegFrm instructions.
 *
 * @param insn  - See readOpcodeModifier().
 * @param size  - The width (in bytes) of the register being specified.
 *                1 means AL and friends, 2 means AX, 4 means EAX, and 8 means
 *                RAX.
 * @return      - 0 on success; nonzero otherwise.
 */
static int readOpcodeRegister(struct InternalInstruction* insn, uint8_t size) {
  dbgprintf(insn, "readOpcodeRegister()");

  if (readOpcodeModifier(insn))
    return -1;
  
  if (size == 0)
    size = insn->registerSize;
  
  switch (size) {
  case 1:
    insn->opcodeRegister = (Reg)(MODRM_REG_AL + ((bFromREX(insn->rexPrefix) << 3) 
                                                  | insn->opcodeModifier));
    if (insn->rexPrefix && 
        insn->opcodeRegister >= MODRM_REG_AL + 0x4 &&
        insn->opcodeRegister < MODRM_REG_AL + 0x8) {
      insn->opcodeRegister = (Reg)(MODRM_REG_SPL
                                   + (insn->opcodeRegister - MODRM_REG_AL - 4));
    }
      
    break;
  case 2:
    insn->opcodeRegister = (Reg)(MODRM_REG_AX
                                 + ((bFromREX(insn->rexPrefix) << 3) 
                                    | insn->opcodeModifier));
    break;
  case 4:
    insn->opcodeRegister = (Reg)(MODRM_REG_EAX
                                 + ((bFromREX(insn->rexPrefix) << 3) 
                                    | insn->opcodeModifier));
    break;
  case 8:
    insn->opcodeRegister = (Reg)(MODRM_REG_RAX 
                                 + ((bFromREX(insn->rexPrefix) << 3) 
                                    | insn->opcodeModifier));
    break;
  }
  
  return 0;
}

/*
 * readImmediate - Consumes an immediate operand from an instruction, given the
 *   desired operand size.
 *
 * @param insn  - The instruction whose operand is to be read.
 * @param size  - The width (in bytes) of the operand.
 * @return      - 0 if the immediate was successfully consumed; nonzero
 *                otherwise.
 */
static int readImmediate(struct InternalInstruction* insn, uint8_t size) {
  uint8_t imm8;
  uint16_t imm16;
  uint32_t imm32;
  uint64_t imm64;
  
  dbgprintf(insn, "readImmediate()");
  
  if (insn->numImmediatesConsumed == 2) {
    debug("Already consumed two immediates");
    return -1;
  }
  
  if (size == 0)
    size = insn->immediateSize;
  else
    insn->immediateSize = size;
  insn->immediateOffset = insn->readerCursor - insn->startLocation;
  
  switch (size) {
  case 1:
    if (consumeByte(insn, &imm8))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm8;
    break;
  case 2:
    if (consumeUInt16(insn, &imm16))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm16;
    break;
  case 4:
    if (consumeUInt32(insn, &imm32))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm32;
    break;
  case 8:
    if (consumeUInt64(insn, &imm64))
      return -1;
    insn->immediates[insn->numImmediatesConsumed] = imm64;
    break;
  }
  
  insn->numImmediatesConsumed++;
  
  return 0;
}

/*
 * readVVVV - Consumes vvvv from an instruction if it has a VEX prefix.
 *
 * @param insn  - The instruction whose operand is to be read.
 * @return      - 0 if the vvvv was successfully consumed; nonzero
 *                otherwise.
 */
static int readVVVV(struct InternalInstruction* insn) {
  dbgprintf(insn, "readVVVV()");
        
  if (insn->vexSize == 3)
    insn->vvvv = vvvvFromVEX3of3(insn->vexPrefix[2]);
  else if (insn->vexSize == 2)
    insn->vvvv = vvvvFromVEX2of2(insn->vexPrefix[1]);
  else
    return -1;

  if (insn->mode != MODE_64BIT)
    insn->vvvv &= 0x7;

  return 0;
}

/*
 * readOperands - Consults the specifier for an instruction and consumes all
 *   operands for that instruction, interpreting them as it goes.
 *
 * @param insn  - The instruction whose operands are to be read and interpreted.
 * @return      - 0 if all operands could be read; nonzero otherwise.
 */
static int readOperands(struct InternalInstruction* insn) {
  int index;
  int hasVVVV, needVVVV;
  int sawRegImm = 0;
  
  dbgprintf(insn, "readOperands()");

  /* If non-zero vvvv specified, need to make sure one of the operands
     uses it. */
  hasVVVV = !readVVVV(insn);
  needVVVV = hasVVVV && (insn->vvvv != 0);
  
  for (index = 0; index < X86_MAX_OPERANDS; ++index) {
    switch (x86OperandSets[insn->spec->operands][index].encoding) {
    case ENCODING_NONE:
      break;
    case ENCODING_REG:
    case ENCODING_RM:
      if (readModRM(insn))
        return -1;
      if (fixupReg(insn, &x86OperandSets[insn->spec->operands][index]))
        return -1;
      break;
    case ENCODING_CB:
    case ENCODING_CW:
    case ENCODING_CD:
    case ENCODING_CP:
    case ENCODING_CO:
    case ENCODING_CT:
      dbgprintf(insn, "We currently don't hande code-offset encodings");
      return -1;
    case ENCODING_IB:
      if (sawRegImm) {
        /* Saw a register immediate so don't read again and instead split the
           previous immediate.  FIXME: This is a hack. */
        insn->immediates[insn->numImmediatesConsumed] =
          insn->immediates[insn->numImmediatesConsumed - 1] & 0xf;
        ++insn->numImmediatesConsumed;
        break;
      }
      if (readImmediate(insn, 1))
        return -1;
      if (x86OperandSets[insn->spec->operands][index].type == TYPE_IMM3 &&
          insn->immediates[insn->numImmediatesConsumed - 1] > 7)
        return -1;
      if (x86OperandSets[insn->spec->operands][index].type == TYPE_IMM5 &&
          insn->immediates[insn->numImmediatesConsumed - 1] > 31)
        return -1;
      if (x86OperandSets[insn->spec->operands][index].type == TYPE_XMM128 ||
          x86OperandSets[insn->spec->operands][index].type == TYPE_XMM256)
        sawRegImm = 1;
      break;
    case ENCODING_IW:
      if (readImmediate(insn, 2))
        return -1;
      break;
    case ENCODING_ID:
      if (readImmediate(insn, 4))
        return -1;
      break;
    case ENCODING_IO:
      if (readImmediate(insn, 8))
        return -1;
      break;
    case ENCODING_Iv:
      if (readImmediate(insn, insn->immediateSize))
        return -1;
      break;
    case ENCODING_Ia:
      if (readImmediate(insn, insn->addressSize))
        return -1;
      break;
    case ENCODING_RB:
      if (readOpcodeRegister(insn, 1))
        return -1;
      break;
    case ENCODING_RW:
      if (readOpcodeRegister(insn, 2))
        return -1;
      break;
    case ENCODING_RD:
      if (readOpcodeRegister(insn, 4))
        return -1;
      break;
    case ENCODING_RO:
      if (readOpcodeRegister(insn, 8))
        return -1;
      break;
    case ENCODING_Rv:
      if (readOpcodeRegister(insn, 0))
        return -1;
      break;
    case ENCODING_I:
      if (readOpcodeModifier(insn))
        return -1;
      break;
    case ENCODING_VVVV:
      needVVVV = 0; /* Mark that we have found a VVVV operand. */
      if (!hasVVVV)
        return -1;
      if (fixupReg(insn, &x86OperandSets[insn->spec->operands][index]))
        return -1;
      break;
    case ENCODING_DUP:
      break;
    default:
      dbgprintf(insn, "Encountered an operand with an unknown encoding.");
      return -1;
    }
  }

  /* If we didn't find ENCODING_VVVV operand, but non-zero vvvv present, fail */
  if (needVVVV) return -1;
  
  return 0;
}

/*
 * decodeInstruction - Reads and interprets a full instruction provided by the
 *   user.
 *
 * @param insn      - A pointer to the instruction to be populated.  Must be 
 *                    pre-allocated.
 * @param reader    - The function to be used to read the instruction's bytes.
 * @param readerArg - A generic argument to be passed to the reader to store
 *                    any internal state.
 * @param logger    - If non-NULL, the function to be used to write log messages
 *                    and warnings.
 * @param loggerArg - A generic argument to be passed to the logger to store
 *                    any internal state.
 * @param startLoc  - The address (in the reader's address space) of the first
 *                    byte in the instruction.
 * @param mode      - The mode (real mode, IA-32e, or IA-32e in 64-bit mode) to
 *                    decode the instruction in.
 * @return          - 0 if the instruction's memory could be read; nonzero if
 *                    not.
 */
int decodeInstruction(struct InternalInstruction* insn,
                      byteReader_t reader,
                      const void* readerArg,
                      dlog_t logger,
                      void* loggerArg,
                      const void* miiArg,
                      uint64_t startLoc,
                      DisassemblerMode mode) {
  memset(insn, 0, sizeof(struct InternalInstruction));
    
  insn->reader = reader;
  insn->readerArg = readerArg;
  insn->dlog = logger;
  insn->dlogArg = loggerArg;
  insn->startLocation = startLoc;
  insn->readerCursor = startLoc;
  insn->mode = mode;
  insn->numImmediatesConsumed = 0;
  
  if (readPrefixes(insn)       ||
      readOpcode(insn)         ||
      getID(insn, miiArg)      ||
      insn->instructionID == 0 ||
      readOperands(insn))
    return -1;

  insn->operands = &x86OperandSets[insn->spec->operands][0];
  
  insn->length = insn->readerCursor - insn->startLocation;
  
  dbgprintf(insn, "Read from 0x%llx to 0x%llx: length %zu",
            startLoc, insn->readerCursor, insn->length);
    
  if (insn->length > 15)
    dbgprintf(insn, "Instruction exceeds 15-byte limit");
  
  return 0;
}
