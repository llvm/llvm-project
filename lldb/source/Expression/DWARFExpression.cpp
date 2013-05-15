//===-- DWARFExpression.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"

#include <vector>

#include "lldb/Core/DataEncoder.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/VMRange.h"

#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionVariable.h"

#include "lldb/Host/Endian.h"
#include "lldb/Host/Host.h"

#include "lldb/lldb-private-log.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Type.h"

#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;

const char *
DW_OP_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x03: return "DW_OP_addr";
    case 0x06: return "DW_OP_deref";
    case 0x08: return "DW_OP_const1u";
    case 0x09: return "DW_OP_const1s";
    case 0x0a: return "DW_OP_const2u";
    case 0x0b: return "DW_OP_const2s";
    case 0x0c: return "DW_OP_const4u";
    case 0x0d: return "DW_OP_const4s";
    case 0x0e: return "DW_OP_const8u";
    case 0x0f: return "DW_OP_const8s";
    case 0x10: return "DW_OP_constu";
    case 0x11: return "DW_OP_consts";
    case 0x12: return "DW_OP_dup";
    case 0x13: return "DW_OP_drop";
    case 0x14: return "DW_OP_over";
    case 0x15: return "DW_OP_pick";
    case 0x16: return "DW_OP_swap";
    case 0x17: return "DW_OP_rot";
    case 0x18: return "DW_OP_xderef";
    case 0x19: return "DW_OP_abs";
    case 0x1a: return "DW_OP_and";
    case 0x1b: return "DW_OP_div";
    case 0x1c: return "DW_OP_minus";
    case 0x1d: return "DW_OP_mod";
    case 0x1e: return "DW_OP_mul";
    case 0x1f: return "DW_OP_neg";
    case 0x20: return "DW_OP_not";
    case 0x21: return "DW_OP_or";
    case 0x22: return "DW_OP_plus";
    case 0x23: return "DW_OP_plus_uconst";
    case 0x24: return "DW_OP_shl";
    case 0x25: return "DW_OP_shr";
    case 0x26: return "DW_OP_shra";
    case 0x27: return "DW_OP_xor";
    case 0x2f: return "DW_OP_skip";
    case 0x28: return "DW_OP_bra";
    case 0x29: return "DW_OP_eq";
    case 0x2a: return "DW_OP_ge";
    case 0x2b: return "DW_OP_gt";
    case 0x2c: return "DW_OP_le";
    case 0x2d: return "DW_OP_lt";
    case 0x2e: return "DW_OP_ne";
    case 0x30: return "DW_OP_lit0";
    case 0x31: return "DW_OP_lit1";
    case 0x32: return "DW_OP_lit2";
    case 0x33: return "DW_OP_lit3";
    case 0x34: return "DW_OP_lit4";
    case 0x35: return "DW_OP_lit5";
    case 0x36: return "DW_OP_lit6";
    case 0x37: return "DW_OP_lit7";
    case 0x38: return "DW_OP_lit8";
    case 0x39: return "DW_OP_lit9";
    case 0x3a: return "DW_OP_lit10";
    case 0x3b: return "DW_OP_lit11";
    case 0x3c: return "DW_OP_lit12";
    case 0x3d: return "DW_OP_lit13";
    case 0x3e: return "DW_OP_lit14";
    case 0x3f: return "DW_OP_lit15";
    case 0x40: return "DW_OP_lit16";
    case 0x41: return "DW_OP_lit17";
    case 0x42: return "DW_OP_lit18";
    case 0x43: return "DW_OP_lit19";
    case 0x44: return "DW_OP_lit20";
    case 0x45: return "DW_OP_lit21";
    case 0x46: return "DW_OP_lit22";
    case 0x47: return "DW_OP_lit23";
    case 0x48: return "DW_OP_lit24";
    case 0x49: return "DW_OP_lit25";
    case 0x4a: return "DW_OP_lit26";
    case 0x4b: return "DW_OP_lit27";
    case 0x4c: return "DW_OP_lit28";
    case 0x4d: return "DW_OP_lit29";
    case 0x4e: return "DW_OP_lit30";
    case 0x4f: return "DW_OP_lit31";
    case 0x50: return "DW_OP_reg0";
    case 0x51: return "DW_OP_reg1";
    case 0x52: return "DW_OP_reg2";
    case 0x53: return "DW_OP_reg3";
    case 0x54: return "DW_OP_reg4";
    case 0x55: return "DW_OP_reg5";
    case 0x56: return "DW_OP_reg6";
    case 0x57: return "DW_OP_reg7";
    case 0x58: return "DW_OP_reg8";
    case 0x59: return "DW_OP_reg9";
    case 0x5a: return "DW_OP_reg10";
    case 0x5b: return "DW_OP_reg11";
    case 0x5c: return "DW_OP_reg12";
    case 0x5d: return "DW_OP_reg13";
    case 0x5e: return "DW_OP_reg14";
    case 0x5f: return "DW_OP_reg15";
    case 0x60: return "DW_OP_reg16";
    case 0x61: return "DW_OP_reg17";
    case 0x62: return "DW_OP_reg18";
    case 0x63: return "DW_OP_reg19";
    case 0x64: return "DW_OP_reg20";
    case 0x65: return "DW_OP_reg21";
    case 0x66: return "DW_OP_reg22";
    case 0x67: return "DW_OP_reg23";
    case 0x68: return "DW_OP_reg24";
    case 0x69: return "DW_OP_reg25";
    case 0x6a: return "DW_OP_reg26";
    case 0x6b: return "DW_OP_reg27";
    case 0x6c: return "DW_OP_reg28";
    case 0x6d: return "DW_OP_reg29";
    case 0x6e: return "DW_OP_reg30";
    case 0x6f: return "DW_OP_reg31";
    case 0x70: return "DW_OP_breg0";
    case 0x71: return "DW_OP_breg1";
    case 0x72: return "DW_OP_breg2";
    case 0x73: return "DW_OP_breg3";
    case 0x74: return "DW_OP_breg4";
    case 0x75: return "DW_OP_breg5";
    case 0x76: return "DW_OP_breg6";
    case 0x77: return "DW_OP_breg7";
    case 0x78: return "DW_OP_breg8";
    case 0x79: return "DW_OP_breg9";
    case 0x7a: return "DW_OP_breg10";
    case 0x7b: return "DW_OP_breg11";
    case 0x7c: return "DW_OP_breg12";
    case 0x7d: return "DW_OP_breg13";
    case 0x7e: return "DW_OP_breg14";
    case 0x7f: return "DW_OP_breg15";
    case 0x80: return "DW_OP_breg16";
    case 0x81: return "DW_OP_breg17";
    case 0x82: return "DW_OP_breg18";
    case 0x83: return "DW_OP_breg19";
    case 0x84: return "DW_OP_breg20";
    case 0x85: return "DW_OP_breg21";
    case 0x86: return "DW_OP_breg22";
    case 0x87: return "DW_OP_breg23";
    case 0x88: return "DW_OP_breg24";
    case 0x89: return "DW_OP_breg25";
    case 0x8a: return "DW_OP_breg26";
    case 0x8b: return "DW_OP_breg27";
    case 0x8c: return "DW_OP_breg28";
    case 0x8d: return "DW_OP_breg29";
    case 0x8e: return "DW_OP_breg30";
    case 0x8f: return "DW_OP_breg31";
    case 0x90: return "DW_OP_regx";
    case 0x91: return "DW_OP_fbreg";
    case 0x92: return "DW_OP_bregx";
    case 0x93: return "DW_OP_piece";
    case 0x94: return "DW_OP_deref_size";
    case 0x95: return "DW_OP_xderef_size";
    case 0x96: return "DW_OP_nop";
    case 0x97: return "DW_OP_push_object_address";
    case 0x98: return "DW_OP_call2";
    case 0x99: return "DW_OP_call4";
    case 0x9a: return "DW_OP_call_ref";
//    case DW_OP_APPLE_array_ref: return "DW_OP_APPLE_array_ref";
//    case DW_OP_APPLE_extern: return "DW_OP_APPLE_extern";
    case DW_OP_APPLE_uninit: return "DW_OP_APPLE_uninit";
//    case DW_OP_APPLE_assign: return "DW_OP_APPLE_assign";
//    case DW_OP_APPLE_address_of: return "DW_OP_APPLE_address_of";
//    case DW_OP_APPLE_value_of: return "DW_OP_APPLE_value_of";
//    case DW_OP_APPLE_deref_type: return "DW_OP_APPLE_deref_type";
//    case DW_OP_APPLE_expr_local: return "DW_OP_APPLE_expr_local";
//    case DW_OP_APPLE_constf: return "DW_OP_APPLE_constf";
//    case DW_OP_APPLE_scalar_cast: return "DW_OP_APPLE_scalar_cast";
//    case DW_OP_APPLE_clang_cast: return "DW_OP_APPLE_clang_cast";
//    case DW_OP_APPLE_clear: return "DW_OP_APPLE_clear";
//    case DW_OP_APPLE_error: return "DW_OP_APPLE_error";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_OP constant: 0x%x", val);
       return invalid;
  }
}


//----------------------------------------------------------------------
// DWARFExpression constructor
//----------------------------------------------------------------------
DWARFExpression::DWARFExpression() :
    m_data(),
    m_reg_kind (eRegisterKindDWARF),
    m_loclist_slide (LLDB_INVALID_ADDRESS)
{
}

DWARFExpression::DWARFExpression(const DWARFExpression& rhs) :
    m_data(rhs.m_data),
    m_reg_kind (rhs.m_reg_kind),
    m_loclist_slide(rhs.m_loclist_slide)
{
}


DWARFExpression::DWARFExpression(const DataExtractor& data, lldb::offset_t data_offset, lldb::offset_t data_length) :
    m_data(data, data_offset, data_length),
    m_reg_kind (eRegisterKindDWARF),
    m_loclist_slide(LLDB_INVALID_ADDRESS)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
DWARFExpression::~DWARFExpression()
{
}


bool
DWARFExpression::IsValid() const
{
    return m_data.GetByteSize() > 0;
}

void
DWARFExpression::SetOpcodeData (const DataExtractor& data)
{
    m_data = data;
}

void
DWARFExpression::CopyOpcodeData (const DataExtractor& data, lldb::offset_t data_offset, lldb::offset_t data_length)
{
    const uint8_t *bytes = data.PeekData(data_offset, data_length);
    if (bytes)
    {
        m_data.SetData(DataBufferSP(new DataBufferHeap(bytes, data_length)));
        m_data.SetByteOrder(data.GetByteOrder());
        m_data.SetAddressByteSize(data.GetAddressByteSize());
    }
}

void
DWARFExpression::SetOpcodeData (const DataExtractor& data, lldb::offset_t data_offset, lldb::offset_t data_length)
{
    m_data.SetData(data, data_offset, data_length);
}

void
DWARFExpression::DumpLocation (Stream *s, lldb::offset_t offset, lldb::offset_t length, lldb::DescriptionLevel level, ABI *abi) const
{
    if (!m_data.ValidOffsetForDataOfSize(offset, length))
        return;
    const lldb::offset_t start_offset = offset;
    const lldb::offset_t end_offset = offset + length;
    while (m_data.ValidOffset(offset) && offset < end_offset)
    {
        const lldb::offset_t op_offset = offset;
        const uint8_t op = m_data.GetU8(&offset);

        switch (level)
        {
        default:
            break;

        case lldb::eDescriptionLevelBrief:
            if (offset > start_offset)
                s->PutChar(' ');
            break;

        case lldb::eDescriptionLevelFull:
        case lldb::eDescriptionLevelVerbose:
            if (offset > start_offset)
                s->EOL();
            s->Indent();
            if (level == lldb::eDescriptionLevelFull)
                break;
            // Fall through for verbose and print offset and DW_OP prefix..
            s->Printf("0x%8.8" PRIx64 ": %s", op_offset, op >= DW_OP_APPLE_uninit ? "DW_OP_APPLE_" : "DW_OP_");
            break;
        }

        switch (op)
        {
        case DW_OP_addr:    *s << "DW_OP_addr(" << m_data.GetAddress(&offset) << ") "; break;         // 0x03 1 address
        case DW_OP_deref:   *s << "DW_OP_deref"; break;                                               // 0x06
        case DW_OP_const1u: s->Printf("DW_OP_const1u(0x%2.2x) ", m_data.GetU8(&offset)); break;       // 0x08 1 1-byte constant
        case DW_OP_const1s: s->Printf("DW_OP_const1s(0x%2.2x) ", m_data.GetU8(&offset)); break;       // 0x09 1 1-byte constant
        case DW_OP_const2u: s->Printf("DW_OP_const2u(0x%4.4x) ", m_data.GetU16(&offset)); break;      // 0x0a 1 2-byte constant
        case DW_OP_const2s: s->Printf("DW_OP_const2s(0x%4.4x) ", m_data.GetU16(&offset)); break;      // 0x0b 1 2-byte constant
        case DW_OP_const4u: s->Printf("DW_OP_const4u(0x%8.8x) ", m_data.GetU32(&offset)); break;      // 0x0c 1 4-byte constant
        case DW_OP_const4s: s->Printf("DW_OP_const4s(0x%8.8x) ", m_data.GetU32(&offset)); break;      // 0x0d 1 4-byte constant
        case DW_OP_const8u: s->Printf("DW_OP_const8u(0x%16.16" PRIx64 ") ", m_data.GetU64(&offset)); break;  // 0x0e 1 8-byte constant
        case DW_OP_const8s: s->Printf("DW_OP_const8s(0x%16.16" PRIx64 ") ", m_data.GetU64(&offset)); break;  // 0x0f 1 8-byte constant
        case DW_OP_constu:  s->Printf("DW_OP_constu(0x%" PRIx64 ") ", m_data.GetULEB128(&offset)); break;    // 0x10 1 ULEB128 constant
        case DW_OP_consts:  s->Printf("DW_OP_consts(0x%" PRId64 ") ", m_data.GetSLEB128(&offset)); break;    // 0x11 1 SLEB128 constant
        case DW_OP_dup:     s->PutCString("DW_OP_dup"); break;                                        // 0x12
        case DW_OP_drop:    s->PutCString("DW_OP_drop"); break;                                       // 0x13
        case DW_OP_over:    s->PutCString("DW_OP_over"); break;                                       // 0x14
        case DW_OP_pick:    s->Printf("DW_OP_pick(0x%2.2x) ", m_data.GetU8(&offset)); break;          // 0x15 1 1-byte stack index
        case DW_OP_swap:    s->PutCString("DW_OP_swap"); break;                                       // 0x16
        case DW_OP_rot:     s->PutCString("DW_OP_rot"); break;                                        // 0x17
        case DW_OP_xderef:  s->PutCString("DW_OP_xderef"); break;                                     // 0x18
        case DW_OP_abs:     s->PutCString("DW_OP_abs"); break;                                        // 0x19
        case DW_OP_and:     s->PutCString("DW_OP_and"); break;                                        // 0x1a
        case DW_OP_div:     s->PutCString("DW_OP_div"); break;                                        // 0x1b
        case DW_OP_minus:   s->PutCString("DW_OP_minus"); break;                                      // 0x1c
        case DW_OP_mod:     s->PutCString("DW_OP_mod"); break;                                        // 0x1d
        case DW_OP_mul:     s->PutCString("DW_OP_mul"); break;                                        // 0x1e
        case DW_OP_neg:     s->PutCString("DW_OP_neg"); break;                                        // 0x1f
        case DW_OP_not:     s->PutCString("DW_OP_not"); break;                                        // 0x20
        case DW_OP_or:      s->PutCString("DW_OP_or"); break;                                         // 0x21
        case DW_OP_plus:    s->PutCString("DW_OP_plus"); break;                                       // 0x22
        case DW_OP_plus_uconst:                                                                 // 0x23 1 ULEB128 addend
            s->Printf("DW_OP_plus_uconst(0x%" PRIx64 ") ", m_data.GetULEB128(&offset));
            break;

        case DW_OP_shl:     s->PutCString("DW_OP_shl"); break;                                        // 0x24
        case DW_OP_shr:     s->PutCString("DW_OP_shr"); break;                                        // 0x25
        case DW_OP_shra:    s->PutCString("DW_OP_shra"); break;                                       // 0x26
        case DW_OP_xor:     s->PutCString("DW_OP_xor"); break;                                        // 0x27
        case DW_OP_skip:    s->Printf("DW_OP_skip(0x%4.4x)", m_data.GetU16(&offset)); break;          // 0x2f 1 signed 2-byte constant
        case DW_OP_bra:     s->Printf("DW_OP_bra(0x%4.4x)", m_data.GetU16(&offset)); break;           // 0x28 1 signed 2-byte constant
        case DW_OP_eq:      s->PutCString("DW_OP_eq"); break;                                         // 0x29
        case DW_OP_ge:      s->PutCString("DW_OP_ge"); break;                                         // 0x2a
        case DW_OP_gt:      s->PutCString("DW_OP_gt"); break;                                         // 0x2b
        case DW_OP_le:      s->PutCString("DW_OP_le"); break;                                         // 0x2c
        case DW_OP_lt:      s->PutCString("DW_OP_lt"); break;                                         // 0x2d
        case DW_OP_ne:      s->PutCString("DW_OP_ne"); break;                                         // 0x2e

        case DW_OP_lit0:    // 0x30
        case DW_OP_lit1:    // 0x31
        case DW_OP_lit2:    // 0x32
        case DW_OP_lit3:    // 0x33
        case DW_OP_lit4:    // 0x34
        case DW_OP_lit5:    // 0x35
        case DW_OP_lit6:    // 0x36
        case DW_OP_lit7:    // 0x37
        case DW_OP_lit8:    // 0x38
        case DW_OP_lit9:    // 0x39
        case DW_OP_lit10:   // 0x3A
        case DW_OP_lit11:   // 0x3B
        case DW_OP_lit12:   // 0x3C
        case DW_OP_lit13:   // 0x3D
        case DW_OP_lit14:   // 0x3E
        case DW_OP_lit15:   // 0x3F
        case DW_OP_lit16:   // 0x40
        case DW_OP_lit17:   // 0x41
        case DW_OP_lit18:   // 0x42
        case DW_OP_lit19:   // 0x43
        case DW_OP_lit20:   // 0x44
        case DW_OP_lit21:   // 0x45
        case DW_OP_lit22:   // 0x46
        case DW_OP_lit23:   // 0x47
        case DW_OP_lit24:   // 0x48
        case DW_OP_lit25:   // 0x49
        case DW_OP_lit26:   // 0x4A
        case DW_OP_lit27:   // 0x4B
        case DW_OP_lit28:   // 0x4C
        case DW_OP_lit29:   // 0x4D
        case DW_OP_lit30:   // 0x4E
        case DW_OP_lit31:   s->Printf("DW_OP_lit%i", op - DW_OP_lit0); break; // 0x4f

        case DW_OP_reg0:    // 0x50
        case DW_OP_reg1:    // 0x51
        case DW_OP_reg2:    // 0x52
        case DW_OP_reg3:    // 0x53
        case DW_OP_reg4:    // 0x54
        case DW_OP_reg5:    // 0x55
        case DW_OP_reg6:    // 0x56
        case DW_OP_reg7:    // 0x57
        case DW_OP_reg8:    // 0x58
        case DW_OP_reg9:    // 0x59
        case DW_OP_reg10:   // 0x5A
        case DW_OP_reg11:   // 0x5B
        case DW_OP_reg12:   // 0x5C
        case DW_OP_reg13:   // 0x5D
        case DW_OP_reg14:   // 0x5E
        case DW_OP_reg15:   // 0x5F
        case DW_OP_reg16:   // 0x60
        case DW_OP_reg17:   // 0x61
        case DW_OP_reg18:   // 0x62
        case DW_OP_reg19:   // 0x63
        case DW_OP_reg20:   // 0x64
        case DW_OP_reg21:   // 0x65
        case DW_OP_reg22:   // 0x66
        case DW_OP_reg23:   // 0x67
        case DW_OP_reg24:   // 0x68
        case DW_OP_reg25:   // 0x69
        case DW_OP_reg26:   // 0x6A
        case DW_OP_reg27:   // 0x6B
        case DW_OP_reg28:   // 0x6C
        case DW_OP_reg29:   // 0x6D
        case DW_OP_reg30:   // 0x6E
        case DW_OP_reg31:   // 0x6F
            {
                uint32_t reg_num = op - DW_OP_reg0;
                if (abi)
                {
                    RegisterInfo reg_info;
                    if (abi->GetRegisterInfoByKind(m_reg_kind, reg_num, reg_info))
                    {
                        if (reg_info.name)
                        {
                            s->PutCString (reg_info.name);
                            break;
                        }
                        else if (reg_info.alt_name)
                        {
                            s->PutCString (reg_info.alt_name);
                            break;
                        }
                    }
                }
                s->Printf("DW_OP_reg%u", reg_num); break; 
            }
            break;

        case DW_OP_breg0:
        case DW_OP_breg1:
        case DW_OP_breg2:
        case DW_OP_breg3:
        case DW_OP_breg4:
        case DW_OP_breg5:
        case DW_OP_breg6:
        case DW_OP_breg7:
        case DW_OP_breg8:
        case DW_OP_breg9:
        case DW_OP_breg10:
        case DW_OP_breg11:
        case DW_OP_breg12:
        case DW_OP_breg13:
        case DW_OP_breg14:
        case DW_OP_breg15:
        case DW_OP_breg16:
        case DW_OP_breg17:
        case DW_OP_breg18:
        case DW_OP_breg19:
        case DW_OP_breg20:
        case DW_OP_breg21:
        case DW_OP_breg22:
        case DW_OP_breg23:
        case DW_OP_breg24:
        case DW_OP_breg25:
        case DW_OP_breg26:
        case DW_OP_breg27:
        case DW_OP_breg28:
        case DW_OP_breg29:
        case DW_OP_breg30:
        case DW_OP_breg31:  
            {
                uint32_t reg_num = op - DW_OP_breg0;
                int64_t reg_offset = m_data.GetSLEB128(&offset);
                if (abi)
                {
                    RegisterInfo reg_info;
                    if (abi->GetRegisterInfoByKind(m_reg_kind, reg_num, reg_info))
                    {
                        if (reg_info.name)
                        {
                            s->Printf("[%s%+" PRIi64 "]", reg_info.name, reg_offset);
                            break;
                        }
                        else if (reg_info.alt_name)
                        {
                            s->Printf("[%s%+" PRIi64 "]", reg_info.alt_name, reg_offset);
                            break;
                        }
                    }
                }
                s->Printf("DW_OP_breg%i(0x%" PRIx64 ")", reg_num, reg_offset);
            }
            break;

        case DW_OP_regx:                                                    // 0x90 1 ULEB128 register
            {
                uint32_t reg_num = m_data.GetULEB128(&offset);
                if (abi)
                {
                    RegisterInfo reg_info;
                    if (abi->GetRegisterInfoByKind(m_reg_kind, reg_num, reg_info))
                    {
                        if (reg_info.name)
                        {
                            s->PutCString (reg_info.name);
                            break;
                        }
                        else if (reg_info.alt_name)
                        {
                            s->PutCString (reg_info.alt_name);
                            break;
                        }
                    }
                }
                s->Printf("DW_OP_regx(%" PRIu32 ")", reg_num); break;
            }
            break;
        case DW_OP_fbreg:                                                   // 0x91 1 SLEB128 offset
            s->Printf("DW_OP_fbreg(%" PRIi64 ")",m_data.GetSLEB128(&offset));
            break;
        case DW_OP_bregx:                                                   // 0x92 2 ULEB128 register followed by SLEB128 offset
            {
                uint32_t reg_num = m_data.GetULEB128(&offset);
                int64_t reg_offset = m_data.GetSLEB128(&offset);
                if (abi)
                {
                    RegisterInfo reg_info;
                    if (abi->GetRegisterInfoByKind(m_reg_kind, reg_num, reg_info))
                    {
                        if (reg_info.name)
                        {
                            s->Printf("[%s%+" PRIi64 "]", reg_info.name, reg_offset);
                            break;
                        }
                        else if (reg_info.alt_name)
                        {
                            s->Printf("[%s%+" PRIi64 "]", reg_info.alt_name, reg_offset);
                            break;
                        }
                    }
                }
                s->Printf("DW_OP_bregx(reg=%" PRIu32 ",offset=%" PRIi64 ")", reg_num, reg_offset);
            }
            break;
        case DW_OP_piece:                                                   // 0x93 1 ULEB128 size of piece addressed
            s->Printf("DW_OP_piece(0x%" PRIx64 ")", m_data.GetULEB128(&offset));
            break;
        case DW_OP_deref_size:                                              // 0x94 1 1-byte size of data retrieved
            s->Printf("DW_OP_deref_size(0x%2.2x)", m_data.GetU8(&offset));
            break;
        case DW_OP_xderef_size:                                             // 0x95 1 1-byte size of data retrieved
            s->Printf("DW_OP_xderef_size(0x%2.2x)", m_data.GetU8(&offset));
            break;
        case DW_OP_nop: s->PutCString("DW_OP_nop"); break;                                    // 0x96
        case DW_OP_push_object_address: s->PutCString("DW_OP_push_object_address"); break;    // 0x97 DWARF3
        case DW_OP_call2:                                                   // 0x98 DWARF3 1 2-byte offset of DIE
            s->Printf("DW_OP_call2(0x%4.4x)", m_data.GetU16(&offset));
            break;
        case DW_OP_call4:                                                   // 0x99 DWARF3 1 4-byte offset of DIE
            s->Printf("DW_OP_call4(0x%8.8x)", m_data.GetU32(&offset));
            break;
        case DW_OP_call_ref:                                                // 0x9a DWARF3 1 4- or 8-byte offset of DIE
            s->Printf("DW_OP_call_ref(0x%8.8" PRIx64 ")", m_data.GetAddress(&offset));
            break;
//      case DW_OP_form_tls_address: s << "form_tls_address"; break;        // 0x9b DWARF3
//      case DW_OP_call_frame_cfa: s << "call_frame_cfa"; break;            // 0x9c DWARF3
//      case DW_OP_bit_piece:                                               // 0x9d DWARF3 2
//          s->Printf("DW_OP_bit_piece(0x%x, 0x%x)", m_data.GetULEB128(&offset), m_data.GetULEB128(&offset));
//          break;
//      case DW_OP_lo_user:     s->PutCString("DW_OP_lo_user"); break;                        // 0xe0
//      case DW_OP_hi_user:     s->PutCString("DW_OP_hi_user"); break;                        // 0xff
//        case DW_OP_APPLE_extern:
//            s->Printf("DW_OP_APPLE_extern(%" PRIu64 ")", m_data.GetULEB128(&offset));
//            break;
//        case DW_OP_APPLE_array_ref:
//            s->PutCString("DW_OP_APPLE_array_ref");
//            break;
        case DW_OP_APPLE_uninit:
            s->PutCString("DW_OP_APPLE_uninit");  // 0xF0
            break;
//        case DW_OP_APPLE_assign:        // 0xF1 - pops value off and assigns it to second item on stack (2nd item must have assignable context)
//            s->PutCString("DW_OP_APPLE_assign");
//            break;
//        case DW_OP_APPLE_address_of:    // 0xF2 - gets the address of the top stack item (top item must be a variable, or have value_type that is an address already)
//            s->PutCString("DW_OP_APPLE_address_of");
//            break;
//        case DW_OP_APPLE_value_of:      // 0xF3 - pops the value off the stack and pushes the value of that object (top item must be a variable, or expression local)
//            s->PutCString("DW_OP_APPLE_value_of");
//            break;
//        case DW_OP_APPLE_deref_type:    // 0xF4 - gets the address of the top stack item (top item must be a variable, or a clang type)
//            s->PutCString("DW_OP_APPLE_deref_type");
//            break;
//        case DW_OP_APPLE_expr_local:    // 0xF5 - ULEB128 expression local index
//            s->Printf("DW_OP_APPLE_expr_local(%" PRIu64 ")", m_data.GetULEB128(&offset));
//            break;
//        case DW_OP_APPLE_constf:        // 0xF6 - 1 byte float size, followed by constant float data
//            {
//                uint8_t float_length = m_data.GetU8(&offset);
//                s->Printf("DW_OP_APPLE_constf(<%u> ", float_length);
//                m_data.Dump(s, offset, eFormatHex, float_length, 1, UINT32_MAX, DW_INVALID_ADDRESS, 0, 0);
//                s->PutChar(')');
//                // Consume the float data
//                m_data.GetData(&offset, float_length);
//            }
//            break;
//        case DW_OP_APPLE_scalar_cast:
//            s->Printf("DW_OP_APPLE_scalar_cast(%s)", Scalar::GetValueTypeAsCString ((Scalar::Type)m_data.GetU8(&offset)));
//            break;
//        case DW_OP_APPLE_clang_cast:
//            {
//                clang::Type *clang_type = (clang::Type *)m_data.GetMaxU64(&offset, sizeof(void*));
//                s->Printf("DW_OP_APPLE_clang_cast(%p)", clang_type);
//            }
//            break;
//        case DW_OP_APPLE_clear:
//            s->PutCString("DW_OP_APPLE_clear");
//            break;
//        case DW_OP_APPLE_error:         // 0xFF - Stops expression evaluation and returns an error (no args)
//            s->PutCString("DW_OP_APPLE_error");
//            break;
        }
    }
}

void
DWARFExpression::SetLocationListSlide (addr_t slide)
{
    m_loclist_slide = slide;
}

int
DWARFExpression::GetRegisterKind ()
{
    return m_reg_kind;
}

void
DWARFExpression::SetRegisterKind (RegisterKind reg_kind)
{
    m_reg_kind = reg_kind;
}

bool
DWARFExpression::IsLocationList() const
{
    return m_loclist_slide != LLDB_INVALID_ADDRESS;
}

void
DWARFExpression::GetDescription (Stream *s, lldb::DescriptionLevel level, addr_t location_list_base_addr, ABI *abi) const
{
    if (IsLocationList())
    {
        // We have a location list
        lldb::offset_t offset = 0;
        uint32_t count = 0;
        addr_t curr_base_addr = location_list_base_addr;
        while (m_data.ValidOffset(offset))
        {
            lldb::addr_t begin_addr_offset = m_data.GetAddress(&offset);
            lldb::addr_t end_addr_offset = m_data.GetAddress(&offset);
            if (begin_addr_offset < end_addr_offset)
            {
                if (count > 0)
                    s->PutCString(", ");
                VMRange addr_range(curr_base_addr + begin_addr_offset, curr_base_addr + end_addr_offset);
                addr_range.Dump(s, 0, 8);
                s->PutChar('{');
                lldb::offset_t location_length = m_data.GetU16(&offset);
                DumpLocation (s, offset, location_length, level, abi);
                s->PutChar('}');
                offset += location_length;
            }
            else if (begin_addr_offset == 0 && end_addr_offset == 0)
            {
                // The end of the location list is marked by both the start and end offset being zero
                break;
            }
            else
            {
                if ((m_data.GetAddressByteSize() == 4 && (begin_addr_offset == UINT32_MAX)) ||
                    (m_data.GetAddressByteSize() == 8 && (begin_addr_offset == UINT64_MAX)))
                {
                    curr_base_addr = end_addr_offset + location_list_base_addr;
                    // We have a new base address
                    if (count > 0)
                        s->PutCString(", ");
                    *s << "base_addr = " << end_addr_offset;
                }
            }

            count++;
        }
    }
    else
    {
        // We have a normal location that contains DW_OP location opcodes
        DumpLocation (s, 0, m_data.GetByteSize(), level, abi);
    }
}

static bool
ReadRegisterValueAsScalar
(
    RegisterContext *reg_ctx,
    uint32_t reg_kind,
    uint32_t reg_num,
    Error *error_ptr,
    Value &value
)
{
    if (reg_ctx == NULL)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("No register context in frame.\n");
    }
    else
    {
        uint32_t native_reg = reg_ctx->ConvertRegisterKindToRegisterNumber(reg_kind, reg_num);
        if (native_reg == LLDB_INVALID_REGNUM)
        {
            if (error_ptr)
                error_ptr->SetErrorStringWithFormat("Unable to convert register kind=%u reg_num=%u to a native register number.\n", reg_kind, reg_num);
        }
        else
        {
            const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoAtIndex(native_reg);
            RegisterValue reg_value;
            if (reg_ctx->ReadRegister (reg_info, reg_value))
            {
                if (reg_value.GetScalarValue(value.GetScalar()))
                {
                    value.SetValueType (Value::eValueTypeScalar);
                    value.SetContext (Value::eContextTypeRegisterInfo, 
                                      const_cast<RegisterInfo *>(reg_info));
                    if (error_ptr)
                        error_ptr->Clear();
                    return true;
                }
                else
                {
                    // If we get this error, then we need to implement a value
                    // buffer in the dwarf expression evaluation function...
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("register %s can't be converted to a scalar value", 
                                                             reg_info->name);
                }
            }
            else
            {
                if (error_ptr)
                    error_ptr->SetErrorStringWithFormat("register %s is not available", reg_info->name);
            }
        }
    }
    return false;
}

//bool
//DWARFExpression::LocationListContainsLoadAddress (Process* process, const Address &addr) const
//{
//    return LocationListContainsLoadAddress(process, addr.GetLoadAddress(process));
//}
//
//bool
//DWARFExpression::LocationListContainsLoadAddress (Process* process, addr_t load_addr) const
//{
//    if (load_addr == LLDB_INVALID_ADDRESS)
//        return false;
//
//    if (IsLocationList())
//    {
//        lldb::offset_t offset = 0;
//
//        addr_t loc_list_base_addr = m_loclist_slide.GetLoadAddress(process);
//
//        if (loc_list_base_addr == LLDB_INVALID_ADDRESS)
//            return false;
//
//        while (m_data.ValidOffset(offset))
//        {
//            // We need to figure out what the value is for the location.
//            addr_t lo_pc = m_data.GetAddress(&offset);
//            addr_t hi_pc = m_data.GetAddress(&offset);
//            if (lo_pc == 0 && hi_pc == 0)
//                break;
//            else
//            {
//                lo_pc += loc_list_base_addr;
//                hi_pc += loc_list_base_addr;
//
//                if (lo_pc <= load_addr && load_addr < hi_pc)
//                    return true;
//
//                offset += m_data.GetU16(&offset);
//            }
//        }
//    }
//    return false;
//}

static offset_t
GetOpcodeDataSize (const DataExtractor &data, const lldb::offset_t data_offset, const uint8_t op)
{
    lldb::offset_t offset = data_offset;
    switch (op)
    {
        case DW_OP_addr:    
        case DW_OP_call_ref:    // 0x9a 1 address sized offset of DIE (DWARF3)
            return data.GetAddressByteSize();

        // Opcodes with no arguments
        case DW_OP_deref:   // 0x06
        case DW_OP_dup:     // 0x12
        case DW_OP_drop:    // 0x13
        case DW_OP_over:    // 0x14
        case DW_OP_swap:    // 0x16
        case DW_OP_rot:     // 0x17
        case DW_OP_xderef:  // 0x18
        case DW_OP_abs:     // 0x19
        case DW_OP_and:     // 0x1a
        case DW_OP_div:     // 0x1b
        case DW_OP_minus:   // 0x1c
        case DW_OP_mod:     // 0x1d
        case DW_OP_mul:     // 0x1e
        case DW_OP_neg:     // 0x1f
        case DW_OP_not:     // 0x20
        case DW_OP_or:      // 0x21
        case DW_OP_plus:    // 0x22
        case DW_OP_shl:     // 0x24
        case DW_OP_shr:     // 0x25
        case DW_OP_shra:    // 0x26
        case DW_OP_xor:     // 0x27
        case DW_OP_eq:      // 0x29
        case DW_OP_ge:      // 0x2a
        case DW_OP_gt:      // 0x2b
        case DW_OP_le:      // 0x2c
        case DW_OP_lt:      // 0x2d
        case DW_OP_ne:      // 0x2e
        case DW_OP_lit0:    // 0x30
        case DW_OP_lit1:    // 0x31
        case DW_OP_lit2:    // 0x32
        case DW_OP_lit3:    // 0x33
        case DW_OP_lit4:    // 0x34
        case DW_OP_lit5:    // 0x35
        case DW_OP_lit6:    // 0x36
        case DW_OP_lit7:    // 0x37
        case DW_OP_lit8:    // 0x38
        case DW_OP_lit9:    // 0x39
        case DW_OP_lit10:   // 0x3A
        case DW_OP_lit11:   // 0x3B
        case DW_OP_lit12:   // 0x3C
        case DW_OP_lit13:   // 0x3D
        case DW_OP_lit14:   // 0x3E
        case DW_OP_lit15:   // 0x3F
        case DW_OP_lit16:   // 0x40
        case DW_OP_lit17:   // 0x41
        case DW_OP_lit18:   // 0x42
        case DW_OP_lit19:   // 0x43
        case DW_OP_lit20:   // 0x44
        case DW_OP_lit21:   // 0x45
        case DW_OP_lit22:   // 0x46
        case DW_OP_lit23:   // 0x47
        case DW_OP_lit24:   // 0x48
        case DW_OP_lit25:   // 0x49
        case DW_OP_lit26:   // 0x4A
        case DW_OP_lit27:   // 0x4B
        case DW_OP_lit28:   // 0x4C
        case DW_OP_lit29:   // 0x4D
        case DW_OP_lit30:   // 0x4E
        case DW_OP_lit31:   // 0x4f
        case DW_OP_reg0:    // 0x50
        case DW_OP_reg1:    // 0x51
        case DW_OP_reg2:    // 0x52
        case DW_OP_reg3:    // 0x53
        case DW_OP_reg4:    // 0x54
        case DW_OP_reg5:    // 0x55
        case DW_OP_reg6:    // 0x56
        case DW_OP_reg7:    // 0x57
        case DW_OP_reg8:    // 0x58
        case DW_OP_reg9:    // 0x59
        case DW_OP_reg10:   // 0x5A
        case DW_OP_reg11:   // 0x5B
        case DW_OP_reg12:   // 0x5C
        case DW_OP_reg13:   // 0x5D
        case DW_OP_reg14:   // 0x5E
        case DW_OP_reg15:   // 0x5F
        case DW_OP_reg16:   // 0x60
        case DW_OP_reg17:   // 0x61
        case DW_OP_reg18:   // 0x62
        case DW_OP_reg19:   // 0x63
        case DW_OP_reg20:   // 0x64
        case DW_OP_reg21:   // 0x65
        case DW_OP_reg22:   // 0x66
        case DW_OP_reg23:   // 0x67
        case DW_OP_reg24:   // 0x68
        case DW_OP_reg25:   // 0x69
        case DW_OP_reg26:   // 0x6A
        case DW_OP_reg27:   // 0x6B
        case DW_OP_reg28:   // 0x6C
        case DW_OP_reg29:   // 0x6D
        case DW_OP_reg30:   // 0x6E
        case DW_OP_reg31:   // 0x6F
        case DW_OP_nop:     // 0x96
        case DW_OP_push_object_address: // 0x97 DWARF3
        case DW_OP_form_tls_address:    // 0x9b DWARF3
        case DW_OP_call_frame_cfa:      // 0x9c DWARF3
        case DW_OP_stack_value: // 0x9f DWARF4
            return 0; 

        // Opcodes with a single 1 byte arguments
        case DW_OP_const1u:     // 0x08 1 1-byte constant
        case DW_OP_const1s:     // 0x09 1 1-byte constant
        case DW_OP_pick:        // 0x15 1 1-byte stack index
        case DW_OP_deref_size:  // 0x94 1 1-byte size of data retrieved
        case DW_OP_xderef_size: // 0x95 1 1-byte size of data retrieved
            return 1;
            
        // Opcodes with a single 2 byte arguments
        case DW_OP_const2u:     // 0x0a 1 2-byte constant
        case DW_OP_const2s:     // 0x0b 1 2-byte constant
        case DW_OP_skip:        // 0x2f 1 signed 2-byte constant
        case DW_OP_bra:         // 0x28 1 signed 2-byte constant
        case DW_OP_call2:       // 0x98 1 2-byte offset of DIE (DWARF3)
            return 2;
            
        // Opcodes with a single 4 byte arguments
        case DW_OP_const4u:     // 0x0c 1 4-byte constant
        case DW_OP_const4s:     // 0x0d 1 4-byte constant
        case DW_OP_call4:       // 0x99 1 4-byte offset of DIE (DWARF3)
            return 4;
            
        // Opcodes with a single 8 byte arguments
        case DW_OP_const8u:     // 0x0e 1 8-byte constant
        case DW_OP_const8s:     // 0x0f 1 8-byte constant
             return 8;
            
        // All opcodes that have a single ULEB (signed or unsigned) argument
        case DW_OP_constu:      // 0x10 1 ULEB128 constant
        case DW_OP_consts:      // 0x11 1 SLEB128 constant
        case DW_OP_plus_uconst: // 0x23 1 ULEB128 addend
        case DW_OP_breg0:       // 0x70 1 ULEB128 register
        case DW_OP_breg1:       // 0x71 1 ULEB128 register
        case DW_OP_breg2:       // 0x72 1 ULEB128 register
        case DW_OP_breg3:       // 0x73 1 ULEB128 register
        case DW_OP_breg4:       // 0x74 1 ULEB128 register
        case DW_OP_breg5:       // 0x75 1 ULEB128 register
        case DW_OP_breg6:       // 0x76 1 ULEB128 register
        case DW_OP_breg7:       // 0x77 1 ULEB128 register
        case DW_OP_breg8:       // 0x78 1 ULEB128 register
        case DW_OP_breg9:       // 0x79 1 ULEB128 register
        case DW_OP_breg10:      // 0x7a 1 ULEB128 register
        case DW_OP_breg11:      // 0x7b 1 ULEB128 register
        case DW_OP_breg12:      // 0x7c 1 ULEB128 register
        case DW_OP_breg13:      // 0x7d 1 ULEB128 register
        case DW_OP_breg14:      // 0x7e 1 ULEB128 register
        case DW_OP_breg15:      // 0x7f 1 ULEB128 register
        case DW_OP_breg16:      // 0x80 1 ULEB128 register
        case DW_OP_breg17:      // 0x81 1 ULEB128 register
        case DW_OP_breg18:      // 0x82 1 ULEB128 register
        case DW_OP_breg19:      // 0x83 1 ULEB128 register
        case DW_OP_breg20:      // 0x84 1 ULEB128 register
        case DW_OP_breg21:      // 0x85 1 ULEB128 register
        case DW_OP_breg22:      // 0x86 1 ULEB128 register
        case DW_OP_breg23:      // 0x87 1 ULEB128 register
        case DW_OP_breg24:      // 0x88 1 ULEB128 register
        case DW_OP_breg25:      // 0x89 1 ULEB128 register
        case DW_OP_breg26:      // 0x8a 1 ULEB128 register
        case DW_OP_breg27:      // 0x8b 1 ULEB128 register
        case DW_OP_breg28:      // 0x8c 1 ULEB128 register
        case DW_OP_breg29:      // 0x8d 1 ULEB128 register
        case DW_OP_breg30:      // 0x8e 1 ULEB128 register
        case DW_OP_breg31:      // 0x8f 1 ULEB128 register
        case DW_OP_regx:        // 0x90 1 ULEB128 register
        case DW_OP_fbreg:       // 0x91 1 SLEB128 offset
        case DW_OP_piece:       // 0x93 1 ULEB128 size of piece addressed
            data.Skip_LEB128(&offset); 
            return offset - data_offset;   
            
            // All opcodes that have a 2 ULEB (signed or unsigned) arguments
        case DW_OP_bregx:       // 0x92 2 ULEB128 register followed by SLEB128 offset
        case DW_OP_bit_piece:   // 0x9d ULEB128 bit size, ULEB128 bit offset (DWARF3);
            data.Skip_LEB128(&offset); 
            data.Skip_LEB128(&offset); 
            return offset - data_offset;   

        case DW_OP_implicit_value: // 0x9e ULEB128 size followed by block of that size (DWARF4)
            {
                uint64_t block_len = data.Skip_LEB128(&offset); 
                offset += block_len;
                return offset - data_offset;   
            }
            
        default:
            break;
    }
    return LLDB_INVALID_OFFSET;
}

lldb::addr_t
DWARFExpression::GetLocation_DW_OP_addr (uint32_t op_addr_idx, bool &error) const
{
    error = false;
    if (IsLocationList())
        return LLDB_INVALID_ADDRESS;
    lldb::offset_t offset = 0;
    uint32_t curr_op_addr_idx = 0;
    while (m_data.ValidOffset(offset))
    {
        const uint8_t op = m_data.GetU8(&offset);
        
        if (op == DW_OP_addr)
        {
            const lldb::addr_t op_file_addr = m_data.GetAddress(&offset);
            if (curr_op_addr_idx == op_addr_idx)
                return op_file_addr;
            else
                ++curr_op_addr_idx;
        }
        else
        {
            const offset_t op_arg_size = GetOpcodeDataSize (m_data, offset, op);
            if (op_arg_size == LLDB_INVALID_OFFSET)
            {
                error = true;
                break;
            }
            offset += op_arg_size;
        }
    }
    return LLDB_INVALID_ADDRESS;
}

bool
DWARFExpression::Update_DW_OP_addr (lldb::addr_t file_addr)
{
    if (IsLocationList())
        return false;
    lldb::offset_t offset = 0;
    while (m_data.ValidOffset(offset))
    {
        const uint8_t op = m_data.GetU8(&offset);
        
        if (op == DW_OP_addr)
        {
            const uint32_t addr_byte_size = m_data.GetAddressByteSize();
            // We have to make a copy of the data as we don't know if this
            // data is from a read only memory mapped buffer, so we duplicate
            // all of the data first, then modify it, and if all goes well,
            // we then replace the data for this expression
            
            // So first we copy the data into a heap buffer
            std::unique_ptr<DataBufferHeap> head_data_ap (new DataBufferHeap (m_data.GetDataStart(),
                                                                             m_data.GetByteSize()));
            
            // Make en encoder so we can write the address into the buffer using
            // the correct byte order (endianness)
            DataEncoder encoder (head_data_ap->GetBytes(), 
                                 head_data_ap->GetByteSize(),
                                 m_data.GetByteOrder(),
                                 addr_byte_size);
            
            // Replace the address in the new buffer
            if (encoder.PutMaxU64 (offset, addr_byte_size, file_addr) == UINT32_MAX)
                return false;

            // All went well, so now we can reset the data using a shared
            // pointer to the heap data so "m_data" will now correctly 
            // manage the heap data.
            m_data.SetData (DataBufferSP (head_data_ap.release()));
            return true;
        }
        else
        {
            const offset_t op_arg_size = GetOpcodeDataSize (m_data, offset, op);
            if (op_arg_size == LLDB_INVALID_OFFSET)
                break;
            offset += op_arg_size;
        }
    }
    return false;
}

bool
DWARFExpression::LocationListContainsAddress (lldb::addr_t loclist_base_addr, lldb::addr_t addr) const
{
    if (addr == LLDB_INVALID_ADDRESS)
        return false;

    if (IsLocationList())
    {
        lldb::offset_t offset = 0;

        if (loclist_base_addr == LLDB_INVALID_ADDRESS)
            return false;

        while (m_data.ValidOffset(offset))
        {
            // We need to figure out what the value is for the location.
            addr_t lo_pc = m_data.GetAddress(&offset);
            addr_t hi_pc = m_data.GetAddress(&offset);
            if (lo_pc == 0 && hi_pc == 0)
                break;
            else
            {
                lo_pc += loclist_base_addr - m_loclist_slide;
                hi_pc += loclist_base_addr - m_loclist_slide;

                if (lo_pc <= addr && addr < hi_pc)
                    return true;

                offset += m_data.GetU16(&offset);
            }
        }
    }
    return false;
}

bool
DWARFExpression::GetLocation (addr_t base_addr, addr_t pc, lldb::offset_t &offset, lldb::offset_t &length)
{
    offset = 0;
    if (!IsLocationList())
    {
        length = m_data.GetByteSize();
        return true;
    }

    if (base_addr != LLDB_INVALID_ADDRESS && pc != LLDB_INVALID_ADDRESS)
    {
        addr_t curr_base_addr = base_addr;

        while (m_data.ValidOffset(offset))
        {
            // We need to figure out what the value is for the location.
            addr_t lo_pc = m_data.GetAddress(&offset);
            addr_t hi_pc = m_data.GetAddress(&offset);
            if (lo_pc == 0 && hi_pc == 0)
            {
                break;
            }
            else
            {
                lo_pc += curr_base_addr - m_loclist_slide;
                hi_pc += curr_base_addr - m_loclist_slide;
                
                length = m_data.GetU16(&offset);
                
                if (length > 0 && lo_pc <= pc && pc < hi_pc)
                    return true;

                offset += length;
            }
        }
    }
    offset = LLDB_INVALID_OFFSET;
    length = 0;
    return false;
}

bool
DWARFExpression::DumpLocationForAddress (Stream *s,
                                         lldb::DescriptionLevel level,
                                         addr_t base_addr,
                                         addr_t address,
                                         ABI *abi)
{
    lldb::offset_t offset = 0;
    lldb::offset_t length = 0;
    
    if (GetLocation (base_addr, address, offset, length))
    {
        if (length > 0)
        {
            DumpLocation(s, offset, length, level, abi);
            return true;
        }
    }
    return false;
}

bool
DWARFExpression::Evaluate
(
    ExecutionContextScope *exe_scope,
    clang::ASTContext *ast_context,
    ClangExpressionVariableList *expr_locals,
    ClangExpressionDeclMap *decl_map,
    lldb::addr_t loclist_base_load_addr,
    const Value* initial_value_ptr,
    Value& result,
    Error *error_ptr
) const
{
    ExecutionContext exe_ctx (exe_scope);
    return Evaluate(&exe_ctx, ast_context, expr_locals, decl_map, NULL, loclist_base_load_addr, initial_value_ptr, result, error_ptr);
}

bool
DWARFExpression::Evaluate
(
    ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    ClangExpressionVariableList *expr_locals,
    ClangExpressionDeclMap *decl_map,
    RegisterContext *reg_ctx,
    lldb::addr_t loclist_base_load_addr,
    const Value* initial_value_ptr,
    Value& result,
    Error *error_ptr
) const
{
    if (IsLocationList())
    {
        lldb::offset_t offset = 0;
        addr_t pc;
        StackFrame *frame = NULL;
        if (reg_ctx)
            pc = reg_ctx->GetPC();
        else
        {
            frame = exe_ctx->GetFramePtr();
            if (!frame)
                return false;
            RegisterContextSP reg_ctx_sp = frame->GetRegisterContext();
            if (!reg_ctx_sp)
                return false;
            pc = reg_ctx_sp->GetPC();
        }

        if (loclist_base_load_addr != LLDB_INVALID_ADDRESS)
        {
            if (pc == LLDB_INVALID_ADDRESS)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Invalid PC in frame.");
                return false;
            }

            addr_t curr_loclist_base_load_addr = loclist_base_load_addr;

            while (m_data.ValidOffset(offset))
            {
                // We need to figure out what the value is for the location.
                addr_t lo_pc = m_data.GetAddress(&offset);
                addr_t hi_pc = m_data.GetAddress(&offset);
                if (lo_pc == 0 && hi_pc == 0)
                {
                    break;
                }
                else
                {
                    lo_pc += curr_loclist_base_load_addr - m_loclist_slide;
                    hi_pc += curr_loclist_base_load_addr - m_loclist_slide;

                    uint16_t length = m_data.GetU16(&offset);

                    if (length > 0 && lo_pc <= pc && pc < hi_pc)
                    {
                        return DWARFExpression::Evaluate (exe_ctx, ast_context, expr_locals, decl_map, reg_ctx, m_data, offset, length, m_reg_kind, initial_value_ptr, result, error_ptr);
                    }
                    offset += length;
                }
            }
        }
        if (error_ptr)
            error_ptr->SetErrorString ("variable not available");
        return false;
    }

    // Not a location list, just a single expression.
    return DWARFExpression::Evaluate (exe_ctx, ast_context, expr_locals, decl_map, reg_ctx, m_data, 0, m_data.GetByteSize(), m_reg_kind, initial_value_ptr, result, error_ptr);
}



bool
DWARFExpression::Evaluate
(
    ExecutionContext *exe_ctx,
    clang::ASTContext *ast_context,
    ClangExpressionVariableList *expr_locals,
    ClangExpressionDeclMap *decl_map,
    RegisterContext *reg_ctx,
    const DataExtractor& opcodes,
    const lldb::offset_t opcodes_offset,
    const lldb::offset_t opcodes_length,
    const uint32_t reg_kind,
    const Value* initial_value_ptr,
    Value& result,
    Error *error_ptr
)
{
    std::vector<Value> stack;

    Process *process = NULL;
    StackFrame *frame = NULL;
    
    if (exe_ctx)
    {
        process = exe_ctx->GetProcessPtr();
        frame = exe_ctx->GetFramePtr();
    }
    if (reg_ctx == NULL && frame)
        reg_ctx = frame->GetRegisterContext().get();

    if (initial_value_ptr)
        stack.push_back(*initial_value_ptr);

    lldb::offset_t offset = opcodes_offset;
    const lldb::offset_t end_offset = opcodes_offset + opcodes_length;
    Value tmp;
    uint32_t reg_num;

    // Make sure all of the data is available in opcodes.
    if (!opcodes.ValidOffsetForDataOfSize(opcodes_offset, opcodes_length))
    {
        if (error_ptr)
            error_ptr->SetErrorString ("Invalid offset and/or length for opcodes buffer.");
        return false;
    }
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));


    while (opcodes.ValidOffset(offset) && offset < end_offset)
    {
        const lldb::offset_t op_offset = offset;
        const uint8_t op = opcodes.GetU8(&offset);

        if (log && log->GetVerbose())
        {
            size_t count = stack.size();
            log->Printf("Stack before operation has %lu values:", count);
            for (size_t i=0; i<count; ++i)
            {
                StreamString new_value;
                new_value.Printf("[%" PRIu64 "]", (uint64_t)i);
                stack[i].Dump(&new_value);
                log->Printf("  %s", new_value.GetData());
            }
            log->Printf("0x%8.8" PRIx64 ": %s", op_offset, DW_OP_value_to_name(op));
        }
        switch (op)
        {
        //----------------------------------------------------------------------
        // The DW_OP_addr operation has a single operand that encodes a machine
        // address and whose size is the size of an address on the target machine.
        //----------------------------------------------------------------------
        case DW_OP_addr:
            stack.push_back(Scalar(opcodes.GetAddress(&offset)));
            stack.back().SetValueType (Value::eValueTypeFileAddress);
            break;

        //----------------------------------------------------------------------
        // The DW_OP_addr_sect_offset4 is used for any location expressions in
        // shared libraries that have a location like:
        //  DW_OP_addr(0x1000)
        // If this address resides in a shared library, then this virtual
        // address won't make sense when it is evaluated in the context of a
        // running process where shared libraries have been slid. To account for
        // this, this new address type where we can store the section pointer
        // and a 4 byte offset.
        //----------------------------------------------------------------------
//      case DW_OP_addr_sect_offset4:
//          {
//              result_type = eResultTypeFileAddress;
//              lldb::Section *sect = (lldb::Section *)opcodes.GetMaxU64(&offset, sizeof(void *));
//              lldb::addr_t sect_offset = opcodes.GetU32(&offset);
//
//              Address so_addr (sect, sect_offset);
//              lldb::addr_t load_addr = so_addr.GetLoadAddress();
//              if (load_addr != LLDB_INVALID_ADDRESS)
//              {
//                  // We successfully resolve a file address to a load
//                  // address.
//                  stack.push_back(load_addr);
//                  break;
//              }
//              else
//              {
//                  // We were able
//                  if (error_ptr)
//                      error_ptr->SetErrorStringWithFormat ("Section %s in %s is not currently loaded.\n", sect->GetName().AsCString(), sect->GetModule()->GetFileSpec().GetFilename().AsCString());
//                  return false;
//              }
//          }
//          break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_deref
        // OPERANDS: none
        // DESCRIPTION: Pops the top stack entry and treats it as an address.
        // The value retrieved from that address is pushed. The size of the
        // data retrieved from the dereferenced address is the size of an
        // address on the target machine.
        //----------------------------------------------------------------------
        case DW_OP_deref:
            {
                Value::ValueType value_type = stack.back().GetValueType();
                switch (value_type)
                {
                case Value::eValueTypeHostAddress:
                    {
                        void *src = (void *)stack.back().GetScalar().ULongLong();
                        intptr_t ptr;
                        ::memcpy (&ptr, src, sizeof(void *));
                        stack.back().GetScalar() = ptr;
                        stack.back().ClearContext();
                    }
                    break;
                case Value::eValueTypeLoadAddress:
                    if (exe_ctx)
                    {
                        if (process)
                        {
                            lldb::addr_t pointer_addr = stack.back().GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                            uint8_t addr_bytes[sizeof(lldb::addr_t)];
                            uint32_t addr_size = process->GetAddressByteSize();
                            Error error;
                            if (process->ReadMemory(pointer_addr, &addr_bytes, addr_size, error) == addr_size)
                            {
                                DataExtractor addr_data(addr_bytes, sizeof(addr_bytes), process->GetByteOrder(), addr_size);
                                lldb::offset_t addr_data_offset = 0;
                                stack.back().GetScalar() = addr_data.GetPointer(&addr_data_offset);
                                stack.back().ClearContext();
                            }
                            else
                            {
                                if (error_ptr)
                                    error_ptr->SetErrorStringWithFormat ("Failed to dereference pointer from 0x%" PRIx64 " for DW_OP_deref: %s\n",
                                                                         pointer_addr,
                                                                         error.AsCString());
                                return false;
                            }
                        }
                        else
                        {
                            if (error_ptr)
                                error_ptr->SetErrorStringWithFormat ("NULL process for DW_OP_deref.\n");
                            return false;
                        }
                    }
                    else
                    {
                        if (error_ptr)
                            error_ptr->SetErrorStringWithFormat ("NULL execution context for DW_OP_deref.\n");
                        return false;
                    }
                    break;

                default:
                    break;
                }

            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_deref_size
        // OPERANDS: 1
        //  1 - uint8_t that specifies the size of the data to dereference.
        // DESCRIPTION: Behaves like the DW_OP_deref operation: it pops the top
        // stack entry and treats it as an address. The value retrieved from that
        // address is pushed. In the DW_OP_deref_size operation, however, the
        // size in bytes of the data retrieved from the dereferenced address is
        // specified by the single operand. This operand is a 1-byte unsigned
        // integral constant whose value may not be larger than the size of an
        // address on the target machine. The data retrieved is zero extended
        // to the size of an address on the target machine before being pushed
        // on the expression stack.
        //----------------------------------------------------------------------
        case DW_OP_deref_size:
            {
                uint8_t size = opcodes.GetU8(&offset);
                Value::ValueType value_type = stack.back().GetValueType();
                switch (value_type)
                {
                case Value::eValueTypeHostAddress:
                    {
                        void *src = (void *)stack.back().GetScalar().ULongLong();
                        intptr_t ptr;
                        ::memcpy (&ptr, src, sizeof(void *));
                        // I can't decide whether the size operand should apply to the bytes in their
                        // lldb-host endianness or the target endianness.. I doubt this'll ever come up
                        // but I'll opt for assuming big endian regardless.
                        switch (size)
                        {
                            case 1: ptr = ptr & 0xff; break;
                            case 2: ptr = ptr & 0xffff; break;
                            case 3: ptr = ptr & 0xffffff; break;
                            case 4: ptr = ptr & 0xffffffff; break;
                            // the casts are added to work around the case where intptr_t is a 32 bit quantity;
                            // presumably we won't hit the 5..7 cases if (void*) is 32-bits in this program.
                            case 5: ptr = (intptr_t) ptr & 0xffffffffffULL; break;
                            case 6: ptr = (intptr_t) ptr & 0xffffffffffffULL; break;
                            case 7: ptr = (intptr_t) ptr & 0xffffffffffffffULL; break;
                            default: break;
                        }
                        stack.back().GetScalar() = ptr;
                        stack.back().ClearContext();
                    }
                    break;
                case Value::eValueTypeLoadAddress:
                    if (exe_ctx)
                    {
                        if (process)
                        {
                            lldb::addr_t pointer_addr = stack.back().GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                            uint8_t addr_bytes[sizeof(lldb::addr_t)];
                            Error error;
                            if (process->ReadMemory(pointer_addr, &addr_bytes, size, error) == size)
                            {
                                DataExtractor addr_data(addr_bytes, sizeof(addr_bytes), process->GetByteOrder(), size);
                                lldb::offset_t addr_data_offset = 0;
                                switch (size)
                                {
                                    case 1: stack.back().GetScalar() = addr_data.GetU8(&addr_data_offset); break;
                                    case 2: stack.back().GetScalar() = addr_data.GetU16(&addr_data_offset); break;
                                    case 4: stack.back().GetScalar() = addr_data.GetU32(&addr_data_offset); break;
                                    case 8: stack.back().GetScalar() = addr_data.GetU64(&addr_data_offset); break;
                                    default: stack.back().GetScalar() = addr_data.GetPointer(&addr_data_offset);
                                }
                                stack.back().ClearContext();
                            }
                            else
                            {
                                if (error_ptr)
                                    error_ptr->SetErrorStringWithFormat ("Failed to dereference pointer from 0x%" PRIx64 " for DW_OP_deref: %s\n",
                                                                         pointer_addr,
                                                                         error.AsCString());
                                return false;
                            }
                        }
                        else
                        {
                            if (error_ptr)
                                error_ptr->SetErrorStringWithFormat ("NULL process for DW_OP_deref.\n");
                            return false;
                        }
                    }
                    else
                    {
                        if (error_ptr)
                            error_ptr->SetErrorStringWithFormat ("NULL execution context for DW_OP_deref.\n");
                        return false;
                    }
                    break;

                default:
                    break;
                }

            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_xderef_size
        // OPERANDS: 1
        //  1 - uint8_t that specifies the size of the data to dereference.
        // DESCRIPTION: Behaves like the DW_OP_xderef operation: the entry at
        // the top of the stack is treated as an address. The second stack
        // entry is treated as an "address space identifier" for those
        // architectures that support multiple address spaces. The top two
        // stack elements are popped, a data item is retrieved through an
        // implementation-defined address calculation and pushed as the new
        // stack top. In the DW_OP_xderef_size operation, however, the size in
        // bytes of the data retrieved from the dereferenced address is
        // specified by the single operand. This operand is a 1-byte unsigned
        // integral constant whose value may not be larger than the size of an
        // address on the target machine. The data retrieved is zero extended
        // to the size of an address on the target machine before being pushed
        // on the expression stack.
        //----------------------------------------------------------------------
        case DW_OP_xderef_size:
            if (error_ptr)
                error_ptr->SetErrorString("Unimplemented opcode: DW_OP_xderef_size.");
            return false;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_xderef
        // OPERANDS: none
        // DESCRIPTION: Provides an extended dereference mechanism. The entry at
        // the top of the stack is treated as an address. The second stack entry
        // is treated as an "address space identifier" for those architectures
        // that support multiple address spaces. The top two stack elements are
        // popped, a data item is retrieved through an implementation-defined
        // address calculation and pushed as the new stack top. The size of the
        // data retrieved from the dereferenced address is the size of an address
        // on the target machine.
        //----------------------------------------------------------------------
        case DW_OP_xderef:
            if (error_ptr)
                error_ptr->SetErrorString("Unimplemented opcode: DW_OP_xderef.");
            return false;

        //----------------------------------------------------------------------
        // All DW_OP_constXXX opcodes have a single operand as noted below:
        //
        // Opcode           Operand 1
        // ---------------  ----------------------------------------------------
        // DW_OP_const1u    1-byte unsigned integer constant
        // DW_OP_const1s    1-byte signed integer constant
        // DW_OP_const2u    2-byte unsigned integer constant
        // DW_OP_const2s    2-byte signed integer constant
        // DW_OP_const4u    4-byte unsigned integer constant
        // DW_OP_const4s    4-byte signed integer constant
        // DW_OP_const8u    8-byte unsigned integer constant
        // DW_OP_const8s    8-byte signed integer constant
        // DW_OP_constu     unsigned LEB128 integer constant
        // DW_OP_consts     signed LEB128 integer constant
        //----------------------------------------------------------------------
        case DW_OP_const1u             :    stack.push_back(Scalar(( uint8_t)opcodes.GetU8 (&offset))); break;
        case DW_OP_const1s             :    stack.push_back(Scalar((  int8_t)opcodes.GetU8 (&offset))); break;
        case DW_OP_const2u             :    stack.push_back(Scalar((uint16_t)opcodes.GetU16 (&offset))); break;
        case DW_OP_const2s             :    stack.push_back(Scalar(( int16_t)opcodes.GetU16 (&offset))); break;
        case DW_OP_const4u             :    stack.push_back(Scalar((uint32_t)opcodes.GetU32 (&offset))); break;
        case DW_OP_const4s             :    stack.push_back(Scalar(( int32_t)opcodes.GetU32 (&offset))); break;
        case DW_OP_const8u             :    stack.push_back(Scalar((uint64_t)opcodes.GetU64 (&offset))); break;
        case DW_OP_const8s             :    stack.push_back(Scalar(( int64_t)opcodes.GetU64 (&offset))); break;
        case DW_OP_constu              :    stack.push_back(Scalar(opcodes.GetULEB128 (&offset))); break;
        case DW_OP_consts              :    stack.push_back(Scalar(opcodes.GetSLEB128 (&offset))); break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_dup
        // OPERANDS: none
        // DESCRIPTION: duplicates the value at the top of the stack
        //----------------------------------------------------------------------
        case DW_OP_dup:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack empty for DW_OP_dup.");
                return false;
            }
            else
                stack.push_back(stack.back());
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_drop
        // OPERANDS: none
        // DESCRIPTION: pops the value at the top of the stack
        //----------------------------------------------------------------------
        case DW_OP_drop:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack empty for DW_OP_drop.");
                return false;
            }
            else
                stack.pop_back();
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_over
        // OPERANDS: none
        // DESCRIPTION: Duplicates the entry currently second in the stack at
        // the top of the stack.
        //----------------------------------------------------------------------
        case DW_OP_over:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_over.");
                return false;
            }
            else
                stack.push_back(stack[stack.size() - 2]);
            break;


        //----------------------------------------------------------------------
        // OPCODE: DW_OP_pick
        // OPERANDS: uint8_t index into the current stack
        // DESCRIPTION: The stack entry with the specified index (0 through 255,
        // inclusive) is pushed on the stack
        //----------------------------------------------------------------------
        case DW_OP_pick:
            {
                uint8_t pick_idx = opcodes.GetU8(&offset);
                if (pick_idx < stack.size())
                    stack.push_back(stack[pick_idx]);
                else
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat("Index %u out of range for DW_OP_pick.\n", pick_idx);
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_swap
        // OPERANDS: none
        // DESCRIPTION: swaps the top two stack entries. The entry at the top
        // of the stack becomes the second stack entry, and the second entry
        // becomes the top of the stack
        //----------------------------------------------------------------------
        case DW_OP_swap:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_swap.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.back() = stack[stack.size() - 2];
                stack[stack.size() - 2] = tmp;
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_rot
        // OPERANDS: none
        // DESCRIPTION: Rotates the first three stack entries. The entry at
        // the top of the stack becomes the third stack entry, the second
        // entry becomes the top of the stack, and the third entry becomes
        // the second entry.
        //----------------------------------------------------------------------
        case DW_OP_rot:
            if (stack.size() < 3)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 3 items for DW_OP_rot.");
                return false;
            }
            else
            {
                size_t last_idx = stack.size() - 1;
                Value old_top = stack[last_idx];
                stack[last_idx] = stack[last_idx - 1];
                stack[last_idx - 1] = stack[last_idx - 2];
                stack[last_idx - 2] = old_top;
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_abs
        // OPERANDS: none
        // DESCRIPTION: pops the top stack entry, interprets it as a signed
        // value and pushes its absolute value. If the absolute value can not be
        // represented, the result is undefined.
        //----------------------------------------------------------------------
        case DW_OP_abs:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_abs.");
                return false;
            }
            else if (stack.back().ResolveValue(exe_ctx, ast_context).AbsoluteValue() == false)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Failed to take the absolute value of the first stack item.");
                return false;
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_and
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, performs a bitwise and
        // operation on the two, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_and:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_and.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) & tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_div
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, divides the former second
        // entry by the former top of the stack using signed division, and
        // pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_div:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_div.");
                return false;
            }
            else
            {
                tmp = stack.back();
                if (tmp.ResolveValue(exe_ctx, ast_context).IsZero())
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Divide by zero.");
                    return false;
                }
                else
                {
                    stack.pop_back();
                    stack.back() = stack.back().ResolveValue(exe_ctx, ast_context) / tmp.ResolveValue(exe_ctx, ast_context);
                    if (!stack.back().ResolveValue(exe_ctx, ast_context).IsValid())
                    {
                        if (error_ptr)
                            error_ptr->SetErrorString("Divide failed.");
                        return false;
                    }
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_minus
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, subtracts the former top
        // of the stack from the former second entry, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_minus:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_minus.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) - tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_mod
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values and pushes the result of
        // the calculation: former second stack entry modulo the former top of
        // the stack.
        //----------------------------------------------------------------------
        case DW_OP_mod:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_mod.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) % tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;


        //----------------------------------------------------------------------
        // OPCODE: DW_OP_mul
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, multiplies them
        // together, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_mul:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_mul.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) * tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_neg
        // OPERANDS: none
        // DESCRIPTION: pops the top stack entry, and pushes its negation.
        //----------------------------------------------------------------------
        case DW_OP_neg:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_neg.");
                return false;
            }
            else
            {
                if (stack.back().ResolveValue(exe_ctx, ast_context).UnaryNegate() == false)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Unary negate failed.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_not
        // OPERANDS: none
        // DESCRIPTION: pops the top stack entry, and pushes its bitwise
        // complement
        //----------------------------------------------------------------------
        case DW_OP_not:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_not.");
                return false;
            }
            else
            {
                if (stack.back().ResolveValue(exe_ctx, ast_context).OnesComplement() == false)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Logical NOT failed.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_or
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, performs a bitwise or
        // operation on the two, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_or:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_or.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) | tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_plus
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, adds them together, and
        // pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_plus:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_plus.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) + tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_plus_uconst
        // OPERANDS: none
        // DESCRIPTION: pops the top stack entry, adds it to the unsigned LEB128
        // constant operand and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_plus_uconst:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_plus_uconst.");
                return false;
            }
            else
            {
                const uint64_t uconst_value = opcodes.GetULEB128(&offset);
                // Implicit conversion from a UINT to a Scalar...
                stack.back().ResolveValue(exe_ctx, ast_context) += uconst_value;
                if (!stack.back().ResolveValue(exe_ctx, ast_context).IsValid())
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("DW_OP_plus_uconst failed.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_shl
        // OPERANDS: none
        // DESCRIPTION:  pops the top two stack entries, shifts the former
        // second entry left by the number of bits specified by the former top
        // of the stack, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_shl:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_shl.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) <<= tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_shr
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, shifts the former second
        // entry right logically (filling with zero bits) by the number of bits
        // specified by the former top of the stack, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_shr:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_shr.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                if (stack.back().ResolveValue(exe_ctx, ast_context).ShiftRightLogical(tmp.ResolveValue(exe_ctx, ast_context)) == false)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("DW_OP_shr failed.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_shra
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, shifts the former second
        // entry right arithmetically (divide the magnitude by 2, keep the same
        // sign for the result) by the number of bits specified by the former
        // top of the stack, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_shra:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_shra.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) >>= tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_xor
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack entries, performs the bitwise
        // exclusive-or operation on the two, and pushes the result.
        //----------------------------------------------------------------------
        case DW_OP_xor:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_xor.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) ^ tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;


        //----------------------------------------------------------------------
        // OPCODE: DW_OP_skip
        // OPERANDS: int16_t
        // DESCRIPTION:  An unconditional branch. Its single operand is a 2-byte
        // signed integer constant. The 2-byte constant is the number of bytes
        // of the DWARF expression to skip forward or backward from the current
        // operation, beginning after the 2-byte constant.
        //----------------------------------------------------------------------
        case DW_OP_skip:
            {
                int16_t skip_offset = (int16_t)opcodes.GetU16(&offset);
                lldb::offset_t new_offset = offset + skip_offset;
                if (new_offset >= opcodes_offset && new_offset < end_offset)
                    offset = new_offset;
                else
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Invalid opcode offset in DW_OP_skip.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_bra
        // OPERANDS: int16_t
        // DESCRIPTION: A conditional branch. Its single operand is a 2-byte
        // signed integer constant. This operation pops the top of stack. If
        // the value popped is not the constant 0, the 2-byte constant operand
        // is the number of bytes of the DWARF expression to skip forward or
        // backward from the current operation, beginning after the 2-byte
        // constant.
        //----------------------------------------------------------------------
        case DW_OP_bra:
            {
                tmp = stack.back();
                stack.pop_back();
                int16_t bra_offset = (int16_t)opcodes.GetU16(&offset);
                Scalar zero(0);
                if (tmp.ResolveValue(exe_ctx, ast_context) != zero)
                {
                    lldb::offset_t new_offset = offset + bra_offset;
                    if (new_offset >= opcodes_offset && new_offset < end_offset)
                        offset = new_offset;
                    else
                    {
                        if (error_ptr)
                            error_ptr->SetErrorString("Invalid opcode offset in DW_OP_bra.");
                        return false;
                    }
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_eq
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // equals (==) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_eq:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_eq.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) == tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_ge
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // greater than or equal to (>=) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_ge:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_ge.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) >= tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_gt
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // greater than (>) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_gt:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_gt.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) > tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_le
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // less than or equal to (<=) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_le:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_le.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) <= tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_lt
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // less than (<) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_lt:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_lt.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) < tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_ne
        // OPERANDS: none
        // DESCRIPTION: pops the top two stack values, compares using the
        // not equal (!=) operator.
        // STACK RESULT: push the constant value 1 onto the stack if the result
        // of the operation is true or the constant value 0 if the result of the
        // operation is false.
        //----------------------------------------------------------------------
        case DW_OP_ne:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_ne.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                stack.back().ResolveValue(exe_ctx, ast_context) = stack.back().ResolveValue(exe_ctx, ast_context) != tmp.ResolveValue(exe_ctx, ast_context);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_litn
        // OPERANDS: none
        // DESCRIPTION: encode the unsigned literal values from 0 through 31.
        // STACK RESULT: push the unsigned literal constant value onto the top
        // of the stack.
        //----------------------------------------------------------------------
        case DW_OP_lit0:
        case DW_OP_lit1:
        case DW_OP_lit2:
        case DW_OP_lit3:
        case DW_OP_lit4:
        case DW_OP_lit5:
        case DW_OP_lit6:
        case DW_OP_lit7:
        case DW_OP_lit8:
        case DW_OP_lit9:
        case DW_OP_lit10:
        case DW_OP_lit11:
        case DW_OP_lit12:
        case DW_OP_lit13:
        case DW_OP_lit14:
        case DW_OP_lit15:
        case DW_OP_lit16:
        case DW_OP_lit17:
        case DW_OP_lit18:
        case DW_OP_lit19:
        case DW_OP_lit20:
        case DW_OP_lit21:
        case DW_OP_lit22:
        case DW_OP_lit23:
        case DW_OP_lit24:
        case DW_OP_lit25:
        case DW_OP_lit26:
        case DW_OP_lit27:
        case DW_OP_lit28:
        case DW_OP_lit29:
        case DW_OP_lit30:
        case DW_OP_lit31:
            stack.push_back(Scalar(op - DW_OP_lit0));
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_regN
        // OPERANDS: none
        // DESCRIPTION: Push the value in register n on the top of the stack.
        //----------------------------------------------------------------------
        case DW_OP_reg0:
        case DW_OP_reg1:
        case DW_OP_reg2:
        case DW_OP_reg3:
        case DW_OP_reg4:
        case DW_OP_reg5:
        case DW_OP_reg6:
        case DW_OP_reg7:
        case DW_OP_reg8:
        case DW_OP_reg9:
        case DW_OP_reg10:
        case DW_OP_reg11:
        case DW_OP_reg12:
        case DW_OP_reg13:
        case DW_OP_reg14:
        case DW_OP_reg15:
        case DW_OP_reg16:
        case DW_OP_reg17:
        case DW_OP_reg18:
        case DW_OP_reg19:
        case DW_OP_reg20:
        case DW_OP_reg21:
        case DW_OP_reg22:
        case DW_OP_reg23:
        case DW_OP_reg24:
        case DW_OP_reg25:
        case DW_OP_reg26:
        case DW_OP_reg27:
        case DW_OP_reg28:
        case DW_OP_reg29:
        case DW_OP_reg30:
        case DW_OP_reg31:
            {
                reg_num = op - DW_OP_reg0;

                if (ReadRegisterValueAsScalar (reg_ctx, reg_kind, reg_num, error_ptr, tmp))
                    stack.push_back(tmp);
                else
                    return false;
            }
            break;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_regx
        // OPERANDS:
        //      ULEB128 literal operand that encodes the register.
        // DESCRIPTION: Push the value in register on the top of the stack.
        //----------------------------------------------------------------------
        case DW_OP_regx:
            {
                reg_num = opcodes.GetULEB128(&offset);
                if (ReadRegisterValueAsScalar (reg_ctx, reg_kind, reg_num, error_ptr, tmp))
                    stack.push_back(tmp);
                else
                    return false;
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_bregN
        // OPERANDS:
        //      SLEB128 offset from register N
        // DESCRIPTION: Value is in memory at the address specified by register
        // N plus an offset.
        //----------------------------------------------------------------------
        case DW_OP_breg0:
        case DW_OP_breg1:
        case DW_OP_breg2:
        case DW_OP_breg3:
        case DW_OP_breg4:
        case DW_OP_breg5:
        case DW_OP_breg6:
        case DW_OP_breg7:
        case DW_OP_breg8:
        case DW_OP_breg9:
        case DW_OP_breg10:
        case DW_OP_breg11:
        case DW_OP_breg12:
        case DW_OP_breg13:
        case DW_OP_breg14:
        case DW_OP_breg15:
        case DW_OP_breg16:
        case DW_OP_breg17:
        case DW_OP_breg18:
        case DW_OP_breg19:
        case DW_OP_breg20:
        case DW_OP_breg21:
        case DW_OP_breg22:
        case DW_OP_breg23:
        case DW_OP_breg24:
        case DW_OP_breg25:
        case DW_OP_breg26:
        case DW_OP_breg27:
        case DW_OP_breg28:
        case DW_OP_breg29:
        case DW_OP_breg30:
        case DW_OP_breg31:
            {
                reg_num = op - DW_OP_breg0;

                if (ReadRegisterValueAsScalar (reg_ctx, reg_kind, reg_num, error_ptr, tmp))
                {
                    int64_t breg_offset = opcodes.GetSLEB128(&offset);
                    tmp.ResolveValue(exe_ctx, ast_context) += (uint64_t)breg_offset;
                    tmp.ClearContext();
                    stack.push_back(tmp);
                    stack.back().SetValueType (Value::eValueTypeLoadAddress);
                }
                else
                    return false;
            }
            break;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_bregx
        // OPERANDS: 2
        //      ULEB128 literal operand that encodes the register.
        //      SLEB128 offset from register N
        // DESCRIPTION: Value is in memory at the address specified by register
        // N plus an offset.
        //----------------------------------------------------------------------
        case DW_OP_bregx:
            {
                reg_num = opcodes.GetULEB128(&offset);

                if (ReadRegisterValueAsScalar (reg_ctx, reg_kind, reg_num, error_ptr, tmp))
                {
                    int64_t breg_offset = opcodes.GetSLEB128(&offset);
                    tmp.ResolveValue(exe_ctx, ast_context) += (uint64_t)breg_offset;
                    tmp.ClearContext();
                    stack.push_back(tmp);
                    stack.back().SetValueType (Value::eValueTypeLoadAddress);
                }
                else
                    return false;
            }
            break;

        case DW_OP_fbreg:
            if (exe_ctx)
            {
                if (frame)
                {
                    Scalar value;
                    if (frame->GetFrameBaseValue(value, error_ptr))
                    {
                        int64_t fbreg_offset = opcodes.GetSLEB128(&offset);
                        value += fbreg_offset;
                        stack.push_back(value);
                        stack.back().SetValueType (Value::eValueTypeLoadAddress);
                    }
                    else
                        return false;
                }
                else
                {
                    if (error_ptr)
                        error_ptr->SetErrorString ("Invalid stack frame in context for DW_OP_fbreg opcode.");
                    return false;
                }
            }
            else
            {
                if (error_ptr)
                    error_ptr->SetErrorStringWithFormat ("NULL execution context for DW_OP_fbreg.\n");
                return false;
            }

            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_nop
        // OPERANDS: none
        // DESCRIPTION: A place holder. It has no effect on the location stack
        // or any of its values.
        //----------------------------------------------------------------------
        case DW_OP_nop:
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_piece
        // OPERANDS: 1
        //      ULEB128: byte size of the piece
        // DESCRIPTION: The operand describes the size in bytes of the piece of
        // the object referenced by the DWARF expression whose result is at the
        // top of the stack. If the piece is located in a register, but does not
        // occupy the entire register, the placement of the piece within that
        // register is defined by the ABI.
        //
        // Many compilers store a single variable in sets of registers, or store
        // a variable partially in memory and partially in registers.
        // DW_OP_piece provides a way of describing how large a part of a
        // variable a particular DWARF expression refers to.
        //----------------------------------------------------------------------
        case DW_OP_piece:
            if (error_ptr)
                error_ptr->SetErrorString ("Unimplemented opcode DW_OP_piece.");
            return false;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_push_object_address
        // OPERANDS: none
        // DESCRIPTION: Pushes the address of the object currently being
        // evaluated as part of evaluation of a user presented expression.
        // This object may correspond to an independent variable described by
        // its own DIE or it may be a component of an array, structure, or class
        // whose address has been dynamically determined by an earlier step
        // during user expression evaluation.
        //----------------------------------------------------------------------
        case DW_OP_push_object_address:
            if (error_ptr)
                error_ptr->SetErrorString ("Unimplemented opcode DW_OP_push_object_address.");
            return false;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_call2
        // OPERANDS:
        //      uint16_t compile unit relative offset of a DIE
        // DESCRIPTION: Performs subroutine calls during evaluation
        // of a DWARF expression. The operand is the 2-byte unsigned offset
        // of a debugging information entry in the current compilation unit.
        //
        // Operand interpretation is exactly like that for DW_FORM_ref2.
        //
        // This operation transfers control of DWARF expression evaluation
        // to the DW_AT_location attribute of the referenced DIE. If there is
        // no such attribute, then there is no effect. Execution of the DWARF
        // expression of a DW_AT_location attribute may add to and/or remove from
        // values on the stack. Execution returns to the point following the call
        // when the end of the attribute is reached. Values on the stack at the
        // time of the call may be used as parameters by the called expression
        // and values left on the stack by the called expression may be used as
        // return values by prior agreement between the calling and called
        // expressions.
        //----------------------------------------------------------------------
        case DW_OP_call2:
            if (error_ptr)
                error_ptr->SetErrorString ("Unimplemented opcode DW_OP_call2.");
            return false;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_call4
        // OPERANDS: 1
        //      uint32_t compile unit relative offset of a DIE
        // DESCRIPTION: Performs a subroutine call during evaluation of a DWARF
        // expression. For DW_OP_call4, the operand is a 4-byte unsigned offset
        // of a debugging information entry in  the current compilation unit.
        //
        // Operand interpretation DW_OP_call4 is exactly like that for
        // DW_FORM_ref4.
        //
        // This operation transfers control of DWARF expression evaluation
        // to the DW_AT_location attribute of the referenced DIE. If there is
        // no such attribute, then there is no effect. Execution of the DWARF
        // expression of a DW_AT_location attribute may add to and/or remove from
        // values on the stack. Execution returns to the point following the call
        // when the end of the attribute is reached. Values on the stack at the
        // time of the call may be used as parameters by the called expression
        // and values left on the stack by the called expression may be used as
        // return values by prior agreement between the calling and called
        // expressions.
        //----------------------------------------------------------------------
        case DW_OP_call4:
            if (error_ptr)
                error_ptr->SetErrorString ("Unimplemented opcode DW_OP_call4.");
            return false;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_stack_value
        // OPERANDS: None
        // DESCRIPTION: Specifies that the object does not exist in memory but
        // rather is a constant value.  The value from the top of the stack is
        // the value to be used.  This is the actual object value and not the
        // location.
        //----------------------------------------------------------------------
        case DW_OP_stack_value:
            stack.back().SetValueType(Value::eValueTypeScalar);
            break;

#if 0
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_call_ref
        // OPERANDS:
        //      uint32_t absolute DIE offset for 32-bit DWARF or a uint64_t
        //               absolute DIE offset for 64 bit DWARF.
        // DESCRIPTION: Performs a subroutine call during evaluation of a DWARF
        // expression. Takes a single operand. In the 32-bit DWARF format, the
        // operand is a 4-byte unsigned value; in the 64-bit DWARF format, it
        // is an 8-byte unsigned value. The operand is used as the offset of a
        // debugging information entry in a .debug_info section which may be
        // contained in a shared object for executable other than that
        // containing the operator. For references from one shared object or
        // executable to another, the relocation must be performed by the
        // consumer.
        //
        // Operand interpretation of DW_OP_call_ref is exactly like that for
        // DW_FORM_ref_addr.
        //
        // This operation transfers control of DWARF expression evaluation
        // to the DW_AT_location attribute of the referenced DIE. If there is
        // no such attribute, then there is no effect. Execution of the DWARF
        // expression of a DW_AT_location attribute may add to and/or remove from
        // values on the stack. Execution returns to the point following the call
        // when the end of the attribute is reached. Values on the stack at the
        // time of the call may be used as parameters by the called expression
        // and values left on the stack by the called expression may be used as
        // return values by prior agreement between the calling and called
        // expressions.
        //----------------------------------------------------------------------
        case DW_OP_call_ref:
            if (error_ptr)
                error_ptr->SetErrorString ("Unimplemented opcode DW_OP_call_ref.");
            return false;

                //----------------------------------------------------------------------
                // OPCODE: DW_OP_APPLE_array_ref
                // OPERANDS: none
                // DESCRIPTION: Pops a value off the stack and uses it as the array 
                // index.  Pops a second value off the stack and uses it as the array
                // itself.  Pushes a value onto the stack representing the element of
                // the array specified by the index.
                //----------------------------------------------------------------------
            case DW_OP_APPLE_array_ref:
            {
                if (stack.size() < 2)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_APPLE_array_ref.");
                    return false;
                }
                
                Value index_val = stack.back();
                stack.pop_back();
                Value array_val = stack.back();
                stack.pop_back();
                
                Scalar &index_scalar = index_val.ResolveValue(exe_ctx, ast_context);
                int64_t index = index_scalar.SLongLong(LLONG_MAX);
                
                if (index == LLONG_MAX)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Invalid array index.");
                    return false;
                }
                
                if (array_val.GetContextType() != Value::eContextTypeClangType)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Arrays without Clang types are unhandled at this time.");
                    return false;
                }
                
                if (array_val.GetValueType() != Value::eValueTypeLoadAddress &&
                    array_val.GetValueType() != Value::eValueTypeHostAddress)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Array must be stored in memory.");
                    return false;
                }
                
                void *array_type = array_val.GetClangType();
                
                void *member_type;
                uint64_t size = 0;
                
                if ((!ClangASTContext::IsPointerType(array_type, &member_type)) &&
                    (!ClangASTContext::IsArrayType(array_type, &member_type, &size)))
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Array reference from something that is neither a pointer nor an array.");
                    return false;
                }
                
                if (size && (index >= size || index < 0))
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat("Out of bounds array access.  %" PRId64 " is not in [0, %" PRIu64 "]", index, size);
                    return false;
                }
                
                uint64_t member_bit_size = ClangASTType::GetClangTypeBitWidth(ast_context, member_type);
                uint64_t member_bit_align = ClangASTType::GetTypeBitAlign(ast_context, member_type);
                uint64_t member_bit_incr = ((member_bit_size + member_bit_align - 1) / member_bit_align) * member_bit_align;
                if (member_bit_incr % 8)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat("Array increment is not byte aligned");
                    return false;
                }
                int64_t member_offset = (int64_t)(member_bit_incr / 8) * index;
                
                Value member;
                
                member.SetContext(Value::eContextTypeClangType, member_type);
                member.SetValueType(array_val.GetValueType());
                
                addr_t array_base = (addr_t)array_val.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                addr_t member_loc = array_base + member_offset;
                member.GetScalar() = (uint64_t)member_loc;
                
                stack.push_back(member);
            }
                break;
                
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_uninit
        // OPERANDS: none
        // DESCRIPTION: Lets us know that the value is currently not initialized
        //----------------------------------------------------------------------
        case DW_OP_APPLE_uninit:
            //return eResultTypeErrorUninitialized;
            break;  // Ignore this as we have seen cases where this value is incorrectly added

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_assign
        // OPERANDS: none
        // DESCRIPTION: Pops a value off of the stack and assigns it to the next
        // item on the stack which must be something assignable (inferior
        // Variable, inferior Type with address, inferior register, or
        // expression local variable.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_assign:
            if (stack.size() < 2)
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 2 items for DW_OP_APPLE_assign.");
                return false;
            }
            else
            {
                tmp = stack.back();
                stack.pop_back();
                Value::ContextType context_type = stack.back().GetContextType();
                StreamString new_value(Stream::eBinary, 4, lldb::endian::InlHostByteOrder());
                switch (context_type)
                {
                case Value::eContextTypeClangType:
                    {
                        void *clang_type = stack.back().GetClangType();
                        
                        if (ClangASTContext::IsAggregateType (clang_type))
                        {
                            Value::ValueType source_value_type = tmp.GetValueType();
                            Value::ValueType target_value_type = stack.back().GetValueType();
                            
                            addr_t source_addr = (addr_t)tmp.GetScalar().ULongLong();
                            addr_t target_addr = (addr_t)stack.back().GetScalar().ULongLong();
                            
                            const uint64_t byte_size = ClangASTType::GetTypeByteSize(ast_context, clang_type);
                            
                            switch (source_value_type)
                            {
                            case Value::eValueTypeScalar:
                            case Value::eValueTypeFileAddress:
                                break;

                            case Value::eValueTypeLoadAddress:
                                switch (target_value_type)
                                {
                                case Value::eValueTypeLoadAddress:
                                    {
                                        DataBufferHeap data;
                                        data.SetByteSize(byte_size);
                                        
                                        Error error;
                                        if (process->ReadMemory (source_addr, data.GetBytes(), byte_size, error) != byte_size)
                                        {
                                            if (error_ptr)
                                                error_ptr->SetErrorStringWithFormat ("Couldn't read a composite type from the target: %s", error.AsCString());
                                            return false;
                                        }
                                        
                                        if (process->WriteMemory (target_addr, data.GetBytes(), byte_size, error) != byte_size)
                                        {
                                            if (error_ptr)
                                                error_ptr->SetErrorStringWithFormat ("Couldn't write a composite type to the target: %s", error.AsCString());
                                            return false;
                                        }
                                    }
                                    break;
                                case Value::eValueTypeHostAddress:
                                    if (process->GetByteOrder() != lldb::endian::InlHostByteOrder())
                                    {
                                        if (error_ptr)
                                            error_ptr->SetErrorStringWithFormat ("Copy of composite types between incompatible byte orders is unimplemented");
                                        return false;
                                    }
                                    else
                                    {
                                        Error error;
                                        if (process->ReadMemory (source_addr, (uint8_t*)target_addr, byte_size, error) != byte_size)
                                        {
                                            if (error_ptr)
                                                error_ptr->SetErrorStringWithFormat ("Couldn't read a composite type from the target: %s", error.AsCString());
                                            return false;
                                        }
                                    }
                                    break;
                                default:
                                    return false;
                                }
                                break;
                            case Value::eValueTypeHostAddress:
                                switch (target_value_type)
                                {
                                case Value::eValueTypeLoadAddress:
                                    if (process->GetByteOrder() != lldb::endian::InlHostByteOrder())
                                    {
                                        if (error_ptr)
                                            error_ptr->SetErrorStringWithFormat ("Copy of composite types between incompatible byte orders is unimplemented");
                                        return false;
                                    }
                                    else
                                    {
                                        Error error;
                                        if (process->WriteMemory (target_addr, (uint8_t*)source_addr, byte_size, error) != byte_size)
                                        {
                                            if (error_ptr)
                                                error_ptr->SetErrorStringWithFormat ("Couldn't write a composite type to the target: %s", error.AsCString());
                                            return false;
                                        }
                                    }
                                case Value::eValueTypeHostAddress:
                                    memcpy ((uint8_t*)target_addr, (uint8_t*)source_addr, byte_size);
                                    break;
                                default:
                                    return false;
                                }
                            }
                        }
                        else
                        {
                            if (!ClangASTType::SetValueFromScalar (ast_context,
                                                                  clang_type,
                                                                  tmp.ResolveValue(exe_ctx, ast_context),
                                                                  new_value))
                            {
                                if (error_ptr)
                                    error_ptr->SetErrorStringWithFormat ("Couldn't extract a value from an integral type.\n");
                                return false;
                            }
                                
                            Value::ValueType value_type = stack.back().GetValueType();
                            
                            switch (value_type)
                            {
                            case Value::eValueTypeLoadAddress:
                            case Value::eValueTypeHostAddress:
                                {
                                    AddressType address_type = (value_type == Value::eValueTypeLoadAddress ? eAddressTypeLoad : eAddressTypeHost);
                                    lldb::addr_t addr = stack.back().GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
                                    if (!ClangASTType::WriteToMemory (ast_context,
                                                                          clang_type,
                                                                          exe_ctx,
                                                                          addr,
                                                                          address_type,
                                                                          new_value))
                                    {
                                        if (error_ptr)
                                            error_ptr->SetErrorStringWithFormat ("Failed to write value to memory at 0x%" PRIx64 ".\n", addr);
                                        return false;
                                    }
                                }
                                break;

                            default:
                                break;
                            }
                        }
                    }
                    break;

                default:
                    if (error_ptr)
                        error_ptr->SetErrorString ("Assign failed.");
                    return false;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_address_of
        // OPERANDS: none
        // DESCRIPTION: Pops a value off of the stack and pushed its address.
        // The top item on the stack must be a variable, or already be a memory
        // location.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_address_of:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_APPLE_address_of.");
                return false;
            }
            else
            {
                Value::ValueType value_type = stack.back().GetValueType();
                switch (value_type)
                {
                default:
                case Value::eValueTypeScalar:      // raw scalar value
                    if (error_ptr)
                        error_ptr->SetErrorString("Top stack item isn't a memory based object.");
                    return false;

                case Value::eValueTypeLoadAddress: // load address value
                case Value::eValueTypeFileAddress: // file address value
                case Value::eValueTypeHostAddress: // host address value (for memory in the process that is using liblldb)
                    // Taking the address of an object reduces it to the address
                    // of the value and removes any extra context it had.
                    //stack.back().SetValueType(Value::eValueTypeScalar);
                    stack.back().ClearContext();
                    break;
                }
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_value_of
        // OPERANDS: none
        // DESCRIPTION: Pops a value off of the stack and pushed its value.
        // The top item on the stack must be a variable, expression variable.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_value_of:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 items for DW_OP_APPLE_value_of.");
                return false;
            }
            else if (!stack.back().ValueOf(exe_ctx, ast_context))
            {
                if (error_ptr)
                    error_ptr->SetErrorString ("Top stack item isn't a valid candidate for DW_OP_APPLE_value_of.");
                return false;
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_deref_type
        // OPERANDS: none
        // DESCRIPTION: gets the value pointed to by the top stack item
        //----------------------------------------------------------------------
        case DW_OP_APPLE_deref_type:
            {
                if (stack.empty())
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Expression stack needs at least 1 items for DW_OP_APPLE_deref_type.");
                    return false;
                }
                    
                tmp = stack.back();
                stack.pop_back();
                
                if (tmp.GetContextType() != Value::eContextTypeClangType)
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Item at top of expression stack must have a Clang type");
                    return false;
                }
                    
                void *ptr_type = tmp.GetClangType();
                void *target_type;
            
                if (!ClangASTContext::IsPointerType(ptr_type, &target_type))
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Dereferencing a non-pointer type");
                    return false;
                }
                
                // TODO do we want all pointers to be dereferenced as load addresses?
                Value::ValueType value_type = tmp.GetValueType();
                
                tmp.ResolveValue(exe_ctx, ast_context);
                
                tmp.SetValueType(value_type);
                tmp.SetContext(Value::eContextTypeClangType, target_type);
                
                stack.push_back(tmp);
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_expr_local
        // OPERANDS: ULEB128
        // DESCRIPTION: pushes the expression local variable index onto the
        // stack and set the appropriate context so we know the stack item is
        // an expression local variable index.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_expr_local:
            {
                /*
                uint32_t idx = opcodes.GetULEB128(&offset);
                if (expr_locals == NULL)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("DW_OP_APPLE_expr_local(%u) opcode encountered with no local variable list.\n", idx);
                    return false;
                }
                Value *expr_local_variable = expr_locals->GetVariableAtIndex(idx);
                if (expr_local_variable == NULL)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("DW_OP_APPLE_expr_local(%u) with invalid index %u.\n", idx, idx);
                    return false;
                }
                 // The proxy code has been removed. If it is ever re-added, please
                 // use shared pointers or return by value to avoid possible memory
                 // leak (there is no leak here, but in general, no returning pointers
                 // that must be manually freed please.
                Value *proxy = expr_local_variable->CreateProxy();
                stack.push_back(*proxy);
                delete proxy;
                //stack.back().SetContext (Value::eContextTypeClangType, expr_local_variable->GetClangType());
                */
            }
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_extern
        // OPERANDS: ULEB128
        // DESCRIPTION: pushes a proxy for the extern object index onto the
        // stack.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_extern:
            {
                /*
                uint32_t idx = opcodes.GetULEB128(&offset);
                if (!decl_map)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("DW_OP_APPLE_extern(%u) opcode encountered with no decl map.\n", idx);
                    return false;
                }
                Value *extern_var = decl_map->GetValueForIndex(idx);
                if (!extern_var)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("DW_OP_APPLE_extern(%u) with invalid index %u.\n", idx, idx);
                    return false;
                }
                // The proxy code has been removed. If it is ever re-added, please
                // use shared pointers or return by value to avoid possible memory
                // leak (there is no leak here, but in general, no returning pointers
                // that must be manually freed please.
                Value *proxy = extern_var->CreateProxy();
                stack.push_back(*proxy);
                delete proxy;
                */
            }
            break;

        case DW_OP_APPLE_scalar_cast:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_APPLE_scalar_cast.");
                return false;
            }
            else
            {
                // Simple scalar cast
                if (!stack.back().ResolveValue(exe_ctx, ast_context).Cast((Scalar::Type)opcodes.GetU8(&offset)))
                {
                    if (error_ptr)
                        error_ptr->SetErrorString("Cast failed.");
                    return false;
                }
            }
            break;


        case DW_OP_APPLE_clang_cast:
            if (stack.empty())
            {
                if (error_ptr)
                    error_ptr->SetErrorString("Expression stack needs at least 1 item for DW_OP_APPLE_clang_cast.");
                return false;
            }
            else
            {
                void *clang_type = (void *)opcodes.GetMaxU64(&offset, sizeof(void*));
                stack.back().SetContext (Value::eContextTypeClangType, clang_type);
            }
            break;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_constf
        // OPERANDS: 1 byte float length, followed by that many bytes containing
        // the constant float data.
        // DESCRIPTION: Push a float value onto the expression stack.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_constf:        // 0xF6 - 1 byte float size, followed by constant float data
            {
                uint8_t float_length = opcodes.GetU8(&offset);
                if (sizeof(float) == float_length)
                    tmp.ResolveValue(exe_ctx, ast_context) = opcodes.GetFloat (&offset);
                else if (sizeof(double) == float_length)
                    tmp.ResolveValue(exe_ctx, ast_context) = opcodes.GetDouble (&offset);
                else if (sizeof(long double) == float_length)
                    tmp.ResolveValue(exe_ctx, ast_context) = opcodes.GetLongDouble (&offset);
                else
                {
                    StreamString new_value;
                    opcodes.Dump(&new_value, offset, eFormatBytes, 1, float_length, UINT32_MAX, DW_INVALID_ADDRESS, 0, 0);

                     if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("DW_OP_APPLE_constf(<%u> %s) unsupported float size.\n", float_length, new_value.GetData());
                    return false;
               }
               tmp.SetValueType(Value::eValueTypeScalar);
               tmp.ClearContext();
               stack.push_back(tmp);
            }
            break;
        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_clear
        // OPERANDS: none
        // DESCRIPTION: Clears the expression stack.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_clear:
            stack.clear();
            break;

        //----------------------------------------------------------------------
        // OPCODE: DW_OP_APPLE_error
        // OPERANDS: none
        // DESCRIPTION: Pops a value off of the stack and pushed its value.
        // The top item on the stack must be a variable, expression variable.
        //----------------------------------------------------------------------
        case DW_OP_APPLE_error:         // 0xFF - Stops expression evaluation and returns an error (no args)
            if (error_ptr)
                error_ptr->SetErrorString ("Generic error.");
            return false;
#endif // #if 0
                
        }
    }

    if (stack.empty())
    {
        if (error_ptr)
            error_ptr->SetErrorString ("Stack empty after evaluation.");
        return false;
    }
    else if (log && log->GetVerbose())
    {
        size_t count = stack.size();
        log->Printf("Stack after operation has %lu values:", count);
        for (size_t i=0; i<count; ++i)
        {
            StreamString new_value;
            new_value.Printf("[%" PRIu64 "]", (uint64_t)i);
            stack[i].Dump(&new_value);
            log->Printf("  %s", new_value.GetData());
        }
    }

    result = stack.back();
    return true;    // Return true on success
}

