//===-- DWARFLocationDescription.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFLocationDescription.h"
#include "DWARFDefines.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/Stream.h"


using namespace lldb_private;

static int print_dwarf_exp_op (Stream &s, const DataExtractor& data, uint32_t* offset_ptr, int address_size, int dwarf_ref_size);

int
print_dwarf_expression (Stream &s,
                        const DataExtractor& data,
                        int address_size,
                        int dwarf_ref_size,
                        bool location_expression)
{
    int op_count = 0;
    uint32_t offset = 0;
    while (data.ValidOffset(offset))
    {
        if (location_expression && op_count > 0)
        {
            //  err (baton, "Dwarf location expressions may only have one operand!");
            return 1;
        }
        if (op_count > 0)
        {
            s.PutCString(", ");
        }
        if (print_dwarf_exp_op (s, data, &offset, address_size, dwarf_ref_size) == 1)
            return 1;
        op_count++;
    }

    return 0;
}

static int
print_dwarf_exp_op (Stream &s,
                    const DataExtractor& data,
                    uint32_t* offset_ptr,
                    int address_size,
                    int dwarf_ref_size)
{
    uint8_t opcode = data.GetU8(offset_ptr);
    DRC_class opcode_class;
    uint64_t  uint;
    int64_t   sint;

    int size;

    opcode_class = DW_OP_value_to_class (opcode) & (~DRC_DWARFv3);

    s.Printf("%s ", DW_OP_value_to_name (opcode));

    /* Does this take zero parameters?  If so we can shortcut this function.  */
    if (opcode_class == DRC_ZEROOPERANDS)
        return 0;

    if (opcode_class == DRC_TWOOPERANDS && opcode == DW_OP_bregx)
    {
        uint = data.GetULEB128(offset_ptr);
        sint = data.GetSLEB128(offset_ptr);
        s.Printf("%llu %lli", uint, sint);
        return 0;
    }
    if (opcode_class != DRC_ONEOPERAND)
    {
        s.Printf("UNKNOWN OP %u", opcode);
        return 1;
    }

    switch (opcode)
    {
        case DW_OP_addr:    size = address_size;    break;
        case DW_OP_const1u: size = 1;               break;
        case DW_OP_const1s: size = -1;              break;
        case DW_OP_const2u: size = 2;               break;
        case DW_OP_const2s: size = -2;              break;
        case DW_OP_const4u: size = 4;               break;
        case DW_OP_const4s: size = -4;              break;
        case DW_OP_const8u: size = 8;               break;
        case DW_OP_const8s: size = -8;              break;
        case DW_OP_constu:  size = 128;             break;
        case DW_OP_consts:  size = -128;            break;
        case DW_OP_fbreg:   size = -128;            break;
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
            size = -128; break;
        case DW_OP_pick:
            size = 1;       break;
        case DW_OP_deref_size:
            size = 1;       break;
        case DW_OP_xderef_size:
            size = 1;       break;
        case DW_OP_plus_uconst:
            size = 128;     break;
        case DW_OP_skip:
            size = -2;      break;
        case DW_OP_bra:
            size = -2;      break;
        case DW_OP_call2:
            size = 2;       break;
        case DW_OP_call4:
            size = 4;       break;
        case DW_OP_call_ref:
            size = dwarf_ref_size;  break;
        case DW_OP_piece:
            size = 128; break;
        case DW_OP_regx:
            size = 128; break;
        default:
            s.Printf("UNKNOWN ONE-OPERAND OPCODE, #%u", opcode);
            return 1;
    }

    switch (size)
    {
    case -1:    sint = (int8_t)     data.GetU8(offset_ptr);     s.Printf("%+lli", sint); break;
    case -2:    sint = (int16_t)    data.GetU16(offset_ptr);    s.Printf("%+lli", sint); break;
    case -4:    sint = (int32_t)    data.GetU32(offset_ptr);    s.Printf("%+lli", sint); break;
    case -8:    sint = (int64_t)    data.GetU64(offset_ptr);    s.Printf("%+lli", sint); break;
    case -128:  sint = data.GetSLEB128(offset_ptr);             s.Printf("%+lli", sint); break;
    case 1:     uint = data.GetU8(offset_ptr);                  s.Printf("0x%2.2llx", uint); break;
    case 2:     uint = data.GetU16(offset_ptr);                 s.Printf("0x%4.4llx", uint); break;
    case 4:     uint = data.GetU32(offset_ptr);                 s.Printf("0x%8.8llx", uint); break;
    case 8:     uint = data.GetU64(offset_ptr);                 s.Printf("0x%16.16llx", uint); break;
    case 128:   uint = data.GetULEB128(offset_ptr);             s.Printf("0x%llx", uint); break;
    }

    return 0;
}
