//===-- NVPTXutil.cpp - Functions exported to CodeGen --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the functions that can be used in CodeGen.
//
//===----------------------------------------------------------------------===//

#include "NVPTXutil.h"
#include "NVPTX.h"

using namespace llvm;

namespace llvm {

bool isParamLoad(const MachineInstr *MI)
{
  if ((MI->getOpcode() != NVPTX::LD_i32_avar) &&
      (MI->getOpcode() != NVPTX::LD_i64_avar))
    return false;
  if (MI->getOperand(2).isImm() == false)
    return false;
  if (MI->getOperand(2).getImm() != NVPTX::PTXLdStInstCode::PARAM)
    return false;
  return true;
}

#define DATA_MASK     0x7f
#define DIGIT_WIDTH   7
#define MORE_BYTES    0x80

static int encode_leb128(uint64_t val, int *nbytes,
                         char *space, int splen)
{
  char *a;
  char *end = space + splen;

  a = space;
  do {
    unsigned char uc;

    if (a >= end)
      return 1;
    uc = val & DATA_MASK;
    val >>= DIGIT_WIDTH;
    if (val != 0)
      uc |= MORE_BYTES;
    *a = uc;
    a++;
  } while (val);
  *nbytes = a - space;
  return 0;
}

#undef DATA_MASK
#undef DIGIT_WIDTH
#undef MORE_BYTES

uint64_t encode_leb128(const char *str)
{
  union { uint64_t x; char a[8]; } temp64;

  temp64.x = 0;

  for (unsigned i=0,e=strlen(str); i!=e; ++i)
    temp64.a[i] = str[e-1-i];

  char encoded[16];
  int nbytes;

  int retval = encode_leb128(temp64.x, &nbytes, encoded, 16);

  (void)retval;
  assert(retval == 0 &&
         "Encoding to leb128 failed");

  assert(nbytes <= 8 &&
         "Cannot support register names with leb128 encoding > 8 bytes");

  temp64.x = 0;
  for (int i=0; i<nbytes; ++i)
    temp64.a[i] = encoded[i];

  return temp64.x;
}

} // end namespace llvm
