#include "ABICall.h"
#include "ABIFunctionInfo.h"
#include "Type.h"
#include "llvm/Support/Casting.h"

using namespace ABI;
using namespace ABIFunction;

/// X86_64ABIInfo - The X86_64 ABI information.
class X86_64ABIInfo : public ABICall {
public:
  enum Class {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

public:
  ABIArgInfo classifyReturnType(Type RetTy) const;
  ABIArgInfo classifyArgumentType(Type RetTy) const;
  void classify(Type Ty, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg, bool isRegCall) const;

  void computeInfo(ABIFunctionInfo &FI) const override;
};

void X86_64ABIInfo::classify(Type Ty, uint64_t OffsetBase, Class &Lo, Class &Hi,
                             bool isNamedArg, bool IsRegCall = true) const {
  Lo = Hi = NoClass;

  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Memory;

  if (const ABIBuiltinType *BT = llvm::dyn_cast<ABIBuiltinType>(&Ty)) {
    ABIBuiltinType::Kind k = BT->getKind();

    if (k == ABIBuiltinType::Void) {
      Current = NoClass;
    } else if (k == ABIBuiltinType::Int128 || k == ABIBuiltinType::UInt128) {
      Lo = Integer;
      Hi = Integer;
    } else if (k >= ABIBuiltinType::Bool && k <= ABIBuiltinType::LongLong) {
      Current = Integer;
    } else if (k == ABIBuiltinType::Float || k == ABIBuiltinType::Double ||
               k == ABIBuiltinType::Float16 || k == ABIBuiltinType::BFloat16) {
      Current = SSE;
    } else if (k == ABIBuiltinType::Float128) {
      Lo = SSE;
      Hi = SSEUp;
    } else if (k == ABIBuiltinType::LongDouble) {
      Lo = X87;
      Hi = X87Up;
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }
}

ABIArgInfo X86_64ABIInfo::classifyReturnType(Type Ty) const {
  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, /*isNamedArg*/ true);
  assert((Hi != Memory || Lo == Memory) && "Invalid memory classification.");
  assert((Hi != SSEUp || Lo == SSE) && "Invalid SSEUp classification.");

  switch (Lo) {
  case NoClass:
    if (Hi == NoClass)
      return ABIArgInfo::getIgnore();
    // If the low part is just padding, it takes no register, leave ResType
    // null.
    assert((Hi == SSE || Hi == Integer || Hi == X87Up) &&
           "Unknown missing lo part");
    break;

  case SSEUp:
  case X87Up:
    assert(true && "Invalid classification for lo word.");

    // AMD64-ABI 3.2.3p4: Rule 2. Types of class memory are returned via
    // hidden argument.
  case Memory:
    return ABIArgInfo::getIndirect();

    // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next
    // available register of the sequence %rax, %rdx is used.
  case Integer:
    return ABIArgInfo::getExtend();

    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case SSE:
    break;

    // AMD64-ABI 3.2.3p4: Rule 6. If the class is X87, the value is
    // returned on the X87 stack in %st0 as 80-bit x87 number.
  case X87:
    ResType = llvm::Type::getX86_FP80Ty(getVMContext());
    break;

    // AMD64-ABI 3.2.3p4: Rule 8. If the class is COMPLEX_X87, the real
    // part of the value is returned in %st0 and the imaginary part in
    // %st1.
  case ComplexX87:
    assert(Hi == ComplexX87 && "Unexpected ComplexX87 classification.");
    ResType = llvm::StructType::get(llvm::Type::getX86_FP80Ty(getVMContext()),
                                    llvm::Type::getX86_FP80Ty(getVMContext()));
    break;
  }

  Type *HighPart = nullptr;
  switch (Hi) {
    // Memory was handled previously and X87 should
    // never occur as a hi class.
  case Memory:
  case X87:
    assert(true && "Invalid classification for hi word.");

  case ComplexX87: // Previously handled.
  case NoClass:
    break;

  case Integer:
    HighPart = GetINTEGERTypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
    if (Lo == NoClass) // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;
  case SSE:
    HighPart = GetSSETypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
    if (Lo == NoClass) // Return HighPart at offset 8 in memory.
      return ABIArgInfo::getDirect(HighPart, 8);
    break;

    // AMD64-ABI 3.2.3p4: Rule 5. If the class is SSEUP, the eightbyte
    // is passed in the next available eightbyte chunk if the last used
    // vector register.
    //
    // SSEUP should always be preceded by SSE, just widen.
  case SSEUp:
    assert(Lo == SSE && "Unexpected SSEUp classification.");
    ResType = GetByteVectorType(RetTy);
    break;

    // AMD64-ABI 3.2.3p4: Rule 7. If the class is X87UP, the value is
    // returned together with the previous X87 value in %st0.
  case X87Up:
    // If X87Up is preceded by X87, we don't need to do
    // anything. However, in some cases with unions it may not be
    // preceded by X87. In such situations we follow gcc and pass the
    // extra bits in an SSE reg.
    if (Lo != X87) {
      HighPart = GetSSETypeAtOffset(CGT.ConvertType(RetTy), 8, RetTy, 8);
      if (Lo == NoClass) // Return HighPart at offset 8 in memory.
        return ABIArgInfo::getDirect(HighPart, 8);
    }
    break;
  }
}

ABIArgInfo X86_64ABIInfo::classifyArgumentType(Type Ty) const {
  // TODO
}

void X86_64ABIInfo::computeInfo(ABIFunctionInfo &FI) const {

  const unsigned CallingConv = FI.getCallingConvention();

  if (CallingConv == CallingConv::Win64) {
    // stub implementation for windows CC.
    // TODO
    return;
  }

  bool IsRegCall = CallingConv == CallingConv::X86_RegCall;

  // Keep track of the number of assigned registers.
  unsigned FreeIntRegs = IsRegCall ? 11 : 6;
  unsigned FreeSSERegs = IsRegCall ? 16 : 8;
  unsigned NeededInt = 0, NeededSSE = 0, MaxVectorWidth = 0;

  // adjust the return type according to the ABI spec
  FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  for (ABIFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it) {
    it->info = classifyArgumentType(it->type);
  }
}

// TODO: still need to figure out how to pass the Target info to the library.
std::unique_ptr<TargetCodeGenInfo>
CodeGen::createBPFTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<BPFTargetCodeGenInfo>(CGM.getTypes());
}
