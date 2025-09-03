// RISCVArithmeticCostCalculator.h
#pragma once
#include "RISCVTypeHelper.h"
#include "RISCVCostConstants.h"

class RISCVArithmeticCostCalculator {
private:
  struct ArithmeticCostContext {
    unsigned Opcode;
    Type* Ty;
    TTI::TargetCostKind CostKind;
    TTI::OperandValueInfo Op1Info;
    TTI::OperandValueInfo Op2Info;
    ArrayRef<const Value*> Args;
    const Instruction* CxtI;
    const RISCVSubtarget* ST;
    const RISCVTargetLowering* TLI;
  };

  // 取得正確的 RVV 指令操作碼（修正原本的錯誤）
  static unsigned getRVVOpcode(unsigned ISDOpcode, Type* Ty) {
    switch (ISDOpcode) {
    case ISD::ADD:
    case ISD::SUB:
      return RISCV::VADD_VV;
      
    // 修正：位移指令應該對應到正確的操作碼
    case ISD::SHL:
      return RISCV::VSLL_VV;
    case ISD::SRL:
      return RISCV::VSRL_VV;  // 邏輯右移
    case ISD::SRA:
      return RISCV::VSRA_VV;  // 算術右移
      
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      // i1 類型使用遮罩指令，其他使用一般向量指令
      return (Ty->getScalarSizeInBits() == 1) ? RISCV::VMAND_MM : RISCV::VAND_VV;
      
    case ISD::MUL:
    case ISD::MULHS:
    case ISD::MULHU:
      return RISCV::VMUL_VV;
      
    case ISD::SDIV:
    case ISD::UDIV:
      return RISCV::VDIV_VV;
      
    case ISD::SREM:
    case ISD::UREM:
      return RISCV::VREM_VV;
      
    case ISD::FADD:
    case ISD::FSUB:
      return RISCV::VFADD_VV;
      
    case ISD::FMUL:
      return RISCV::VFMUL_VV;
      
    case ISD::FDIV:
      return RISCV::VFDIV_VV;
      
    case ISD::FNEG:
      return RISCV::VFSGNJN_VV;
      
    default:
      return 0; // 表示需要使用預設處理
    }
  }

  // 處理類型提升（如 f16 到 f32）
  static std::pair<InstructionCost, MVT> handleTypePromotion(
      const ArithmeticCostContext& Ctx,
      std::pair<InstructionCost, MVT>& LT) {
    
    InstructionCost CastCost = 0;
    
    // f16 with zvfhmin and bf16 will be promoted to f32
    if ((LT.second.getVectorElementType() == MVT::f16 ||
         LT.second.getVectorElementType() == MVT::bf16) &&
        Ctx.TLI->getOperationAction(Ctx.TLI->InstructionOpcodeToISD(Ctx.Opcode), 
                                   LT.second) == TargetLoweringBase::LegalizeAction::Promote) {
      
      MVT PromotedVT = Ctx.TLI->getTypeToPromoteTo(
          Ctx.TLI->InstructionOpcodeToISD(Ctx.Opcode), LT.second);
      Type* PromotedTy = EVT(PromotedVT).getTypeForEVT(Ctx.Ty->getContext());
      Type* LegalTy = EVT(LT.second).getTypeForEVT(Ctx.Ty->getContext());
      
      // 計算擴展參數的成本
      CastCost += LT.first * Ctx.Args.size() *
                  getCastInstrCost(Instruction::FPExt, PromotedTy, LegalTy,
                                  TTI::CastContextHint::None, Ctx.CostKind);
      
      // 計算截斷結果的成本
      CastCost += LT.first * 
                  getCastInstrCost(Instruction::FPTrunc, LegalTy, PromotedTy,
                                  TTI::CastContextHint::None, Ctx.CostKind);
      
      // 更新為提升後的類型
      LT.second = PromotedVT;
    }
    
    return {CastCost, LT.second};
  }

  // 計算常數具體化成本
  static InstructionCost calculateConstantMaterializationCost(
      const ArithmeticCostContext& Ctx) {
    
    auto getConstantMatCost = [&](unsigned Operand, 
                                 TTI::OperandValueInfo OpInfo) -> InstructionCost {
      if (OpInfo.isUniform() && canSplatOperand(Ctx.Opcode, Operand)) {
        // 兩種情況：
        // * 有 5 位元立即數可以展開
        // * 有較大的立即數必須在純量暫存器中具體化
        // 我們都回傳 0，因為目前忽略在 GPR 中具體化純量常數的成本
        return 0;
      }

      return getConstantPoolLoadCost(Ctx.Ty, Ctx.CostKind);
    };

    InstructionCost ConstantMatCost = 0;
    if (Ctx.Op1Info.isConstant())
      ConstantMatCost += getConstantMatCost(0, Ctx.Op1Info);
    if (Ctx.Op2Info.isConstant())
      ConstantMatCost += getConstantMatCost(1, Ctx.Op2Info);
      
    return ConstantMatCost;
  }

public:
  // 主要的算術指令成本計算函數（重構版本）
  static InstructionCost calculateCost(const ArithmeticCostContext& Ctx) {
    // 早期退出條件檢查
    if (Ctx.CostKind != TTI::TCK_RecipThroughput)
      return BaseT::getArithmeticInstrCost(Ctx.Opcode, Ctx.Ty, Ctx.CostKind, 
                                          Ctx.Op1Info, Ctx.Op2Info, 
                                          Ctx.Args, Ctx.CxtI);

    // 檢查是否應該使用 RVV
    if (!RISCVTypeHelper::shouldUseRVV(Ctx.Ty, Ctx.ST))
      return BaseT::getArithmeticInstrCost(Ctx.Opcode, Ctx.Ty, Ctx.CostKind,
                                          Ctx.Op1Info, Ctx.Op2Info, 
                                          Ctx.Args, Ctx.CxtI);

    // 驗證向量類型
    if (!RISCVTypeHelper::isValidVectorType(Ctx.Ty, Ctx.ST))
      return BaseT::getArithmeticInstrCost(Ctx.Opcode, Ctx.Ty, Ctx.CostKind,
                                          Ctx.Op1Info, Ctx.Op2Info, 
                                          Ctx.Args, Ctx.CxtI);

    // 合法化類型
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ctx.Ty);

    // 純量類型回退到基礎實作
    if (!LT.second.isVector())
      return BaseT::getArithmeticInstrCost(Ctx.Opcode, Ctx.Ty, Ctx.CostKind,
                                          Ctx.Op1Info, Ctx.Op2Info, 
                                          Ctx.Args, Ctx.CxtI);

    // 處理類型提升
    auto [CastCost, FinalMVT] = handleTypePromotion(Ctx, LT);

    // 計算常數具體化成本
    InstructionCost ConstantMatCost = calculateConstantMaterializationCost(Ctx);

    // 取得對應的 RVV 指令
    unsigned ISDOpcode = Ctx.TLI->InstructionOpcodeToISD(Ctx.Opcode);
    unsigned Op = getRVVOpcode(ISDOpcode, Ctx.Ty);
    
    if (Op == 0) {
      // 沒有對應的 RVV 指令，使用預設處理
      return CastCost + ConstantMatCost +
             BaseT::getArithmeticInstrCost(Ctx.Opcode, Ctx.Ty, Ctx.CostKind,
                                          Ctx.Op1Info, Ctx.Op2Info, 
                                          Ctx.Args, Ctx.CxtI);
    }

    // 計算指令成本
    InstructionCost InstrCost = getRISCVInstructionCost(Op, FinalMVT, Ctx.CostKind);
    
    // 浮點運算成本是整數運算的兩倍
    if (Ctx.Ty->isFPOrFPVectorTy())
      InstrCost *= RISCVCostConstants::FP_COST_MULTIPLIER;
      
    return CastCost + ConstantMatCost + LT.first * InstrCost;
  }

  // 檢查是否可以展開操作數
  static bool canSplatOperand(unsigned Opcode, int Operand) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::ICmp:
    case Instruction::FCmp:
      return true;  // 這些指令支援任一操作數展開
      
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::Select:
      return Operand == 1;  // 只有第二個操作數可以展開
      
    default:
      return false;
    }
  }
};

// RISCVIntrinsicCostCalculator.h - 內建函數成本計算
class RISCVIntrinsicCostCalculator {
private:
  // 修正飽和運算指令對應
  static unsigned getSaturatedArithmeticOpcode(Intrinsic::ID IID) {
    switch (IID) {
    case Intrinsic::sadd_sat:
      return RISCV::VSADD_VV;
    case Intrinsic::ssub_sat:
      return RISCV::VSSUB_VV;    // 修正：原本錯誤用了 VSSUBU_VV
    case Intrinsic::uadd_sat:
      return RISCV::VSADDU_VV;
    case Intrinsic::usub_sat:
      return RISCV::VSSUBU_VV;
    default:
      return 0;
    }
  }
  
  // 取得最大最小值指令對應
  static std::pair<unsigned, unsigned> getMinMaxOpcodes(Intrinsic::ID IID) {
    struct OpcodePair { unsigned VectorOp; unsigned ReductionOp; };
    
    switch (IID) {
    case Intrinsic::smax:
      return {RISCV::VMAX_VV, RISCV::VREDMAX_VS};
    case Intrinsic::smin:  
      return {RISCV::VMIN_VV, RISCV::VREDMIN_VS};
    case Intrinsic::umax:
      return {RISCV::VMAXU_VV, RISCV::VREDMAXU_VS};
    case Intrinsic::umin:
      return {RISCV::VMINU_VV, RISCV::VREDMINU_VS};
    case Intrinsic::maxnum:
      return {RISCV::VFMAX_VV, RISCV::VFREDMAX_VS};
    case Intrinsic::minnum:
      return {RISCV::VFMIN_VV, RISCV::VFREDMIN_VS};
    default:
      return {0, 0};
    }
  }

public:
  // 計算內建函數成本
  static InstructionCost calculateIntrinsicCost(
      const IntrinsicCostAttributes& ICA,
      TTI::TargetCostKind CostKind,
      const RISCVSubtarget* ST,
      const RISCVTargetLowering* TLI) {
    
    auto* RetTy = ICA.getReturnType();
    auto LT = getTypeLegalizationCost(RetTy);
    
    switch (ICA.getID()) {
    case Intrinsic::sadd_sat:
    case Intrinsic::ssub_sat:
    case Intrinsic::uadd_sat:
    case Intrinsic::usub_sat: {
      if (ST->hasVInstructions() && LT.second.isVector()) {
        unsigned Op = getSaturatedArithmeticOpcode(ICA.getID());
        if (Op != 0)
          return LT.first * getRISCVInstructionCost(Op, LT.second, CostKind);
      }
      break;
    }
    
    case Intrinsic::umin:
    case Intrinsic::umax:
    case Intrinsic::smin:
    case Intrinsic::smax: {
      // 純量情況：使用 Zbb 擴展
      if (LT.second.isScalarInteger() && ST->hasStdExtZbb())
        return LT.first;

      // 向量情況：使用 RVV 指令
      if (ST->hasVInstructions() && LT.second.isVector()) {
        auto [VectorOp, _] = getMinMaxOpcodes(ICA.getID());
        return LT.first * getRISCVInstructionCost(VectorOp, LT.second, CostKind);
      }
      break;
    }
    
    case Intrinsic::abs: {
      if (ST->hasVInstructions() && LT.second.isVector()) {
        // vrsub.vi v10, v8, 0
        // vmax.vv v8, v8, v10
        return LT.first * getRISCVInstructionCost({RISCV::VRSUB_VI, RISCV::VMAX_VV},
                                                  LT.second, CostKind);
      }
      break;
    }
    
    default:
      break;
    }
    
    return InstructionCost::getInvalid(); // 表示需要使用預設處理
  }
};
