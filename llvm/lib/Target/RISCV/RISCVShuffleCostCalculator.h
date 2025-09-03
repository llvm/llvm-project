// RISCVShuffleCostCalculator.h
#pragma once
#include "RISCVTypeHelper.h"

class RISCVShuffleCostCalculator {
private:
  struct ShuffleCostContext {
    const RISCVTTIImpl& TTI;
    TTI::ShuffleKind Kind;
    VectorType* DstTy;
    VectorType* SrcTy; 
    ArrayRef<int> Mask;
    TTI::TargetCostKind CostKind;
    int Index;
    VectorType* SubTp;
    const RISCVSubtarget* ST;
  };

  // 處理固定長度向量的 shuffle 成本
  static InstructionCost calculateFixedVectorShuffleCost(
      const ShuffleCostContext& Ctx) {
    
    auto* FVTp = dyn_cast<FixedVectorType>(Ctx.SrcTy);
    if (!FVTp || !Ctx.ST->hasVInstructions()) 
      return InstructionCost::getInvalid();
      
    auto LT = Ctx.TTI.getTypeLegalizationCost(Ctx.SrcTy);
    if (!LT.second.isFixedLengthVector())
      return InstructionCost::getInvalid();

    // 嘗試使用 VReg 分割成本計算
    InstructionCost VRegSplittingCost = costShuffleViaVRegSplitting(
        Ctx.TTI, LT.second, Ctx.ST->getRealVLen(),
        Ctx.Kind == TTI::SK_InsertSubvector ? Ctx.DstTy : Ctx.SrcTy, 
        Ctx.Mask, Ctx.CostKind);
        
    if (VRegSplittingCost.isValid())
      return VRegSplittingCost;

    // 根據 shuffle 種類計算成本
    switch (Ctx.Kind) {
    case TTI::SK_PermuteSingleSrc:
      return calculatePermuteSingleSrcCost(Ctx, FVTp, LT);
    case TTI::SK_Transpose:
    case TTI::SK_PermuteTwoSrc:
      return calculatePermuteTwoSrcCost(Ctx, FVTp, LT);
    default:
      break;
    }

    // 如果需要分割，計算分割成本
    if (shouldSplitShuffle(Ctx, LT)) {
      return costShuffleViaSplitting(Ctx.TTI, LT.second, FVTp, 
                                    Ctx.Mask, Ctx.CostKind);
    }

    return InstructionCost::getInvalid();
  }

  // 計算單來源排列成本
  static InstructionCost calculatePermuteSingleSrcCost(
      const ShuffleCostContext& Ctx,
      FixedVectorType* FVTp,
      std::pair<InstructionCost, MVT> LT) {
    
    if (Ctx.Mask.size() < 2)
      return InstructionCost::getInvalid();
      
    MVT EltTp = LT.second.getVectorElementType();
    
    // 處理交錯和去交錯模式
    if (EltTp.getScalarSizeInBits() < Ctx.ST->getELen()) {
      InstructionCost InterleaveCost = calculateInterleaveCost(Ctx, EltTp, LT);
      if (InterleaveCost.isValid())
        return InterleaveCost;
    }

    // 處理重複串接遮罩
    int SubVectorSize;
    if (isRepeatedConcatMask(Ctx.Mask, SubVectorSize)) {
      return calculateRepeatedConcatCost(Ctx, SubVectorSize, LT);
    }

    // 嘗試使用滑動成本
    InstructionCost SlideCost = calculateSlideCost(Ctx, FVTp);
    if (SlideCost.isValid())
      return SlideCost;

    // 使用 vrgather 指令
    return calculateVRGatherCost(Ctx, LT, /*TwoSrc=*/false);
  }

  // 計算交錯操作成本
  static InstructionCost calculateInterleaveCost(
      const ShuffleCostContext& Ctx,
      MVT EltTp,
      std::pair<InstructionCost, MVT> LT) {
    
    // 交錯模式：兩個向量交錯
    if (ShuffleVectorInst::isInterleaveMask(Ctx.Mask, 2, Ctx.Mask.size()))
      return 2 * LT.first * Ctx.TTI.getTLI()->getLMULCost(LT.second);

    // 去交錯模式  
    if (Ctx.Mask[0] == 0 || Ctx.Mask[0] == 1) {
      auto DeinterleaveMask = createStrideMask(Ctx.Mask[0], 2, Ctx.Mask.size());
      if (equal(DeinterleaveMask, Ctx.Mask))
        return LT.first * Ctx.TTI.getRISCVInstructionCost(
            RISCV::VNSRL_WI, LT.second, Ctx.CostKind);
    }
    
    return InstructionCost::getInvalid();
  }

  // 計算重複串接成本
  static InstructionCost calculateRepeatedConcatCost(
      const ShuffleCostContext& Ctx,
      int SubVectorSize,
      std::pair<InstructionCost, MVT> LT) {
    
    if (LT.second.getScalarSizeInBits() == 1)
      return InstructionCost::getInvalid();
      
    InstructionCost Cost = 0;
    unsigned NumSlides = Log2_32(Ctx.Mask.size() / SubVectorSize);
    
    for (unsigned I = 0; I != NumSlides; ++I) {
      unsigned InsertIndex = SubVectorSize * (1 << I);
      
      FixedVectorType* SubTp = FixedVectorType::get(
          Ctx.SrcTy->getElementType(), InsertIndex);
      FixedVectorType* DestTp = 
          FixedVectorType::getDoubleElementsVectorType(SubTp);
      
      auto DestLT = Ctx.TTI.getTypeLegalizationCost(DestTp);
      
      // 向量暫存器移動成本
      Cost += DestLT.first * Ctx.TTI.getTLI()->getLMULCost(DestLT.second);
      
      // 子向量插入成本
      Cost += Ctx.TTI.getShuffleCost(TTI::SK_InsertSubvector, DestTp, DestTp, 
                                     {}, Ctx.CostKind, InsertIndex, SubTp);
    }
    
    return Cost;
  }

  // 計算滑動成本
  static InstructionCost calculateSlideCost(
      const ShuffleCostContext& Ctx,
      FixedVectorType* FVTp) {
    
    return Ctx.TTI.getSlideCost(FVTp, Ctx.Mask, Ctx.CostKind);
  }

  // 計算 vrgather 成本
  static InstructionCost calculateVRGatherCost(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT,
      bool TwoSrc) {
    
    // 檢查是否適用 vrgather
    if (LT.first != 1 || 
        (LT.second.getScalarSizeInBits() == 8 && 
         LT.second.getVectorNumElements() > 256))
      return InstructionCost::getInvalid();

    VectorType* IdxTy = RISCVTypeHelper::getVRGatherIndexType(
        LT.second, *Ctx.ST, Ctx.SrcTy->getContext());
    InstructionCost IndexCost = Ctx.TTI.getConstantPoolLoadCost(IdxTy, Ctx.CostKind);

    if (!TwoSrc) {
      // 單來源 vrgather
      return IndexCost + Ctx.TTI.getRISCVInstructionCost(
          RISCV::VRGATHER_VV, LT.second, Ctx.CostKind);
    } else {
      // 雙來源 vrgather
      auto& C = Ctx.SrcTy->getContext();
      auto EC = Ctx.SrcTy->getElementCount();
      VectorType* MaskTy = VectorType::get(IntegerType::getInt1Ty(C), EC);
      InstructionCost MaskCost = Ctx.TTI.getConstantPoolLoadCost(MaskTy, Ctx.CostKind);
      
      return 2 * IndexCost + 
             Ctx.TTI.getRISCVInstructionCost({RISCV::VRGATHER_VV, RISCV::VRGATHER_VV},
                                            LT.second, Ctx.CostKind) +
             MaskCost;
    }
  }

  // 處理可縮放向量的 shuffle 成本
  static InstructionCost calculateScalableVectorShuffleCost(
      const ShuffleCostContext& Ctx) {
    
    auto LT = Ctx.TTI.getTypeLegalizationCost(Ctx.SrcTy);
    
    switch (Ctx.Kind) {
    case TTI::SK_ExtractSubvector:
      return calculateExtractSubvectorCost(Ctx, LT);
    case TTI::SK_InsertSubvector:
      return calculateInsertSubvectorCost(Ctx, LT);
    case TTI::SK_Select:
      return calculateSelectCost(Ctx, LT);
    case TTI::SK_Broadcast:
      return calculateBroadcastCost(Ctx, LT);
    case TTI::SK_Splice:
      return calculateSpliceCost(Ctx, LT);
    case TTI::SK_Reverse:
      return calculateReverseCost(Ctx, LT);
    default:
      break;
    }
    
    return InstructionCost::getInvalid();
  }

  // 各種特殊 shuffle 操作的成本計算...
  static InstructionCost calculateExtractSubvectorCost(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT) {
    
    // 從零索引提取總是子暫存器提取
    if (Ctx.Index == 0)
      return TTI::TCC_Free;

    // 檢查子暫存器邊界提取
    if (auto SubLT = Ctx.TTI.getTypeLegalizationCost(Ctx.SubTp);
        SubLT.second.isValid() && SubLT.second.isFixedLengthVector()) {
      
      if (auto VLen = Ctx.ST->getRealVLen();
          VLen && SubLT.second.getScalarSizeInBits() * Ctx.Index % *VLen == 0 &&
          SubLT.second.getSizeInBits() <= *VLen)
        return TTI::TCC_Free;
    }

    // 一般情況：使用 vslidedown
    return LT.first * Ctx.TTI.getRISCVInstructionCost(
        RISCV::VSLIDEDOWN_VI, LT.second, Ctx.CostKind);
  }

public:
  // 主要介面：計算 shuffle 成本
  static InstructionCost calculateShuffleCost(const ShuffleCostContext& Ctx) {
    // 驗證輸入
    assert((Ctx.Mask.empty() || Ctx.DstTy->isScalableTy() ||
            Ctx.Mask.size() == Ctx.DstTy->getElementCount().getKnownMinValue()) &&
           "Expected the Mask to match the return size if given");
    assert(Ctx.SrcTy->getScalarType() == Ctx.DstTy->getScalarType() &&
           "Expected the same scalar types");

    // 改善 shuffle 種類
    TTI::ShuffleKind ImprovedKind = improveShuffleKindFromMask(
        Ctx.Kind, Ctx.Mask, Ctx.SrcTy, Ctx.Index, Ctx.SubTp);
    
    ShuffleCostContext ImprovedCtx = Ctx;
    ImprovedCtx.Kind = ImprovedKind;

    // 先處理固定長度向量的特殊情況
    InstructionCost FixedCost = calculateFixedVectorShuffleCost(ImprovedCtx);
    if (FixedCost.isValid())
      return FixedCost;

    // 處理可縮放向量
    return calculateScalableVectorShuffleCost(ImprovedCtx);
  }
  
  // 檢查是否應該分割 shuffle
  static bool shouldSplitShuffle(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT) {
    
    auto shouldSplit = [](TTI::ShuffleKind Kind) {
      switch (Kind) {
      case TTI::SK_PermuteSingleSrc:
      case TTI::SK_Transpose:
      case TTI::SK_PermuteTwoSrc:
        return true;
      default:
        return false;
      }
    };

    return !Ctx.Mask.empty() && LT.first.isValid() && 
           LT.first != 1 && shouldSplit(Ctx.Kind);
  }

  // 計算廣播成本
  static InstructionCost calculateBroadcastCost(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT) {
    
    bool HasScalar = (Ctx.Args.size() > 0) && 
                     (Operator::getOpcode(Ctx.Args[0]) == Instruction::InsertElement);

    if (LT.second.getScalarSizeInBits() == 1) {
      return calculateMaskBroadcastCost(Ctx, LT, HasScalar);
    }

    if (HasScalar) {
      // 有純量來源：vmv.v.x v8, a0
      return LT.first * Ctx.TTI.getRISCVInstructionCost(
          RISCV::VMV_V_X, LT.second, Ctx.CostKind);
    }

    // 向量廣播：vrgather.vi v9, v8, 0
    return LT.first * Ctx.TTI.getRISCVInstructionCost(
        RISCV::VRGATHER_VI, LT.second, Ctx.CostKind);
  }

  // 計算遮罩廣播成本
  static InstructionCost calculateMaskBroadcastCost(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT,
      bool HasScalar) {
    
    if (HasScalar) {
      // 有純量來源的遮罩廣播
      return LT.first * (1 + Ctx.TTI.getRISCVInstructionCost(
          {RISCV::VMV_V_X, RISCV::VMSNE_VI}, LT.second, Ctx.CostKind));
    }

    // 向量來源的遮罩廣播（較複雜）
    return LT.first * (1 + Ctx.TTI.getRISCVInstructionCost(
        {RISCV::VMV_V_I, RISCV::VMERGE_VIM, RISCV::VMV_X_S, 
         RISCV::VMV_V_X, RISCV::VMSNE_VI}, LT.second, Ctx.CostKind));
  }

  // 計算反轉成本
  static InstructionCost calculateReverseCost(
      const ShuffleCostContext& Ctx,
      std::pair<InstructionCost, MVT> LT) {
    
    if (!LT.second.isVector())
      return InstructionCost::getInvalid();

    // 處理 i1 類型的特殊情況
    if (Ctx.SrcTy->getElementType()->isIntegerTy(1)) {
      return calculateI1ReverseCost(Ctx);
    }

    MVT ContainerVT = LT.second;
    if (LT.second.isFixedLengthVector())
      ContainerVT = Ctx.TTI.getTLI()->getContainerForFixedLengthVector(LT.second);
      
    MVT M1VT = RISCVTargetLowering::getM1VT(ContainerVT);
    
    if (ContainerVT.bitsLE(M1VT)) {
      return calculateSmallVectorReverseCost(Ctx, LT);
    } else {
      return calculateLargeVectorReverseCost(Ctx, LT, ContainerVT, M1VT);
    }
  }

public:
  // 工廠方法：建立 shuffle 成本計算器
  static InstructionCost getCost(
      const RISCVTTIImpl& TTI,
      TTI::ShuffleKind Kind, VectorType* DstTy, VectorType* SrcTy,
      ArrayRef<int> Mask, TTI::TargetCostKind CostKind,
      int Index, VectorType* SubTp, ArrayRef<const Value*> Args = {}) {
    
    ShuffleCostContext Ctx{
      TTI, Kind, DstTy, SrcTy, Mask, CostKind, 
      Index, SubTp, TTI.getST()
    };
    
    return calculateShuffleCost(Ctx);
  }
};

// 步驟 5：重構主要的 TTI 類別
// 現在我們可以大幅簡化原本的 RISCVTTIImpl 類別：

class RISCVTTIImpl : public BasicTTIImplBase<RISCVTTIImpl> {
private:
  const RISCVSubtarget *ST;
  const RISCVTargetLowering *TLI;
  
  // 成本計算器註冊表
  mutable std::unique_ptr<RISCVCostCalculatorRegistry> CostRegistry;
  
  // 延遲初始化成本計算器
  const RISCVCostCalculatorRegistry& getCostRegistry() const {
    if (!CostRegistry) {
      CostRegistry = std::make_unique<RISCVCostCalculatorRegistry>();
      CostRegistry->registerCalculators();
    }
    return *CostRegistry;
  }

public:
  // 重構後的主要成本計算函數
  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
      ArrayRef<const Value *> Args, const Instruction *CxtI) const {
    
    // 使用重構後的計算器
    RISCVArithmeticCostCalculator::ArithmeticCostContext Ctx{
      Opcode, Ty, CostKind, Op1Info, Op2Info, Args, CxtI, ST, TLI
    };
    
    InstructionCost Cost = RISCVArithmeticCostCalculator::calculateCost(Ctx);
    if (Cost.isValid())
      return Cost;
      
    // 回退到基礎實作
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, 
                                        Op1Info, Op2Info, Args, CxtI);
  }

  // 重構後的 shuffle 成本計算
  InstructionCost getShuffleCost(
      TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
      ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
      VectorType *SubTp, ArrayRef<const Value *> Args) const {
    
    // 使用重構後的 shuffle 計算器
    InstructionCost Cost = RISCVShuffleCostCalculator::getCost(
        *this, Kind, DstTy, SrcTy, Mask, CostKind, Index, SubTp, Args);
        
    if (Cost.isValid())
      return Cost;
      
    // 回退到基礎實作
    return BaseT::getShuffleCost(Kind, DstTy, SrcTy, Mask, CostKind, 
                                Index, SubTp);
  }

  // 重構後的立即數成本計算
  InstructionCost getIntImmCostInst(
      unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty,
      TTI::TargetCostKind CostKind, Instruction *Inst) const {
    
    RISCVImmediateCostCalculator::ImmediateCostContext Ctx{
      Opcode, Idx, Imm, Ty, CostKind, ST, getDataLayout(), Inst
    };
    
    return RISCVImmediateCostCalculator::calculateCost(Ctx);
  }
};
