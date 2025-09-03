// RISCVCostValidation.h - 成本計算驗證工具
#pragma once

class RISCVCostValidation {
public:
  // 驗證基本前置條件
  static bool validateBasicPreconditions(Type* Ty, 
                                        TTI::TargetCostKind CostKind,
                                        const RISCVSubtarget* ST) {
    // 檢查類型有效性
    if (!RISCVTypeHelper::isValidVectorType(Ty, ST))
      return false;
      
    // 檢查成本種類
    if (CostKind != TTI::TCK_RecipThroughput && 
        CostKind != TTI::TCK_Latency &&
        CostKind != TTI::TCK_CodeSize)
      return false;
      
    return true;
  }
  
  // 驗證向量運算前置條件
  static bool validateVectorOperation(VectorType* Ty, 
                                     const RISCVSubtarget* ST) {
    if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
      return false;
      
    if (Ty->getScalarSizeInBits() > ST->getELen())
      return false;
      
    return true;
  }
};

// 使用範例：重構後的函數片段
namespace RISCVTTIImpl {
  
// 重構後的 getEstimatedVLFor
unsigned getEstimatedVLFor(VectorType *Ty) const {
  if (isa<ScalableVectorType>(Ty)) {
    const unsigned EltSize = DL.getTypeSizeInBits(Ty->getElementType());
    const unsigned MinSize = DL.getTypeSizeInBits(Ty).getKnownMinValue();
    
    // 使用安全的 VScale 取得方法
    const unsigned VectorBits = 
        RISCVTypeHelper::getSafeVScaleForTuning(getVScaleForTuning()) * 
        RISCV::RVVBitsPerBlock;
        
    return RISCVTargetLowering::computeVLMAX(VectorBits, EltSize, MinSize);
  }
  return cast<FixedVectorType>(Ty)->getNumElements();
}

}
