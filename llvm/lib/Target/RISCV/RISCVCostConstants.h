// RISCVCostConstants.h - 常數定義
#pragma once

namespace RISCVCostConstants {
  // VScale 相關常數
  static constexpr unsigned DEFAULT_VSCALE_FALLBACK = 1;
  
  // 立即數相關常數
  static constexpr uint64_t ZEXT_H_MASK = 0xffff;
  static constexpr uint64_t ZEXT_W_MASK = 0xffffffff;
  static constexpr unsigned MAX_5BIT_UNSIGNED = 31;
  
  // 記憶體操作成本
  static constexpr unsigned CONSTANT_POOL_ADDRESS_COST = 2;
  
  // 向量操作相關
  static constexpr unsigned BASIC_VECTOR_OP_COST = 1;
  static constexpr unsigned SLIDE_INDEX_COST = 2;
  
  // 浮點運算倍數（相對於整數運算）
  static constexpr unsigned FP_COST_MULTIPLIER = 2;
}

// RISCVTypeHelper.h - 類型檢查和轉換輔助工具
#pragma once
#include "RISCVSubtarget.h"
#include "llvm/IR/Type.h"
#include "llvm/CodeGen/ValueTypes.h"

class RISCVTypeHelper {
public:
  // 安全的 VScale 取得
  static unsigned getSafeVScaleForTuning(
      std::optional<unsigned> VScaleOpt) {
    return VScaleOpt.value_or(RISCVCostConstants::DEFAULT_VSCALE_FALLBACK);
  }
  
  // 檢查是否為有效的向量類型
  static bool isValidVectorType(Type* Ty, const RISCVSubtarget* ST) {
    if (!Ty->isVectorTy())
      return false;
    
    // 檢查純量大小是否超過 ELEN
    if (Ty->getScalarSizeInBits() > ST->getELen())
      return false;
      
    return true;
  }
  
  // 檢查是否應該使用 RVV
  static bool shouldUseRVV(Type* Ty, const RISCVSubtarget* ST) {
    if (!ST->hasVInstructions())
      return false;
      
    if (auto* FVTy = dyn_cast<FixedVectorType>(Ty))
      return ST->useRVVForFixedLengthVectors();
      
    return true; // 對可縮放向量總是使用 RVV
  }
  
  // 檢查 LMUL 大小
  static bool isM1OrSmaller(MVT VT) {
    RISCVVType::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
    return (LMUL == RISCVVType::VLMUL::LMUL_F8 ||
            LMUL == RISCVVType::VLMUL::LMUL_F4 ||
            LMUL == RISCVVType::VLMUL::LMUL_F2 ||
            LMUL == RISCVVType::VLMUL::LMUL_1);
  }
  
  // 取得向量索引類型
  static VectorType* getVRGatherIndexType(MVT DataVT, 
                                          const RISCVSubtarget& ST,
                                          LLVMContext& C) {
    assert((DataVT.getScalarSizeInBits() != 8 ||
            DataVT.getVectorNumElements() <= 256) && 
           "unhandled case in lowering");
    
    MVT IndexVT = DataVT.changeTypeToInteger();
    if (IndexVT.getScalarType().bitsGT(ST.getXLenVT()))
      IndexVT = IndexVT.changeVectorElementType(MVT::i16);
      
    return cast<VectorType>(EVT(IndexVT).getTypeForEVT(C));
  }
};
