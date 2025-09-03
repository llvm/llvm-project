// RISCVCostCalculatorRegistry.h - 成本計算器註冊表
class RISCVCostCalculatorRegistry {
private:
  using CostCalculatorFunc = std::function<InstructionCost(const auto&)>;
  std::unordered_map<std::string, CostCalculatorFunc> calculators;

public:
  // 註冊不同類型的成本計算器
  void registerCalculators() {
    calculators["arithmetic"] = [](const auto& ctx) {
      return RISCVArithmeticCostCalculator::calculateCost(ctx);
    };
    
    calculators["intrinsic"] = [](const auto& ctx) {
      return RISCVIntrinsicCostCalculator::calculateIntrinsicCost(ctx);
    };
    
    calculators["immediate"] = [](const auto& ctx) {
      return RISCVImmediateCostCalculator::calculateCost(ctx);
    };
  }
  
  InstructionCost calculateCost(const std::string& type, const auto& context) {
    auto it = calculators.find(type);
    if (it != calculators.end())
      return it->second(context);
    return InstructionCost::getInvalid();
  }
};
