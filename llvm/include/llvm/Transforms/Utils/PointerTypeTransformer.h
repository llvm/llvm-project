#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETRANSFORMER_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETRANSFORMER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/PointerTypePrinter.h"
#include "llvm/Transforms/Utils/PointerTypeInFunction.h"

namespace llvm {
    class PointerTypeTransformerPass
        : public PassInfoMixin<PointerTypeTransformerPass> {
    public:
      PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
    };
}

#endif

//  全部在Module层面处理，定义一个辅助类存储中间结果
//  第一阶段：获取全局变量、结构体初始化、构建函数调用关系图
//  第二阶段：处理函数内的类型推断，由于Value的生命周期等于Module，可以将结果直接存储于全局辅助类
//  第三阶段：依据DFA处理函数调用以及全局变量
//  只需要一个ModulePass和若干辅助类，不需要定义成员变量，辅助类作为局部变量声明于run方法中，通过参数传递