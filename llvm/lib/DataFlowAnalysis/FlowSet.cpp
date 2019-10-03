#include "llvm/DataFlowAnalysis/FlowSet.h"

using namespace llvm;

template <typename T>
void FlowSet<T>::push_back(T data)
{
    Node* pTemp = getNode(data);
    if(!getRootNode())
    {
        getRootNode() = pTemp;
    }
    else
    {
        Node* pC = getRootNode();
        while(pC->pNext)
        {
            pC = pC->pNext;
        }
        pC->pNext = pTemp;
    }
}


template class llvm::FlowSet<int>;
